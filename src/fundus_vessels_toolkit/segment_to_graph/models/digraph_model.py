from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch import nn
from torch_geometric.data import Batch as PyGBatch
from torch_geometric.nn.conv import GATv2Conv, MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import group_argsort, softmax
from torchvision.models import EfficientNet_V2_S_Weights
from torchvision.models.efficientnet import efficientnet_v2_s
from torchvision.transforms.functional import normalize

from ...utils.torch import torch_interp_bilinear
from .dataset import VBranchDigraphBatch, VBranchDigraphData


class BranchFeaturesEfficientNetV2S(torch.nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        net = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT if pretrained else None).features
        self.efficient_net = net
        self._features_cache = [None] * 4

        self.net = nn.Sequential(*[nn.Sequential(*[net[_] for _ in idxs]) for idxs in [(0, 1), (2,), (3,), (4, 5, 6)]])

    def forward(self, x):
        # Normalize x to ImageNet stats
        x = normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        features = []
        for f in self.net:
            x = f(x)
            features.append(x)
        return features


class Gatv2GCN(torch.nn.Module):
    def __init__(self, n_in: int = 512, n_out: int = 512):
        super().__init__()
        self.n_in = n_in
        self.n_out = 256 * 4  # n_out

        self.bn0 = pyg_nn.InstanceNorm(n_in)
        self.gat0 = GATv2Conv(n_in, 64, heads=8, dropout=0.1)
        self.bn1 = pyg_nn.InstanceNorm(64 * 8)
        self.gat1a = GATv2Conv(64 * 8, 128, heads=8, dropout=0.1)
        self.gat1b = GATv2Conv(128 * 8, 256, heads=4, dropout=0.1)
        self.bn2 = pyg_nn.InstanceNorm(256 * 4)
        self.gat2a = GATv2Conv(256 * 4, 128, heads=8, dropout=0.1)
        self.gat2b = GATv2Conv(128 * 8, 256, heads=4, dropout=0.1)
        self.bn3 = pyg_nn.InstanceNorm(256 * 4)
        self.gat3a = GATv2Conv(256 * 4, 512, heads=2, dropout=0.1)
        self.gat3b = GATv2Conv(512 * 2, n_out, heads=1, dropout=0.1)

    def forward(self, x, edge_index, batch, batch_size):
        x = self.bn0(x, batch, batch_size)
        x = self.gat0(x, edge_index).relu()
        x = self.bn1(x, batch, batch_size)
        x = self.gat1a(x, edge_index).relu()
        x = self.gat1b(x, edge_index).relu()
        # x = self.bn2(x, batch, batch_size)
        # x = self.gat2a(x, edge_index).relu()
        # x = self.gat2b(x, edge_index).relu()
        # x = self.bn3(x, batch, batch_size)
        # x = self.gat3a(x, edge_index).relu()
        # x = self.gat3b(x, edge_index)

        return x


class BranchDigraphModel(torch.nn.Module):
    def __init__(self, img_feature_extractor: nn.Module, gnn: nn.Module, gnn_out_channels: Optional[int] = None):
        super().__init__()

        self.img_feature_extractor = img_feature_extractor.to(dtype=torch.bfloat16)
        self.gnn = gnn

        if gnn_out_channels is None:
            assert hasattr(gnn, "n_out"), "gnn_out_channels must be specified if gnn does not have n_out attribute"
            gnn_out_channels = int(gnn.n_out)  # type: ignore

        # === Classification layers ===
        self.lin_fp = Linear(gnn_out_channels, 1, weight_initializer="glorot")
        self.lin_av = Linear(gnn_out_channels, 1, weight_initializer="glorot")
        self.lin_dir = Linear(gnn_out_channels, 1, weight_initializer="glorot")
        self.lin_affinity = Linear(gnn_out_channels, 128, weight_initializer="glorot")
        self.lin_root = Linear(gnn_out_channels, 1, weight_initializer="glorot")

    def sample_features(
        self,
        features_map: torch.Tensor | list[torch.Tensor],
        branch_curves: torch.Tensor,
        batch_idx: torch.Tensor,
        img_shape: tuple[int, int],
    ) -> torch.Tensor:
        """
        Sample features along branch curves from a feature map.

        Parameters
        ----------
            features_map: torch.Tensor (B, C_feature, H, W)
                tensor of feature maps
            branch_curves: torch.Tensor (N_branch, L, 2)
                tensor of branch curves, where L is the number of curve points
            batch_idx: torch.Tensor (N_branch,)
                tensor of batch indices for each branch
        Returns
        -------
            torch.Tensor (N_branch, C_feature)
                tensor of sampled branch features
        """
        C = branch_curves.shape[1]
        curves_y_tip0, curves_x_tip0 = branch_curves[:, : C // 2].int().unbind(-1)
        curves_y_tip1, curves_x_tip1 = branch_curves[:, C // 2 :].int().unbind(-1)
        batch_idx = batch_idx[:, None]

        if isinstance(features_map, list):
            b_features0, b_features1 = [], []

            for fmap in features_map:
                fmap_shape = fmap.shape[-2:]
                if fmap_shape == img_shape:
                    assert fmap_shape[-2] > curves_y_tip0.max() and fmap_shape[-1] > curves_x_tip0.max(), (
                        f"Feature map shape {fmap_shape} is smaller than max curve coordinates {(curves_y_tip0.max(), curves_x_tip0.max())}"
                    )
                    b_features0.append(fmap[batch_idx, :, curves_y_tip0, curves_x_tip0].mean(dim=-2))
                    b_features1.append(fmap[batch_idx, :, curves_y_tip1, curves_x_tip1].mean(dim=-2))
                else:
                    Y, X = fmap_shape
                    scale_y = img_shape[0] / Y
                    scale_x = img_shape[1] / X
                    y_tip0, y_tip1 = curves_y_tip0 / scale_y, curves_y_tip1 / scale_y
                    x_tip0, x_tip1 = curves_x_tip0 / scale_x, curves_x_tip1 / scale_x

                    b_features0.append(torch_interp_bilinear(fmap, y_tip0, x_tip0, batch_idx).mean(dim=-2))
                    b_features1.append(torch_interp_bilinear(fmap, y_tip1, x_tip1, batch_idx).mean(dim=-2))

            return torch.cat(b_features0 + b_features1, dim=-1)
        else:
            b_features0 = features_map[batch_idx, :, curves_y_tip0, curves_x_tip0].mean(dim=-2)
            b_features1 = features_map[batch_idx, :, curves_y_tip1, curves_x_tip1].mean(dim=-2)
            return torch.cat([b_features0, b_features1], dim=-1)  # (B, 512)

    def forward(self, data) -> Output:
        device = data.edge_index.device

        # === Extract branch features ===
        img = data.img.reshape(data.batch_size, 3, *data.img.shape[1:]).to(dtype=torch.bfloat16)
        img_features = self.img_feature_extractor(img)
        branch_features = self.sample_features(img_features, data.branch_curves, data.batch, img.shape[-2:])

        # === Refine branch representation with the GNN ===
        lines = Lines(data.edge_index, data.edge_first_tip)
        x = self.gnn(branch_features, lines.edge_index, data.batch, data.batch_size)

        # === Predict branch AV class, direction, affinity and root probability ===
        branch_fp = self.lin_fp(x).squeeze(-1)
        branch_av = self.lin_av(x).squeeze(-1)
        branch_dir = self.lin_dir(x).squeeze(-1)
        branch_affinity_v = self.lin_affinity(x)
        branch_root_score = self.lin_root(x).squeeze(-1)

        # === Based on the branch direction, predict the optimal incident parent ===
        # Select only lines whose direction is consistent with the predicted branch direction
        dir = (data.branch_dir if self.training else branch_dir) > 0
        valid_lines = lines.select_valid_b1_dir(dir)

        # Compute affinity between connected branches
        b0_v = branch_affinity_v[valid_lines.b0]
        b1_v = branch_affinity_v[valid_lines.b1]
        edge_scores = F.cosine_similarity(b0_v, b1_v, dim=-1)

        # Add root affinity
        valid_root_mask = data.branch_root_candidates.clone()
        valid_root_mask[torch.arange(data.num_nodes, device=device), torch.where(dir, 1, 0)] = False
        valid_root_b1 = torch.where(valid_root_mask.any(dim=1))[0]

        edge_scores = torch.cat([edge_scores, branch_root_score[valid_root_b1]], dim=0)

        assert valid_lines.whole_mask is not None, "valid_lines must have whole_mask for correct output"
        return BranchDigraphModel.Output(
            data, branch_fp, branch_av, branch_dir, edge_scores, valid_lines.whole_mask, valid_root_mask
        )

    class Output(NamedTuple):
        batch: VBranchDigraphBatch | VBranchDigraphData
        """Batch data passed to the model, used for convenience to compute losses and metrics"""

        fp_logit: torch.Tensor
        """Tensor of shape (N_branch,) containing the logits for each branch being a false positive (i.e., not corresponding to any GT branch)"""  # noqa: E501

        av_logit: torch.Tensor
        """Tensor of shape (N_branch,) containing the logits for each branch class (1: artery, 0: vein)"""  # noqa: E501

        dir_logit: torch.Tensor
        """Tensor of shape (N_branch,) containing logits for each branch direction (positive: branch is oriented from tip0 to tip1, negative: branch is oriented from tip1 to tip0)"""  # noqa: E501

        lines_logit: torch.Tensor
        """Tensor of shape (N_valid_edges + N_valid_roots,) containing affinity scores for each valid incident edge (from parent to child branch) and for each valid root assignment (from root to child branch)"""  # noqa: E501

        edge_mask: torch.Tensor
        """Boolean mask of shape (N_edges,) indicating which lines are valid for incident edge prediction (i.e., have consistent direction with the predicted branch direction)"""  # noqa: E501

        root_mask: torch.Tensor
        """Boolean mask of shape (N_branch, 2) indicating which branches are valid for root assignment according to the predicted branch direction (i.e., have at least one valid root direction). This tensor contains N_valid_roots True elements."""  # noqa: E501

        def lines_gt_score(self):
            return torch.cat([self.batch.edge_p[self.edge_mask], self.batch.branch_root_p[self.root_mask]], dim=0)

        def b1(self):
            edge_b1 = self.batch.edge_index[1, self.edge_mask]
            root_b1 = torch.where(self.root_mask.any(dim=1))[0]
            return torch.cat([edge_b1, root_b1], dim=0)

        def optimal_parent(self):
            lines = Lines(self.batch.edge_index, self.batch.edge_first_tip).select(self.edge_mask)
            return optimal_parent(self.lines_logit, lines, self.root_mask)

        def optimal_parent_gt(self):
            lines = Lines(self.batch.edge_index, self.batch.edge_first_tip).select(self.edge_mask)
            return optimal_parent(self.lines_gt_score(), lines, self.root_mask)

        @property
        def fp_p(self):
            return torch.sigmoid(self.fp_logit)

        @property
        def av_p(self):
            return torch.sigmoid(self.av_logit)

        @property
        def dir_p(self):
            return torch.sigmoid(self.dir_logit)

        def unbatch(self) -> list[BranchDigraphModel.Output]:
            assert isinstance(self.batch, PyGBatch), "Batch data must be a torch geometric Batch for unbatching"
            outputs = []

            lines_b1 = self.b1()

            for idx, single_data in enumerate(self.batch.to_data_list()):
                branch_mask = self.batch.batch == idx
                edge_idx_mask = branch_mask[self.batch.edge_index[1]]

                outputs.append(
                    BranchDigraphModel.Output(
                        batch=single_data,  # type: ignore
                        fp_logit=self.fp_logit[branch_mask],
                        av_logit=self.av_logit[branch_mask],
                        dir_logit=self.dir_logit[branch_mask],
                        lines_logit=self.lines_logit[branch_mask[lines_b1]],
                        edge_mask=self.edge_mask[edge_idx_mask],
                        root_mask=self.root_mask[branch_mask],
                    )
                )
            return outputs


class Lines:
    def __init__(
        self,
        edge_index: torch.Tensor,
        edge_first_tip: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        whole_mask: Optional[torch.Tensor] = None,
    ):
        self.edge_index = edge_index
        self.edge_first_tip = edge_first_tip
        self.mask = mask
        self.whole_mask = whole_mask

    def __bool__(self):
        return self.edge_index.shape[1] > 0

    @property
    def n_lines(self):
        return self.edge_index.shape[1]

    @property
    def b0(self):
        return self.edge_index[0]

    @property
    def b1(self):
        return self.edge_index[1]

    @property
    def tip0(self):
        return self.edge_first_tip[:, 0]

    @property
    def tip1(self):
        return self.edge_first_tip[:, 1]

    def select(self, mask):
        if self.whole_mask is None:
            whole_mask = mask
        else:
            whole_mask = self.whole_mask.clone()
            whole_mask[self.whole_mask] = mask
        return Lines(self.edge_index[:, mask], self.edge_first_tip[mask], mask=mask, whole_mask=whole_mask)

    def select_valid_b1_dir(self, b_dir) -> Lines:
        return self.select(self.b1_dir_mask(b_dir))

    def b1_dir_mask(self, b_dir):
        return self.tip1 == b_dir[self.b1]

    def groupby_incident_edges(self, select_b1_dir=None):
        if select_b1_dir is None:
            return self.b1
        else:
            b1 = self.b1
            return (b1 + 1) * (self.tip1 == select_b1_dir[b1])

    def sort_b1(self) -> torch.Tensor:
        sort_idx = torch.argsort(self.b1)
        self.edge_index = self.edge_index[:, sort_idx]
        self.edge_first_tip = self.edge_first_tip[sort_idx]
        return sort_idx


def root_lines_mask(data):
    return data.edge_index[0] == -1


def root_lines_index(data, dir):
    b0 = data.edge_index[0]
    b1 = data.edge_index[1]
    tip1 = data.edge_first_tip[:, 1]

    mask = (b0 == -1) & (tip1 == dir[b1])
    return mask, b1[mask]


def groupby_incident_edges(data, b_dir=None):
    b1 = data.edge_index[1]
    tip1 = data.edge_first_tip[:, 1]

    if b_dir is None:
        return b1
    else:
        return (b1 + 1) * (tip1 == b_dir[b1])


def incident_mask(data, branch_dir):
    edge_index = data.edge_index
    edge_first_tip = data.edge_first_tip

    edge_mask = ~root_lines_mask(data)
    edge_index = edge_index[:, edge_mask]
    edge_first_tip = edge_first_tip[edge_mask]

    source_branches = edge_index[0]
    target_branches = edge_index[1]

    source_first_tip = edge_first_tip[:, 0]
    target_first_tip = edge_first_tip[:, 1]

    source_dirs = branch_dir[source_branches]
    target_dirs = branch_dir[target_branches]

    source_out = source_first_tip ^ source_dirs  # if dir is True, the edge should be emitted from the second tip
    target_in = target_first_tip == target_dirs  # if dir is True, the edge should be incident to the first tip

    edge_mask[edge_mask.clone()] = source_out & target_in
    return edge_mask


class BranchDigraphGATv2Conv(MessagePassing):
    def __init__(
        self,
        in_branch_channels: int,
        out_branch_channels: int,
        in_tip_channels: int,
        out_tip_channels: int,
        *,
        heads: int = 1,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__(node_dim=0, aggr="add")

        self.in_branch_channels = in_branch_channels
        self.out_branch_channels = out_branch_channels
        self.in_tip_channels = in_tip_channels
        self.out_tip_channels = out_tip_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        f_Bin, f_Bout = in_branch_channels, out_branch_channels
        f_Tin, f_Tout = in_tip_channels, out_tip_channels

        self.lin_B = Linear(f_Bin, heads * f_Bout, bias=bias, weight_initializer="glorot")
        self.lin_B_self = Linear(f_Bin + f_Tin, heads * f_Bout, bias=bias, weight_initializer="glorot")

        self.lin_T_source = Linear(f_Bin + f_Tin, heads * f_Tout, bias=bias, weight_initializer="glorot")
        self.lin_T_target = Linear(f_Bin + f_Tin, heads * f_Tout, bias=bias, weight_initializer="glorot")
        self.lin_T_self = Linear(f_Tin, heads * f_Tout, bias=bias, weight_initializer="glorot")

        self.att = torch.nn.Parameter(torch.empty(1, heads, f_Bout + f_Tout))

    def reset_parameters(self) -> None:
        super().reset_parameters()

        self.lin_B.reset_parameters()
        self.lin_B_self.reset_parameters()

        self.lin_T_source.reset_parameters()
        self.lin_T_target.reset_parameters()
        self.lin_T_self.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


###########################
# === Utils functions === #
###########################
def b1_from_edges(valid_lines: Lines, valid_root_mask: torch.Tensor):
    return torch.cat([valid_lines.b1, torch.where(valid_root_mask.any(dim=1))[0]], dim=0)


def optimal_parent(line_score: torch.Tensor, valid_edges: Lines, valid_root_mask: torch.Tensor):
    """Retreive for each branch its parent with the maximum edge score, considering the provided valid edge and root assignments.

    Parameters
    ----------

    line_score: torch.Tensor (N_valid_edges + N_valid_roots,)
        tensor of line scores for valid incident edges and valid root assignments

    valid_edges: Lines
        Lines object containing the valid incident edges

    valid_root_mask: torch.Tensor (N_branch, 2)
        boolean mask indicating which branches are valid for root assignment according to the predicted branch direction (i.e., have at least one valid root direction)
    """  # noqa: E501
    b1_root = torch.where(valid_root_mask.any(dim=1))[0]
    edge_index = torch.cat([valid_edges.edge_index, torch.stack([-torch.ones_like(b1_root), b1_root])], dim=1)

    b1_sort_idx = torch.argsort(edge_index[1])
    line_score = line_score[b1_sort_idx]
    edge_index = edge_index[:, b1_sort_idx]

    score_sort_idx = group_argsort(line_score, edge_index[1], descending=True, return_consecutive=True)
    b1, first_idx = torch.unique_consecutive(edge_index[1], return_counts=True)
    first_idx = torch.cumsum(F.pad(first_idx[:-1], (1, 0), value=0), dim=0)
    b0 = edge_index[0, score_sort_idx[first_idx]]

    parent = torch.full((valid_root_mask.shape[0],), -1, device=line_score.device)
    parent[b1] = b0

    return parent


def optimal_parent_gt(batch):
    device = batch.edge_index.device

    lines = Lines(batch.edge_index, batch.edge_first_tip)
    dir = batch.branch_dir > 0.5
    valid_lines = lines.select_valid_b1_dir(dir)

    valid_root_mask = batch.branch_root_candidates.clone()
    valid_root_mask[torch.arange(batch.num_nodes, device=device), torch.where(dir, 1, 0)] = False

    edge_score = torch.cat([batch.edge_p[valid_lines.whole_mask], batch.branch_root_p[valid_root_mask]], dim=0)

    return optimal_parent(edge_score, valid_lines, valid_root_mask)

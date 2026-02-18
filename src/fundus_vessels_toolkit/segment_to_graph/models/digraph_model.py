from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import cached_property
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch import Tensor, nn
from torch_geometric.data import Batch as PyGBatch
from torch_geometric.nn.conv import GATv2Conv, MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torchvision.models import EfficientNet_V2_S_Weights
from torchvision.models.efficientnet import efficientnet_v2_s
from torchvision.transforms.functional import normalize

from ...utils.torch import torch_interp_bilinear, unique_first
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
    def __init__(self, n_in: int = 512, n_out: int = 1024, edge_attr_dim: Optional[int] = None):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out

        @dataclass(frozen=True)
        class GATv2Opt:
            edge_dim: Optional[int] = edge_attr_dim
            residual: bool = True
            add_self_loops: bool = True
            fill_value: float | Tensor | str = "mean"

        opt = asdict(GATv2Opt())  # fill_value=torch.ones(7)))

        self.bn0 = pyg_nn.InstanceNorm(n_in)
        self.gat0 = GATv2Conv(n_in, 64, heads=8, dropout=0.1, **opt)
        self.bn1 = pyg_nn.InstanceNorm(64 * 8)
        self.gat1a = GATv2Conv(64 * 8, 128, heads=8, dropout=0.1, **opt)
        self.gat1b = GATv2Conv(128 * 8, 256, heads=4, dropout=0, **opt)
        # self.bn2 = pyg_nn.InstanceNorm(256 * 4)
        self.gat2a = GATv2Conv(256 * 4, 128, heads=8, dropout=0, **opt)
        self.gat2b = GATv2Conv(128 * 8, 256, heads=4, dropout=0, **opt)
        # self.bn3 = pyg_nn.InstanceNorm(256 * 4)
        self.gat3a = GATv2Conv(256 * 4, 512, heads=2, dropout=0, **opt)
        self.gat3b = GATv2Conv(512 * 2, n_out, heads=1, dropout=0, **opt)

    def forward(self, x, edge_index, batch_idx, batch_size, edge_attr=None):
        x = self.bn0(x, batch_idx, batch_size)
        x = self.gat0(x, edge_index, edge_attr=edge_attr).relu()
        x = self.bn1(x, batch_idx, batch_size)
        x = self.gat1a(x, edge_index, edge_attr=edge_attr).relu()
        x = self.gat1b(x, edge_index, edge_attr=edge_attr).relu()
        # x = self.bn2(x, batch_idx, batch_size)
        x = self.gat2a(x, edge_index, edge_attr=edge_attr).relu()
        x = self.gat2b(x, edge_index, edge_attr=edge_attr).relu()
        # x = self.bn3(x, batch_idx, batch_size)
        x = self.gat3a(x, edge_index, edge_attr=edge_attr).relu()
        x = self.gat3b(x, edge_index, edge_attr=edge_attr)

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
        features_map: Tensor | list[Tensor],
        branch_curves: Tensor,
        batch_idx: Tensor,
        img_shape: tuple[int, int],
    ) -> Tensor:
        """
        Sample features along branch curves from a feature map.

        Parameters
        ----------
            features_map: Tensor (B, C_feature, H, W)
                tensor of feature maps
            branch_curves: Tensor (N_branch, L, 2)
                tensor of branch curves, where L is the number of curve points
            batch_idx: Tensor (N_branch,)
                tensor of batch indices for each branch
        Returns
        -------
            Tensor (N_branch, C_feature)
                tensor of sampled branch features
        """
        C = branch_curves.shape[1]
        curves_y_tip0, curves_x_tip0 = branch_curves[:, : C // 2].int().unbind(-1)
        curves_y_tip1, curves_x_tip1 = branch_curves[:, C // 2 :].int().unbind(-1)
        batch_idx = batch_idx[:, None]

        if isinstance(features_map, list):
            features_tip0, features_tip1 = [], []

            for fmap in features_map:
                fmap_shape = fmap.shape[-2:]
                if fmap_shape == img_shape:
                    assert fmap_shape[-2] > curves_y_tip0.max() and fmap_shape[-1] > curves_x_tip0.max(), (
                        f"Feature map shape {fmap_shape} is smaller than max curve coordinates {(curves_y_tip0.max(), curves_x_tip0.max())}"
                    )
                    features_tip0.append(fmap[batch_idx, :, curves_y_tip0, curves_x_tip0].mean(dim=-2))
                    features_tip1.append(fmap[batch_idx, :, curves_y_tip1, curves_x_tip1].mean(dim=-2))
                else:
                    Y, X = fmap_shape
                    scale_y = img_shape[0] / Y
                    scale_x = img_shape[1] / X
                    y_tip0, y_tip1 = curves_y_tip0 / scale_y, curves_y_tip1 / scale_y
                    x_tip0, x_tip1 = curves_x_tip0 / scale_x, curves_x_tip1 / scale_x

                    features_tip0.append(torch_interp_bilinear(fmap, y_tip0, x_tip0, batch_idx).mean(dim=-2))
                    features_tip1.append(torch_interp_bilinear(fmap, y_tip1, x_tip1, batch_idx).mean(dim=-2))

            features_tip0 = torch.cat(features_tip0, dim=-1)
            features_tip1 = torch.cat(features_tip1, dim=-1)
        else:
            features_tip0 = features_map[batch_idx, :, curves_y_tip0, curves_x_tip0].mean(dim=-2)
            features_tip1 = features_map[batch_idx, :, curves_y_tip1, curves_x_tip1].mean(dim=-2)

        return torch.cat([features_tip0, features_tip1], dim=-1)  # (B, 512)

    def forward(self, data: VBranchDigraphBatch) -> Output:
        # === Extract branch features ===
        img = data.img.reshape(data.batch_size, 3, *data.img.shape[1:]).to(dtype=torch.bfloat16)
        img_size = (img.shape[-2], img.shape[-1])
        img_features = self.img_feature_extractor(img)
        branch_features = self.sample_features(img_features, data.branch_curves, data.batch, img_size)

        # === Refine branch representation with the GNN ===
        lines = Lines(data.edge_index, data.edge_first_tip)
        x = self.gnn(branch_features, lines.edge_index, data.batch, data.batch_size, edge_attr=data.edge_attr)

        # === Predict branch AV class, direction, affinity and root probability ===
        branch_fp = self.lin_fp(x).squeeze(-1)
        branch_av = self.lin_av(x).squeeze(-1)
        branch_dir = self.lin_dir(x).squeeze(-1)
        branch_affinity_v = self.lin_affinity(x)
        branch_root_score = self.lin_root(x).squeeze(-1)

        # Compute affinity between connected branches
        edge_score = F.cosine_similarity(branch_affinity_v[lines.b0], branch_affinity_v[lines.b1], dim=-1)

        return BranchDigraphModel.Output(
            batch=data,
            fp_logit=branch_fp,
            av_logit=branch_av,
            dir_logit=branch_dir,
            edge_score=edge_score,
            root_score=branch_root_score,
        )

    @dataclass(frozen=True)
    class Output:
        batch: VBranchDigraphBatch | VBranchDigraphData
        """Batch data passed to the model, used for convenience to compute losses and metrics"""

        fp_logit: Tensor
        """Tensor of shape (N_branch,) containing the logits for each branch being a false positive (i.e., not corresponding to any GT branch)"""  # noqa: E501

        av_logit: Tensor
        """Tensor of shape (N_branch,) containing the logits for each branch class (1: artery, 0: vein)"""  # noqa: E501

        dir_logit: Tensor
        """Tensor of shape (N_branch,) containing logits for each branch direction (positive: branch is oriented from tip0 to tip1, negative: branch is oriented from tip1 to tip0)"""  # noqa: E501

        edge_score: Tensor
        """Tensor of shape (N_edge,) containing affinity scores"""  # noqa: E501

        root_score: Tensor
        """Tensor of shape (N_branch,) containing root affinity scores"""  # noqa: E501

        @property
        def batch_size(self):
            return self.batch.batch_size if isinstance(self.batch, PyGBatch) else 0

        @property
        def n_branch(self):
            return self.batch.branch_curves.shape[0]

        @property
        def n_edge(self):
            return self.batch.edge_index.shape[1]

        @property
        def names(self):
            return self.batch.name

        @property
        def name(self):
            assert isinstance(self.batch.name, str), "Batch contains multiple graphs, cannot return a single name"
            return self.batch.name

        @property
        def edge_lines(self):
            """Lines linking two branches. Their score are specified by the edge_score attribute."""
            return self.lines[: self.n_edge]

        @property
        def root_lines(self):
            """Lines linking the root node to branches. Their score are specified by the root_score attribute."""
            return self.lines[self.n_edge :]

        @cached_property
        def lines(self):
            """The concatenation of edge lines (linking two branches) and root lines (linking the virtual root node to a branch). The root lines are added based on the valid root candidates indicated in the batch data, and their score is given by the root affinity score of their target branch."""  # noqa: E501
            root_branch, root_tip = torch.where(self.batch.branch_root_candidates)
            root_lines = torch.stack([-torch.ones_like(root_branch), root_branch], dim=0)
            root_first_tip = root_tip == 0
            root_first_tip = torch.stack([torch.zeros_like(root_first_tip), root_first_tip], dim=-1)
            return Lines(
                edge_index=torch.cat([self.batch.edge_index, root_lines], dim=1),
                edge_first_tip=torch.cat([self.batch.edge_first_tip, root_first_tip], dim=0),
            )

        @cached_property
        def lines_score(self):
            root_score = self.root_score[self.root_lines.b1]
            return torch.cat([self.edge_score, root_score], dim=0)

        def lines_mask(self, filter_dir: Literal["gt"] | bool = False, filter_fp: Literal["gt"] | bool = False):
            """
            Compute a boolean mask to select lines based on their consistency with the branches direction and false positive predictions.

            Parameters
            ----------
            filter_dir:
                If True or "gt", only keep lines whose direction is consistent with the predicted branch direction or the ground truth branch direction, respectively.

            filter_fp:
                If True or "gt", only keep lines whose both branches are predicted as true positive by the model or by the ground truth, respectively.

            Returns
            -------
                A boolean tensor of shape (N_line,) indicating the selected lines.
            """  # noqa: E501
            mask = torch.ones(len(self.lines), dtype=torch.bool, device=self.lines.b0.device)
            cache = self.__dict__.setdefault("__dir_cache", {})
            if filter_dir:
                if (dir_mask := cache.get(f"dir_{filter_dir}")) is None:
                    branch_dir = self.gt_dir_p > 0.5 if filter_dir == "gt" else self.dir_logit > 0
                    dir_mask = self.lines.b1_dir_mask(branch_dir)
                    cache[f"dir_{filter_dir}"] = dir_mask
                mask &= dir_mask
            if filter_fp:
                if (fp_mask := cache.get(f"fp_{filter_fp}")) is None:
                    branch_mask = self.gt_fp_p < 0.5 if filter_fp == "gt" else self.fp_logit <= 0
                    fp_mask = branch_mask[self.lines.b1]
                    cache[f"fp_{filter_fp}"] = fp_mask
                mask &= fp_mask
            return mask

        def max_parent(self, use_gt=False):
            """
            Compute the optimal parent branch using the line scores predicted by the model.

            Only the lines consistent with the predicted branch direction and false positive status are considered.

            Parameters
            ----------
            use_gt:
                If True, use the ground truth branch direction and false positive labels instead of the model predictions to select lines.

            Returns
            -------
                An integer tensor of shape (N_branch,) containing for each branch the index of its optimal parent, or -1 if it has no parent. Branch with no incoming valid lines are considered root branches by default.
            """  # noqa: E501
            line_mask = self.lines_mask(filter_dir=use_gt, filter_fp=use_gt)
            return _max_parent(self.lines[line_mask], self.lines_score[line_mask], self.n_branch)

        @cached_property
        def optimal_tree(self) -> tuple[Tensor, Tensor]:
            """
            Compute the optimal parent branch using the line scores predicted by the model, considering only the lines consistent with the ground truth branch direction and false positive status.

            Returns
            -------
                branch_parent: Tensor (N_branch,)
                    Integer tensor containing for each branch the index of its optimal parent, or -1 if it has no parent. Branch with no incoming valid lines are considered root branches by default.
                branch_dir: Tensor (N_branch,)
                    Boolean tensor containing for each branch the direction of the line linking it to its optimal parent (True: from tip0 to tip1, False: from tip1 to tip0). The direction of branches with no parent is set to False by default.
            """  # noqa: E501
            return _optimal_arborescence_parent(
                self.lines, self.lines_score, self.fp_logit, self.av_logit, self.dir_logit
            )

        def gt_parent(self):
            """
            Compute the optimal parent branch using the line scores provided in the ground truth.

            Only the lines consistent with the ground truth branch direction and false positive status are considered.

            Returns
            -------
                An integer tensor of shape (N_branch,) containing for each branch the index of its optimal parent, or -1 if it has no parent.
            """  # noqa: E501
            line_mask = self.lines_mask(filter_dir="gt", filter_fp="gt")
            return _max_parent(self.lines[line_mask], self.gt_lines_score[line_mask], self.n_branch)

        @cached_property
        def fp_p(self):
            """Tensor of shape (N_branch,) containing the probability of each branch being a false positive (i.e., not corresponding to any GT branch)"""  # noqa: E501
            return torch.sigmoid(self.fp_logit)

        @cached_property
        def av_p(self):
            """Tensor of shape (N_branch,) containing the probability of each branch being an artery (values close to 1) or a vein (values close to 0)"""  # noqa: E501
            return torch.sigmoid(self.av_logit)

        @cached_property
        def fp_av_class(self):
            """Tensor of shape (N_branch,) containing the predicted class of each branch (0: vein, 1: artery, 2: false positive)"""  # noqa: E501
            av_class = (self.av_logit > 0).int()
            av_class[self.fp_logit > 0] = -1
            return av_class

        @cached_property
        def dir_p(self):
            """Tensor of shape (N_branch,) containing the probability of each branch being oriented from tip0 to tip1 (values close to 1) or from tip1 to tip0 (values close to 0)"""  # noqa: E501
            return torch.sigmoid(self.dir_logit)

        @property
        def gt_lines_score(self):
            return torch.cat([self.batch.edge_p, self.batch.branch_root_p[self.batch.branch_root_candidates]], dim=0)

        @property
        def gt_dir_p(self):
            """Tensor of shape (N_branch,) containing the ground truth probability of each branch being oriented from tip0 to tip1 (values close to 1) or from tip1 to tip0 (values close to 0)"""  # noqa: E501
            return self.batch.branch_dir_p

        @cached_property
        def _gt_fp_av_p(self):
            return split_fp_av_p(self.batch.branch_av_p)

        @property
        def gt_fp_p(self):
            """Tensor of shape (N_branch,) containing the ground truth probability of each branch being a false positive (i.e., not corresponding to any GT branch)"""  # noqa: E501
            return self._gt_fp_av_p[0]

        @property
        def gt_av_p(self):
            """Tensor of shape (N_branch,) containing the ground truth probability of each branch being an artery (values close to 1) or a vein (values close to 0)"""  # noqa: E501
            return self._gt_fp_av_p[1]

        def unbatch(self) -> list[BranchDigraphModel.Output]:
            assert isinstance(self.batch, PyGBatch), "Batch data must be a torch geometric Batch for unbatching"
            outputs = []

            for idx, single_data in enumerate(self.batch.to_data_list()):
                branch_mask = self.batch.batch == idx
                outputs.append(
                    BranchDigraphModel.Output(
                        batch=single_data,  # type: ignore
                        fp_logit=self.fp_logit[branch_mask],
                        av_logit=self.av_logit[branch_mask],
                        dir_logit=self.dir_logit[branch_mask],
                        edge_score=self.edge_score[branch_mask[self.edge_lines.b0]],
                        root_score=self.root_score[branch_mask],
                    )
                )
            return outputs


class Lines:
    def __init__(
        self,
        edge_index: Tensor,
        edge_first_tip: Tensor,
        mask: Optional[Tensor] = None,
        whole_mask: Optional[Tensor] = None,
    ):
        self.edge_index = edge_index
        self.edge_first_tip = edge_first_tip
        self.mask = mask
        self.whole_mask = whole_mask

    def __bool__(self):
        return self.edge_index.shape[1] > 0

    def __len__(self):
        return self.edge_index.shape[1]

    def __getitem__(self, idx):
        if self.whole_mask is None:
            whole_mask = torch.zeros(self.edge_index.shape[1], dtype=torch.bool, device=self.edge_index.device)
            whole_mask[idx] = True
        else:
            whole_mask = torch.zeros_like(self.whole_mask)
            whole_mask[self.whole_mask][idx] = True
        return Lines(
            self.edge_index[:, idx],
            self.edge_first_tip[idx],
            mask=self.mask[idx] if self.mask is not None else None,
            whole_mask=whole_mask,
        )

    @property
    def n_lines(self) -> int:
        """Number of lines"""
        return self.edge_index.shape[1]

    @property
    def b0(self) -> Tensor:
        """Integer tensor of shape (N_line,) containing the index of the source branch of each line"""
        return self.edge_index[0]

    @property
    def b1(self) -> Tensor:
        """Integer tensor of shape (N_line,) containing the index of the target branch of each line"""
        return self.edge_index[1]

    @property
    def tip0(self) -> Tensor:
        """Boolean tensor of shape (N_line,) indicating whether the line is emitted from the first tip (True) or the second tip (False) of its source branch"""  # noqa: E501
        return self.edge_first_tip[:, 0]

    @property
    def tip1(self) -> Tensor:
        """Boolean tensor of shape (N_line,) indicating whether the line is incident to the first tip (True) or the second tip (False) of its target branch"""  # noqa: E501
        return self.edge_first_tip[:, 1]

    def select_valid_lines(self, b1_dir, fp) -> Lines:
        return self[self.b1_dir_mask(b1_dir) & ~fp[self.b1] & ~fp[self.b0]]

    def b1_dir_mask(self, b_dir: Tensor) -> Tensor:
        """
        Compute the mask indicating which lines is compatible with the provided direction of their target branch.

        Parameters
        ----------
        b_dir: Tensor
            Boolean tensor of shape (N_branch,) indicating the direction of each branch (True: from tip0 to tip1, False: from tip1 to tip0)

        Returns
        -------
        Tensor
            Boolean tensor of shape (N_line,) indicating whether each line is compatible with the provided direction of
            their target branch.
            (True: the line is incident to the tip of the target branch that should be the end of the branch according to b_dir;
            False: the line is incident to the tip of the target branch that should be the start of the branch according to b_dir.)

        """  # noqa: E501
        return self.tip1 == b_dir[self.b1]

    def groupby_incident_edges(self, select_b1_dir=None):
        if select_b1_dir is None:
            return self.b1
        else:
            b1 = self.b1
            return (b1 + 1) * (self.tip1 == select_b1_dir[b1])

    def sort_b1(self) -> Tensor:
        sort_idx = torch.argsort(self.b1)
        self.edge_index = self.edge_index[:, sort_idx]
        self.edge_first_tip = self.edge_first_tip[sort_idx]
        return sort_idx


def split_fp_av_p(av_p: Tensor) -> tuple[Tensor, Tensor]:
    """Split pairs of artery and vein probabilities into false positive and artery/vein probabilities."""
    av_sum = av_p.sum(dim=-1)
    fp = 1 - av_sum
    art = av_p[..., 0]
    art[av_sum != 0] /= av_sum[av_sum != 0]
    return fp, art


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
        pass
        # x, edge_index = data.x, data.edge_index

        # x = self.conv1(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x = self.conv2(x, edge_index)

        # return F.log_softmax(x, dim=1)


###########################
# === Utils functions === #
###########################
def b1_from_edges(valid_lines: Lines, valid_root_mask: Tensor):
    return torch.cat([valid_lines.b1, torch.where(valid_root_mask.any(dim=1))[0]], dim=0)


def _max_parent(lines: Lines, lines_score: Tensor, n_branch: int):
    """Retreive for each branch its parent with the maximum edge score, considering the provided valid edge and root assignments.

    Parameters
    ----------

    line_score: Tensor (N_valid_edges + N_valid_roots,)
        tensor of line scores for valid incident edges and valid root assignments

    valid_edges: Lines
        Lines object containing the valid incident edges

    valid_root_mask: Tensor (N_branch, 2)
        boolean mask indicating which branches are valid for root assignment according to the predicted branch direction (i.e., have at least one valid root direction)
    """  # noqa: E501

    if not lines:
        return torch.full((n_branch,), -1, dtype=torch.long, device=lines_score.device)

    score_sort_idx = torch.argsort(lines_score, descending=True)
    b1, first_idx = unique_first(lines.b1[score_sort_idx])
    # first_idx = torch.cumsum(F.pad(first_idx[:-1], (1, 0), value=0), dim=0)

    parent = torch.full((n_branch,), -1, device=lines_score.device)
    parent[b1] = lines.b0[score_sort_idx[first_idx]]

    return parent


def _optimal_arborescence_parent(
    lines: Lines, lines_score: Tensor, fp_logit: Tensor, av_logit: Tensor, dir_logit: Tensor
) -> tuple[Tensor, Tensor]:
    """Retreive for each branch its parent with the maximum edge score, considering the provided valid edge and root assignments.

    Parameters
    ----------
    lines: Lines
        Lines object containing the valid incident edges and valid root assignments

    line_score: Tensor (N_valid_edges + N_valid_roots,)
        tensor of line scores for valid incident edges and valid root assignments

    fp_logit: Tensor (N_branch,)
        tensor of logits for each branch being a false positive (i.e., not corresponding to any GT branch)

    av_logit: Tensor (N_branch,)
        tensor of logits for each branch being an artery (values close to 1) or a vein (values close to 0)

    dir_logit: Tensor (N_branch,)
        tensor of logits for each branch being oriented from tip0 to tip1 (values close to 1) or from tip1 to tip0 (values close to 0)

    Returns
    -------
        b_parent: Tensor (N_branch,)
            integer tensor containing for each branch the index of its optimal parent, or -1 if it has no parent.

        b_dir: Tensor (N_branch,)
            boolean tensor containing for each branch the direction of its optimal parent line (True: from tip0 to tip1, False: from tip1 to tip0).

    """  # noqa: E501
    from ..vbranch_digraph import VBranchDigraph

    line_list = np.empty((lines.n_lines, 4), dtype=np.int64)
    line_list[:, 0] = lines.b0.numpy(force=True)
    line_list[:, 1] = lines.tip0.numpy(force=True)
    line_list[:, 2] = lines.b1.numpy(force=True)
    line_list[:, 3] = lines.tip1.numpy(force=True)

    line_p = lines_score.numpy(force=True)
    dir_p = torch.sigmoid(dir_logit).numpy(force=True)
    fp_p = torch.sigmoid(fp_logit).numpy(force=True)
    av_p = torch.sigmoid(av_logit).numpy(force=True)
    av_p = np.stack([av_p, 1 - av_p], axis=-1) * (1 - fp_p[:, None])

    digraph = VBranchDigraph(line_list=line_list, line_p=line_p, branch_dir_p=dir_p, branch_av_p=av_p)
    b_parent, b_dir = digraph.solve_optimal_arboresence()

    device = lines_score.device
    return torch.from_numpy(b_parent).to(device), torch.from_numpy(b_dir).to(device)

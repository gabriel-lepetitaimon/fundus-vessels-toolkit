from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import cached_property
from types import EllipsisType
from typing import Literal, Optional

import numpy as np
import torch
import torch_geometric.nn as pyg_nn
from torch import Tensor, nn
from torch_geometric.data import Batch as PyGBatch
from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import softmax
from torchvision.models import EfficientNet_V2_S_Weights
from torchvision.models.efficientnet import efficientnet_v2_s
from torchvision.transforms.functional import normalize

from fundus_vessels_toolkit.segment_to_graph.vbranch_digraph import VBranchDigraph
from fundus_vessels_toolkit.utils.tree import tree_connected_components

from ...utils.torch import groupby_mean, torch_interp_bilinear, unique_first
from .dataset import VBranchDigraphBatch, VBranchDigraphData
from .gnn_with_pos_encoding import APE, RoPE, TransformerConvWithPosEncoding


class BranchFeaturesEfficientNetV2S(torch.nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        net = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT if pretrained else None).features
        self.efficient_net = net
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
            fill_value: float | Tensor | str = torch.ones(7)

        opt = asdict(GATv2Opt())

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

    def forward(self, x, edge_index, batch_idx, batch_size, edge_attr=None, pos=None):
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


class TransformerGCN(torch.nn.Module):
    def __init__(self, n_in: int = 512, n_out: int = 1024, edge_attr_dim: Optional[int] = None):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out

        @dataclass(frozen=True)
        class TransformerConvOpt:
            edge_dim: Optional[int] = edge_attr_dim
            beta: bool = True
            pos_encoding: Optional[RoPE | Literal["axial", "spiral"]] = "spiral"

        opt = asdict(TransformerConvOpt())  # fill_value=torch.ones(7)))

        self.bn0 = pyg_nn.InstanceNorm(n_in)
        self.conv0 = TransformerConvWithPosEncoding(n_in, 64, heads=8, dropout=0.1, **opt)
        self.bn1 = pyg_nn.InstanceNorm(64 * 8)
        self.conv1a = TransformerConvWithPosEncoding(64 * 8, 128, heads=8, dropout=0.1, **opt)
        self.conv1b = TransformerConvWithPosEncoding(128 * 8, 256, heads=4, dropout=0, **opt)
        self.bn2 = pyg_nn.InstanceNorm(256 * 4)
        self.conv2a = TransformerConvWithPosEncoding(256 * 4, 128, heads=8, dropout=0, **opt)
        self.conv2b = TransformerConvWithPosEncoding(128 * 8, 256, heads=4, dropout=0, **opt)
        self.bn3 = pyg_nn.InstanceNorm(256 * 4)
        self.conv3a = TransformerConvWithPosEncoding(256 * 4, 512, heads=2, dropout=0, **opt)
        self.conv3b = TransformerConvWithPosEncoding(512 * 2, n_out, heads=1, dropout=0, **opt)

    def forward(self, x, edge_index, batch_idx, batch_size, edge_attr=None, pos=None):
        x = self.bn0(x, batch_idx, batch_size)
        x = self.conv0(x, edge_index, edge_attr=edge_attr, pos=pos).relu()
        x = self.bn1(x, batch_idx, batch_size)
        x = self.conv1a(x, edge_index, edge_attr=edge_attr, pos=pos).relu()
        x = self.conv1b(x, edge_index, edge_attr=edge_attr, pos=pos).relu()
        # x = self.bn2(x, batch_idx, batch_size)
        x = self.conv2a(x, edge_index, edge_attr=edge_attr, pos=pos).relu()
        x = self.conv2b(x, edge_index, edge_attr=edge_attr, pos=pos).relu()
        # x = self.bn3(x, batch_idx, batch_size)
        x = self.conv3a(x, edge_index, edge_attr=edge_attr, pos=pos).relu()
        x = self.conv3b(x, edge_index, edge_attr=edge_attr, pos=pos)

        return x


class BranchDigraphModel(torch.nn.Module):
    def __init__(
        self,
        img_feature_extractor: nn.Module,
        gnn: nn.Module,
        gnn_out_channels: Optional[int] = None,
        oriented_affinity: bool = True,
    ):
        super().__init__()

        self.img_feature_extractor = img_feature_extractor
        self.gnn = gnn
        self.positional_encoding = APE(head_dim=gnn.n_in // 2)  # type: ignore

        if gnn_out_channels is None:
            assert hasattr(gnn, "n_out"), "gnn_out_channels must be specified if gnn does not have n_out attribute"
            gnn_out_channels = int(gnn.n_out)  # type: ignore

        # === Classification layers ===
        self.lin_fp = Linear(gnn_out_channels, 1, weight_initializer="glorot")
        self.lin_av = Linear(gnn_out_channels, 1, weight_initializer="glorot")
        self.lin_dir = Linear(gnn_out_channels, 1, weight_initializer="glorot")
        self.lin_root = Linear(gnn_out_channels, 1, weight_initializer="glorot")
        self.lin_affinity = Linear(gnn_out_channels, 128 * (2 if oriented_affinity else 1), weight_initializer="glorot")

        self.polarized_affinity = oriented_affinity

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
        if not isinstance(data, PyGBatch):
            data.batch = torch.zeros(data.branch_curves.shape[0], dtype=torch.long, device=data.branch_curves.device)
            data.batch_size = 1

        # === Extract branch features ===
        img = data.img.reshape(data.batch_size, 3, *data.img.shape[1:])
        img_size = (img.shape[-2], img.shape[-1])
        img_features = self.img_feature_extractor(img)
        branch_features = self.sample_features(img_features, data.branch_curves, data.batch, img_size)

        # Add positional encoding
        pos = data.branch_curves[:, [0, -1], :].reshape(-1, 2)  # (N_branch * 2, 2)
        pos = reproject_pos(pos, data.od_yx, data.mac_yx - data.od_yx, data.batch.repeat_interleave(2))
        if False:
            pos_encoding = self.positional_encoding.compute_pos_encoding(pos)
            pos_encoding = pos_encoding.reshape(branch_features.shape)

            branch_features += pos_encoding

        # === Refine branch representation with the GNN ===
        lines = Lines.from_batch(data)
        x = self.gnn(
            branch_features,
            lines.edge_index,
            data.batch,
            data.batch_size,
            edge_attr=data.edge_attr,
            pos=pos.reshape(2, -1, 2),
        )

        # === Predict branch AV class, direction, affinity and root probability ===
        branch_fp = self.lin_fp(x).squeeze(-1)
        branch_av = self.lin_av(x).squeeze(-1)
        branch_dir = self.lin_dir(x).squeeze(-1)
        branch_affinity_v = self.lin_affinity(x)
        branch_root_score = self.lin_root(x).squeeze(-1)

        # Compute affinity between connected branches
        if self.polarized_affinity:
            b0_embedding, b1_embedding = branch_affinity_v.view(x.shape[0], -1, 2).unbind(-1)
        else:
            b0_embedding = b1_embedding = branch_affinity_v

        return BranchDigraphModel.Output(
            batch=data,
            fp_logit=branch_fp,
            av_logit=branch_av,
            dir_logit=branch_dir,
            b0_embedding=b0_embedding,
            b1_embedding=b1_embedding,
            root_logit=branch_root_score,
        )

    @dataclass(frozen=True)
    class Output:
        batch: VBranchDigraphBatch | VBranchDigraphData
        """Batch data passed to the model, used for convenience to compute losses and metrics"""

        fp_logit: Tensor
        """Tensor of shape (N_branch,) containing the logits for each branch being a false positive (i.e., not corresponding to any GT branch)"""  # noqa: E501

        av_logit: Tensor
        """Tensor of shape (N_branch,) containing the logits for each branch class (1: artery, 0: vein)"""

        dir_logit: Tensor
        """Tensor of shape (N_branch,) containing logits for each branch direction (positive: branch is oriented from tip0 to tip1, negative: branch is oriented from tip1 to tip0)"""  # noqa: E501

        b0_embedding: Tensor
        """Tensor of shape (N_branch, F) containing the embeddings for each branch as a parent branch"""

        b1_embedding: Tensor
        """Tensor of shape (N_branch, F) containing the embeddings for each branch as a child branch"""

        root_logit: Tensor
        """Tensor of shape (N_branch,) containing root affinity scores"""

        @property
        def batch_size(self):
            return self.batch.batch_size if isinstance(self.batch, PyGBatch) else 0

        @property
        def batch_idx(self):
            return self.batch.batch if isinstance(self.batch, PyGBatch) else 0

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
        def device(self):
            return self.batch.edge_index.device

        @property
        def edge_lines(self):
            """Lines linking two branches. Their score are specified by the edge_score attribute."""
            return self.lines[: self.n_edge]

        @property
        def root_lines(self):
            """Lines linking the root node to branches. Their score are specified by the root_score attribute."""
            return self.lines[self.n_edge :]

        @cached_property
        def lines(self) -> Lines:
            """The concatenation of edge lines (linking two branches) and root lines (linking the virtual root node to a branch). The root lines are added based on the valid root candidates indicated in the batch data, and their score is given by the root affinity score of their target branch."""  # noqa: E501
            root_branch, root_tip = torch.where(self.batch.branch_root_candidates)
            root_lines = torch.stack([-torch.ones_like(root_branch), root_branch], dim=0)
            root_dir = root_tip == 0
            root_dir = torch.stack([torch.zeros_like(root_dir), root_dir], dim=-1)
            return Lines(
                edge_index=torch.cat([self.batch.edge_index, root_lines], dim=1),
                edge_dir=torch.cat([self.batch.edge_dir, root_dir], dim=0),
                branch_nodes=self.batch.branch_nodes,
            )

        @cached_property
        def edge_score(self):
            """Affinity score for edge lines, computed from the embeddings of the connected branches."""
            return (self.b0_embedding[self.edge_lines.b0] * self.b1_embedding[self.edge_lines.b1]).sum(dim=-1)

        @cached_property
        def lines_logit(self):
            root_score = self.root_logit[self.root_lines.b1]
            return torch.cat([self.edge_score, root_score], dim=0)

        @cached_property
        def final_lines_p(self):
            lines_p = softmax(
                self.lines_logit,
                index=self.lines.b1 + torch.where(self.lines.b1_dir, 0, self.n_branch),
                num_nodes=2 * self.n_branch,
            )
            # Artery/Vein label consistency
            av_consistency = self.av_logit[self.lines.b0] * self.av_logit[self.lines.b1]
            return lines_p + av_consistency.sigmoid()

        def lines_mask(
            self,
            filter_dir: Literal["gt"] | bool | EllipsisType = ...,
            filter_fp: Literal["gt"] | bool | EllipsisType = ...,
        ):
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
            if filter_dir is Ellipsis and filter_fp is Ellipsis:
                filter_dir = filter_fp = True
            elif filter_dir is Ellipsis:
                filter_dir = False
            elif filter_fp is Ellipsis:
                filter_fp = False

            mask = torch.ones(len(self.lines), dtype=torch.bool, device=self.lines.b0.device)
            cache = self.__dict__.setdefault("__dir_cache", {})
            if filter_dir:
                if (dir_mask := cache.get(f"dir_{filter_dir}")) is None:
                    branch_dir = self.gt_dir_p > 0.5 if filter_dir == "gt" else self.dir_logit > 0
                    dir_mask = self.lines.b1_dir == branch_dir[self.lines.b1]
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
            line_mask = self.lines_mask(filter_dir="gt" if use_gt else True, filter_fp="gt" if use_gt else True)
            return _max_parent(self.lines[line_mask], self.lines_logit[line_mask], self.n_branch)

        @cached_property
        def gt_parent(self) -> Tensor:
            """
            Compute the optimal parent branch using the line scores provided in the ground truth.

            Only the lines consistent with the ground truth branch direction and false positive status are considered.

            Returns
            -------
                An integer tensor of shape (N_branch,) containing for each branch the index of its optimal parent, or -1 if it has no parent.
            """  # noqa: E501
            line_mask = self.lines_mask(filter_dir="gt", filter_fp="gt")
            return _max_parent(self.lines[line_mask], self.gt_lines_score[line_mask], self.n_branch)

        def to_digraph(self) -> VBranchDigraph:
            """
            Convert the predicted tree structure to a VBranchDigraph object.

            Only the lines consistent with the predicted branch direction and false positive status are considered.

            Returns
            -------
                A VBranchDigraph object containing the predicted tree structure.
            """  # noqa: E501
            line_list = np.empty((len(self.lines), 4), dtype=np.int64)
            line_list[:, 0] = self.lines.b0.numpy(force=True)
            line_list[:, 1] = self.lines.b0_dir.numpy(force=True).astype(np.int64)
            line_list[:, 2] = self.lines.b1.numpy(force=True)
            line_list[:, 3] = 1 - self.lines.b1_dir.numpy(force=True)

            return VBranchDigraph(
                line_list=line_list,
                line_p=self.final_lines_p.float().numpy(force=True),
                branch_dir_logit=self.dir_logit.float().numpy(force=True),
                branch_fp_logit=self.fp_logit.float().numpy(force=True),
                branch_av_logit=self.av_logit.float().numpy(force=True),
            )

        @cached_property
        def optimal_tree(self) -> tuple[Tensor, Tensor, Tensor]:
            """
            Compute the optimal parent branch using the line scores predicted by the model, considering only the lines consistent with the ground truth branch direction and false positive status.

            Returns
            -------
                branch_parent: Tensor (N_branch,)
                    Integer tensor containing for each branch the index of its optimal parent, or -1 if it has no parent. Branch with no incoming valid lines are considered root branches by default.
                branch_dir: Tensor (N_branch,)
                    Boolean tensor containing for each branch the direction of the line linking it to its optimal parent (True: from tip0 to tip1, False: from tip1 to tip0). The direction of branches with no parent is set to False by default.
                branch_av: Tensor (N_branch,)
                    Tensor containing for each branch the predicted artery/vein logit average over the connected components of the optimal tree.
            """  # noqa: E501
            digraph = self.to_digraph()
            try:
                opti_parent, opti_dir = digraph.solve_optimal_arboresence(detect_major_av_error=True)
            except Exception as e:
                print(f"Error solving optimal arborescence for batch {self.names}: {e}")
                digraph.check_lines("warn", branch_mask=~digraph.branch_fp())
                opti_parent = self.max_parent(use_gt=False)
                opti_dir = self.dir_logit > 0
                opti_av = self.av_logit
                return opti_parent, opti_dir, opti_av

            opti_parent = (opti_parent_cpu := torch.from_numpy(opti_parent)).to(self.device)
            opti_dir = torch.from_numpy(opti_dir).to(self.device)

            opti_subtree = tree_connected_components(opti_parent_cpu.cpu()).to(self.device)
            subtree_inverse = torch.unique(opti_subtree + self.batch_idx, return_inverse=True)[1]
            opti_av = groupby_mean(self.av_logit, subtree_inverse)[subtree_inverse]

            return opti_parent, opti_dir, opti_av

        @cached_property
        def gt_root_p(self) -> Tensor:
            """Ground truth probability of each branch being a root branch (i.e., not having any parent) as a tensor of shape (B,)"""  # noqa: E501
            assert VBranchDigraphData.has_gt(self.batch), (
                "Ground truth root probabilities are not available in the batch data"
            )
            B = torch.arange(self.n_branch, device=self.device)
            return self.batch.branch_root_p[B, (self.gt_dir_p < 0.5).int()]

        @cached_property
        def fp_p(self) -> Tensor:
            """Predicted probability of each branch being a false positive (i.e., not corresponding to any GT branch) as a tensor of shape (B,)"""  # noqa: E501
            return torch.sigmoid(self.fp_logit)

        @cached_property
        def av_p(self) -> Tensor:
            """Predicted probability of each branch being an artery (as opposed to a vein) as a tensor of shape (B,)"""  # noqa: E501
            return torch.sigmoid(self.av_logit)

        @cached_property
        def fp_av_class(self) -> Tensor:
            """Tensor of shape (N_branch,) containing the predicted class of each branch (0: vein, 1: artery, 2: false positive)"""  # noqa: E501
            av_class = (self.av_logit > 0).int()
            av_class[self.fp_logit > 0] = -1
            return av_class

        @cached_property
        def dir_p(self) -> Tensor:
            """Predicted probability of each branch being oriented from tip0 to tip1 (values close to 1) or from tip1 to tip0 (values close to 0) as a tensor of shape (N_line,)"""  # noqa: E501
            return torch.sigmoid(self.dir_logit)

        @property
        def gt_lines_score(self) -> Tensor:
            assert VBranchDigraphData.has_gt(self.batch), "Ground truth line scores are not available in the batch data"
            return torch.cat([self.batch.edge_p, self.batch.branch_root_p[self.batch.branch_root_candidates]], dim=0)

        @property
        def gt_dir_p(self) -> Tensor:
            """Ground truth probability of each branch being oriented from tip0 to tip1 (values close to 1) or from tip1 to tip0 (values close to 0) as a tensor of shape (B,)"""  # noqa: E501
            assert VBranchDigraphData.has_gt(self.batch), (
                "Ground truth branch directions are not available in the batch data"
            )
            return self.batch.branch_dir_p

        @property
        def gt_dir(self) -> Tensor:
            """Ground truth direction of each branch as a boolean tensor of shape (B,) (True: from tip0 to tip1, False: from tip1 to tip0)"""  # noqa: E501
            assert VBranchDigraphData.has_gt(self.batch), (
                "Ground truth branch directions are not available in the batch data"
            )
            return self.gt_dir_p > 0.5

        @property
        def gt_fp_p(self) -> Tensor:
            """Ground truth probability of each branch being a false positive from the segmentation as a tensor of shape (B,)"""  # noqa: E501
            assert VBranchDigraphData.has_gt(self.batch), (
                "Ground truth false positive probabilities are not available in the batch data"
            )
            return self.batch.branch_fp_p

        @property
        def gt_av_p(self) -> Tensor:
            """Ground truth probability of each branch being an artery (as opposed to a vein) as a tensor of shape (B,)"""  # noqa: E501
            assert VBranchDigraphData.has_gt(self.batch), (
                "Ground truth artery probabilities are not available in the batch data"
            )
            return self.batch.branch_av_p

        @property
        def gt_subtree_idx(self) -> Tensor:
            """Tensor of shape (N_branch,) containing the ground truth subtree index of each branch"""  # noqa: E501
            assert VBranchDigraphData.has_gt(self.batch), (
                "Ground truth subtree indices are not available in the batch data"
            )
            return self.batch.branch_subtree_idx

        def gt_tail_nodes(self, use_parent_head: bool = False) -> Tensor:
            """Indices of the node at the tail of each branch (according to gt_dir) as a tensor of shape (N_line,)."""
            assert VBranchDigraphData.has_gt(self.batch), (
                "Ground truth branch directions are not available in the batch data"
            )
            branch_tails = torch.gather(self.batch.branch_nodes, 1, (1 - self.gt_dir[:, None].int())).squeeze()
            if use_parent_head:
                parent = self.gt_parent
                parent = parent[has_parent := parent >= 0]

                nodes = torch.empty_like(self.batch.branch_nodes[:, 0])
                parent_heads = torch.gather(self.batch.branch_nodes[parent], 1, self.gt_dir[parent, None].int())
                nodes[has_parent] = parent_heads.squeeze()
                nodes[~has_parent] = branch_tails[~has_parent]
                return nodes
            else:
                return branch_tails

        @property
        def gt_head_nodes(self) -> Tensor:
            """Indices of the node at the head of each branch (according to gt_dir) as a tensor of shape (N_line,)."""
            return torch.gather(self.batch.branch_nodes, 1, self.gt_dir[:, None].int()).squeeze()

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
                        root_logit=self.root_logit[branch_mask],
                        b0_embedding=self.b0_embedding[branch_mask],
                        b1_embedding=self.b1_embedding[branch_mask],
                    )
                )
            return outputs


@dataclass(frozen=True)
class Lines:
    edge_index: Tensor
    edge_dir: Tensor
    branch_nodes: Tensor
    mask: Optional[Tensor] = None
    whole_mask: Optional[Tensor] = None

    @classmethod
    def from_batch(cls, batch: VBranchDigraphBatch):
        return cls(edge_index=batch.edge_index, edge_dir=batch.edge_dir, branch_nodes=batch.branch_nodes)

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
            edge_index=self.edge_index[:, idx],
            edge_dir=self.edge_dir[idx],
            branch_nodes=self.branch_nodes,
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
    def b0_dir(self) -> Tensor:
        """Boolean tensor of shape (N_line,) indicating for each line the required direction of its source branch (True: from tip0 to tip1, False: from tip1 to tip0)"""  # noqa: E501
        return self.edge_dir[:, 0]

    @property
    def b1_dir(self) -> Tensor:
        """Boolean tensor of shape (N_line,) indicating for each line the required direction of its target branch (True: from tip0 to tip1, False: from tip1 to tip0)"""  # noqa: E501
        return self.edge_dir[:, 1]

    @property
    def b1_node(self) -> Tensor:
        """Integer tensor of shape (N_line, 2) containing the index of the tail node for each target branch."""
        return torch.gather(self.branch_nodes[self.b1], 1, 1 - self.b1_dir[:, None].long()).squeeze()

    @property
    def tip0(self) -> Tensor:
        """Boolean tensor of shape (N_line,) indicating whether the line is emitted from the first tip (True) or the second tip (False) of its source branch"""  # noqa: E501
        return ~self.edge_dir[:, 0]

    @property
    def tip1(self) -> Tensor:
        """Boolean tensor of shape (N_line,) indicating whether the line is incident to the first tip (True) or the second tip (False) of its target branch"""  # noqa: E501
        return self.edge_dir[:, 1]


###########################
# === Utils functions === #
###########################
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

    parent = torch.full((n_branch,), -1, device=lines_score.device)
    parent[b1] = lines.b0[score_sort_idx[first_idx]]

    return parent


def reproject_pos(pos, o, v, batch_index):
    """Reproject position pos onto an orthonormal base defined by the origin o, direction v and u (orthogonal to v).

    Parameters
    ----------
    pos: Tensor (N, 2)
        tensor of positions to reproject
    o: Tensor (B,2)
        origin of the new base
    v: Tensor (B,2)
        direction of the new base
    batch_index: Tensor (N,)
        tensor of batch indices for each position, indicating which origin and direction to use for each position
    """
    o, v = o.view(-1, 2), v.view(-1, 2)
    v_norm = v.norm(dim=-1, keepdim=True) + 1e-8
    v = v / v_norm
    u = torch.stack([-v[:, 1], v[:, 0]], dim=-1)
    R = torch.stack([v, u], dim=-2)
    p = (pos - o[batch_index]) / v_norm[batch_index]
    p = torch.einsum("nij,ni->nj", R[batch_index], p)
    p[:, 0] *= v[batch_index, 1].sign()
    return p

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import cached_property
from types import EllipsisType
from typing import Literal, Optional

import numpy as np
import torch
import torch_geometric.nn as pyg_nn
from pydantic import BaseModel, ConfigDict, Field
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
from .bipolar_gcn import TransformerGCN, TransformerGCNOpt
from .data import BranchDigraphBatch, BranchDigraphData, DigraphLines
from .positionnal_embedding import APE


class BranchDigraphModelCfg(BaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True)

    gcn: TransformerGCNOpt = Field(default_factory=TransformerGCNOpt)
    img_feature_extractor: Literal["efficientnet_v2_s"] = Field(default="efficientnet_v2_s")

    absolute_position_embedding: bool = Field(default=False)
    """If true, adds an absolute positional embedding to the branch features."""

    oriented_affinity: bool = Field(default=True)
    """If true, predicts a different embedding for parent and child branches when computing edge affinities."""

    branch_embedding_dim: int = Field(default=128)
    """Dimension of the branch embedding used to compute edge affinities."""

    class EdgeAttr(BaseModel):
        model_config = ConfigDict(use_attribute_docstrings=True)

        distance: Literal["scalar", "bins", "none"] = "bins"
        angle: bool = True
        calibre: Literal["scalar", "bins", "none"] = "none"

        distance_bins: tuple[float, ...] = (4.0, 16.0, 64.0, 254.0)
        calibre_bins: tuple[float, ...] = (2.0, 4.0, 16.0, 32.0)

        def __post_init__(self):
            if self.distance == "bins":
                assert len(self.distance_bins) > 0, "distance_bins must be provided when distance is set to 'bins'"
            if self.calibre != "none":
                assert len(self.calibre_bins) > 0, "calibre_bins must be provided when calibre is not set to 'none'"

        @property
        def n_edge_attr(self):
            n = 3 if self.angle else 0
            if self.distance != "none":
                n += len(self.distance_bins) if self.distance == "bins" else 1
            if self.calibre != "none":
                n += 2 * (len(self.calibre_bins) if self.calibre == "bins" else 1)
            return n

        def distance_bins_tensor(self, device=None):
            if not hasattr(self, "_distance_bins_tensor") or self._distance_bins_tensor.device != device:
                self._distance_bins_tensor = torch.tensor(self.distance_bins, device=device)
            return self._distance_bins_tensor

        def calibre_bins_tensor(self, device=None):
            if not hasattr(self, "_calibre_bins_tensor") or self._calibre_bins_tensor.device != device:
                self._calibre_bins_tensor = torch.tensor(self.calibre_bins, device=device)
            return self._calibre_bins_tensor

    edge_attr: EdgeAttr = Field(default_factory=EdgeAttr)


class BranchDigraphModel(torch.nn.Module):
    def __init__(self, opt: BranchDigraphModelCfg | dict, compile: bool = False, **kwargs):
        super().__init__()
        self.opt = opt = BranchDigraphModelCfg.model_validate(opt)
        self._compile = compile

        # --- Model components ---
        self.img_feature_extractor = self.create_img_feature_extractor(opt)
        self.gnn = self.create_gnn(opt)
        self.classif_head = self.create_classif_head(opt)
        if opt.absolute_position_embedding:
            self.absolute_pos_encoding = APE(head_dim=self.img_feature_extractor_channels(opt))
        else:
            self.absolute_pos_encoding = None

        # --- Cache variables ---
        self._tip_sample_decay = None

    def configure_model(self):
        if self._compile:
            self.img_feature_extractor = torch.compile(self.img_feature_extractor)
            self.gnn = torch.compile(self.gnn, dynamic=True)

    @classmethod
    def create_img_feature_extractor(cls, opt: BranchDigraphModelCfg) -> nn.Module:
        match opt.img_feature_extractor:
            case "efficientnet_v2_s":
                return BranchFeaturesEfficientNetV2S()
            case _:
                raise ValueError(f"Unsupported image feature extractor: {opt.img_feature_extractor}")

    @classmethod
    def img_feature_extractor_channels(cls, opt: BranchDigraphModelCfg) -> int:
        match opt.img_feature_extractor:
            case "efficientnet_v2_s":
                return BranchFeaturesEfficientNetV2S.N_FEATURES
            case _:
                raise ValueError(f"Unsupported image feature extractor: {opt.img_feature_extractor}")

    @classmethod
    def create_gnn(cls, opt: BranchDigraphModelCfg) -> TransformerGCN:
        n_in = cls.img_feature_extractor_channels(opt) * (3 if opt.gcn.bipolar_node else 2)
        return TransformerGCN(n_in=n_in, edge_attr_dim=opt.edge_attr.n_edge_attr, opt=opt.gcn)

    @classmethod
    def create_classif_head(cls, opt: BranchDigraphModelCfg) -> nn.Module:
        if not opt.gcn.bipolar_node:
            return SimpleClassifHead(opt.gcn.n_out, opt.branch_embedding_dim, oriented_affinity=opt.oriented_affinity)
        else:
            return PolarizedClassifHead(
                branch_channels=opt.gcn.n_out,
                tip_channels=opt.gcn.n_out_pole,
                affinity_embedding_dim=opt.branch_embedding_dim,
                oriented_affinity=opt.oriented_affinity,
            )

    @property
    def device(self):
        return next(self.parameters()).device

    def tip_decay(self, C):
        if self._tip_sample_decay is None or self._tip_sample_decay.shape[0] != C // 2:
            tip_decay = torch.exp(-(torch.linspace(0, 2, C // 2, device=self.device) ** 2))
            tip_decay /= tip_decay.sum()  # Normalize Gaussian decay to sum to 1
            self._tip_sample_decay = tip_decay
        return self._tip_sample_decay

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
            Tensor (N_branch, 2, C_feature) or (N_branch, 3, C_feature)
                Image features sampled along the curve of each branch.

                - If bipolar_node is False, the feature of each branch is the stacked features sampled at its two tips (shape=(N_branch, 2, C_feature)).
                - If bipolar_node is True, the feature of each branch is the stacked features sampled at its two tips and the mean feature along the branch curve (shape=(N_branch, 3, C_feature)).
        """  # noqa: E501
        C = branch_curves.shape[1]
        halfC = C // 2
        curve_y, curve_x = branch_curves.int().unbind(-1)
        batch_idx = batch_idx[:, None]

        if not isinstance(features_map, list):
            features_map = [features_map]

        if not self.opt.gcn.bipolar_node:
            # === Simple node features: concatenate features at both tips ===
            features_tip = [], []
            for fmap in features_map:
                features = torch_interp_bilinear(fmap, curve_y, curve_x, batch_idx, img_shape)  # (N_branch, F, C)
                features_tip[0].append(features[:, halfC:].mean(dim=1))
                features_tip[1].append(features[:, :halfC].mean(dim=1))
            return torch.stack([torch.cat(f, dim=-1) for f in features_tip], dim=1)  # (B, 2, F)
        else:
            # === Bipolar node features: concatenate features at both tips and their average along the branch ===
            # Tip features are weighted to give more importance to the those near the tip.
            tip_decay = self.tip_decay(C).view(1, -1, 1)

            features_tip, features_branch = ([], []), []
            for fmap in features_map:
                features = torch_interp_bilinear(fmap, curve_y, curve_x, batch_idx, img_shape)
                features_tip[0].append((features[:, halfC:] * tip_decay).sum(dim=1))
                features_tip[1].append((features[:, :halfC] * tip_decay.flip(1)).sum(dim=1))
                features_branch.append(features.mean(dim=-2))
            return torch.stack([torch.cat(f, dim=-1) for f in (features_branch,) + features_tip], dim=1)  # (B, 3, F)

    @classmethod
    def extract_edge_attr(cls, data: BranchDigraphBatch, opt: BranchDigraphModelCfg.EdgeAttr) -> Tensor:
        lines = data.edge_lines
        device = data.edge_index.device

        if opt.distance != "none" or opt.angle:
            assert hasattr(data, "branch_tip_pos") and data.branch_tip_pos is not None, (
                "branch_tip_pos must be provided to compute distance and angle edge attributes"
            )
            b0b1 = data.branch_tip_pos[lines.b0, lines.tip0] - data.branch_tip_pos[lines.b1, lines.tip1]
            b0b1_d = torch.norm(b0b1, dim=1)

        attr = []
        if opt.distance == "scalar":
            attr.append(b0b1_d[:, None])
        elif opt.distance == "bins":
            attr.append(1 - torch.clip(b0b1_d[:, None] / opt.distance_bins_tensor(device)[None, :], 0, 1))

        if opt.calibre != "none":
            assert hasattr(data, "branch_tip_calibre") and data.branch_tip_calibre is not None, (
                "branch_tip_calibre must be provided to compute calibre edge attributes"
            )

            b0_calibre = data.branch_tip_calibre[lines.b0, lines.tip0]
            b1_calibre = data.branch_tip_calibre[lines.b1, lines.tip1]

            if opt.calibre == "scalar":
                attr.append(b0_calibre[:, None])
                attr.append(b1_calibre[:, None])
            elif opt.calibre == "bins":
                attr.append(1 - torch.clip(b0_calibre[:, None] / opt.calibre_bins_tensor(device)[None, :], 0, 1))
                attr.append(1 - torch.clip(b1_calibre[:, None] / opt.calibre_bins_tensor(device)[None, :], 0, 1))

        if opt.angle:
            assert hasattr(data, "branch_tip_tan") and data.branch_tip_tan is not None, (
                "branch_tip_tan must be provided to compute angle edge attributes"
            )
            b0_t = data.branch_tip_tan[lines.b0, lines.tip0]
            b1_t = data.branch_tip_tan[lines.b1, lines.tip1]
            b0_b1_t = torch.zeros_like(b0_t)
            b0_b1_t[b0b1_d != 0, :] = b0b1[b0b1_d != 0] / b0b1_d[b0b1_d != 0, None]  # Avoid division by zero
            attr += [
                torch.einsum("ij,ij->i", t1, t2)[:, None] for t1, t2 in [(b0_t, b0_b1_t), (b1_t, b0_b1_t), (b0_t, b1_t)]
            ]

        return torch.hstack(attr)

    def forward(self, data: BranchDigraphBatch) -> Output:
        if not isinstance(data, PyGBatch):
            data.batch = torch.zeros(data.branch_curves.shape[0], dtype=torch.long, device=data.branch_curves.device)
            data.batch_size = 1

        N_branch = data.branch_curves.shape[0]

        # === Extract branch features ===
        img = data.img.reshape(data.batch_size, 3, *data.img.shape[1:])
        img_size = (img.shape[-2], img.shape[-1])
        img_features = self.img_feature_extractor(img)
        branch_features = self.sample_features(img_features, data.branch_curves, data.batch, img_size)

        # === Extract branch attributes ===
        edge_attr = self.extract_edge_attr(data, self.opt.edge_attr).to(branch_features.dtype)

        # === Extract position and positional embedding ===
        pos = data.branch_curves[:, [0, -1], :].view(N_branch * 2, 2)
        pos_batch = data.batch.repeat_interleave(2)
        pos = reproject_pos(pos, data.od_yx, data.mac_yx - data.od_yx, pos_batch)

        if self.absolute_pos_encoding is not None:
            pos_encoding = self.absolute_pos_encoding.compute_pos_encoding(pos).view(N_branch, 2, -1)
            if self.opt.gcn.bipolar_node:
                branch_features[:, 1:] = branch_features[:, 1:] + pos_encoding
            else:
                branch_features = branch_features + pos_encoding
        pos = pos.view(N_branch, 2, 2)
        branch_features = branch_features.view(N_branch, -1)

        # === Apply GNN ===
        lines = data.edge_lines
        x = self.gnn(
            branch_features,
            edge_index=lines.edge_index,
            edge_pole=lines.edge_tip,
            edge_attr=edge_attr,
            batch_idx=data.batch,
            batch_size=data.batch_size,
            pos=pos,
        )

        # === Apply classification head ===
        return self.classif_head(batch=data, x=x)

    def __call__(self, data: BranchDigraphBatch) -> Output:
        return super().__call__(data)

    @dataclass(frozen=True)
    class Output:
        batch: BranchDigraphBatch | BranchDigraphData
        """Batch data passed to the model, used for convenience to compute losses and metrics"""

        fp_logit: Tensor
        """Tensor of shape (N_branch,) containing the logits for each branch being a false positive (i.e., not corresponding to any GT branch)"""  # noqa: E501

        av_logit: Tensor
        """Tensor of shape (N_branch,) containing the logits for each branch class (1: artery, 0: vein)"""

        dir_logit: Tensor
        """Tensor of shape (N_branch,) containing logits for each branch direction (positive: branch is oriented from tip0 to tip1, negative: branch is oriented from tip1 to tip0)"""  # noqa: E501

        b0_embedding: Tensor
        """Embedding representation for each branch as a parent branch. (Tensor of shape (N_branch, F) or (N_branch, 2, F) if polarized affinity is used.)"""  # noqa: E501

        b1_embedding: Tensor
        """Embedding representation for each branch as a child branch. (Tensor of shape (N_branch, F) or (N_branch, 2, F) if polarized affinity is used.)"""  # noqa: E501

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
        def edge_lines(self) -> DigraphLines:
            """Lines linking two branches. Their score are specified by the edge_score attribute."""
            return self.lines[: self.n_edge]

        @property
        def root_lines(self) -> DigraphLines:
            """Lines linking the root node to branches. Their score are specified by the root_score attribute."""
            return self.lines[self.n_edge :]

        @cached_property
        def lines(self) -> DigraphLines:
            """The concatenation of edge lines (linking two branches) and root lines (linking the virtual root node to a branch). The root lines are added based on the valid root candidates indicated in the batch data, and their score is given by the root affinity score of their target branch."""  # noqa: E501
            return self.batch.lines

        def b1_embedding_gt_tail_tip(self):
            """Return the embedding of branches as child branches (b1_embedding) indexed by their ground truth tail node and tip (0 or 1). This is used for the contrastive loss to mine pairs of branches with the same tail node."""  # noqa: E501
            if self.b1_embedding.ndim == 2:
                return self.b1_embedding
            tail_node_idx = (~self.gt_dir).int()
            return self.b1_embedding[torch.arange(self.n_branch, device=self.device), tail_node_idx]

        @cached_property
        def edge_score(self):
            """Affinity score for edge lines, computed from the embeddings of the connected branches."""
            if self.b0_embedding.ndim == 3:
                # Polarized affinity: b0_embedding and b1_embedding have shape (N_branch, 2, F)
                b0_emb = self.b0_embedding[self.edge_lines.b0, self.edge_lines.tip0]
                b1_emb = self.b1_embedding[self.edge_lines.b1, self.edge_lines.tip1]
                return (b0_emb * b1_emb).sum(dim=-1)
            else:
                # Non polarized affinity: b0_embedding and b1_embedding have shape (N_branch, F).
                return (self.b0_embedding[self.edge_lines.b0] * self.b1_embedding[self.edge_lines.b1]).sum(dim=-1)

        @cached_property
        def lines_logit(self):
            root_score = self.root_logit[self.root_lines.b1]
            return torch.cat([self.edge_score, root_score], dim=0)

        @cached_property
        def lines_p(self):
            return softmax(
                self.lines_logit,
                index=self.lines.b1 + torch.where(self.lines.b1_dir, 0, self.n_branch),
                num_nodes=2 * self.n_branch,
            )

        @cached_property
        def final_lines_p(self):
            fp_logit = self.fp_logit[self.lines.b1].clone()
            not_root = self.lines.b0 != -1
            fp_logit[not_root] += self.fp_logit[self.lines.b0[not_root]]
            tp = fp_logit < 0
            lines_p = softmax(
                self.lines_logit,
                index=torch.where(tp, self.lines.b1 + torch.where(self.lines.b1_dir, 1, 1 + self.n_branch), 0),
                num_nodes=2 * self.n_branch + 1,
            )
            return lines_p

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
                line_p=self.lines_p.float().numpy(force=True),
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
                opti_parent, opti_dir = digraph.solve_optimal_arborescence(detect_major_av_error=True)
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
            assert BranchDigraphData.has_gt(self.batch), (
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
            assert BranchDigraphData.has_gt(self.batch), "Ground truth line scores are not available in the batch data"
            return torch.cat([self.batch.edge_p, self.batch.branch_root_p[self.batch.branch_root_candidates]], dim=0)

        @property
        def gt_dir_p(self) -> Tensor:
            """Ground truth probability of each branch being oriented from tip0 to tip1 (values close to 1) or from tip1 to tip0 (values close to 0) as a tensor of shape (B,)"""  # noqa: E501
            assert BranchDigraphData.has_gt(self.batch), (
                "Ground truth branch directions are not available in the batch data"
            )
            return self.batch.branch_dir_p

        @property
        def gt_dir(self) -> Tensor:
            """Ground truth direction of each branch as a boolean tensor of shape (B,) (True: from tip0 to tip1, False: from tip1 to tip0)"""  # noqa: E501
            assert BranchDigraphData.has_gt(self.batch), (
                "Ground truth branch directions are not available in the batch data"
            )
            return self.gt_dir_p > 0.5

        @property
        def gt_fp_p(self) -> Tensor:
            """Ground truth probability of each branch being a false positive from the segmentation as a tensor of shape (B,)"""  # noqa: E501
            assert BranchDigraphData.has_gt(self.batch), (
                "Ground truth false positive probabilities are not available in the batch data"
            )
            return self.batch.branch_fp_p

        @property
        def gt_av_p(self) -> Tensor:
            """Ground truth probability of each branch being an artery (as opposed to a vein) as a tensor of shape (B,)"""  # noqa: E501
            assert BranchDigraphData.has_gt(self.batch), (
                "Ground truth artery probabilities are not available in the batch data"
            )
            return self.batch.branch_av_p

        @property
        def gt_subtree_idx(self) -> Tensor:
            """Tensor of shape (N_branch,) containing the ground truth subtree index of each branch"""  # noqa: E501
            assert BranchDigraphData.has_gt(self.batch), (
                "Ground truth subtree indices are not available in the batch data"
            )
            return self.batch.branch_subtree_idx

        def gt_tail_nodes(self, use_parent_head: bool = False) -> Tensor:
            """Indices of the node at the tail of each branch (according to gt_dir) as a tensor of shape (N_line,)."""
            assert BranchDigraphData.has_gt(self.batch), (
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


################################
# === Classification Heads === #
################################
class ClassifHead(torch.nn.Module):
    def forward(self, batch: BranchDigraphBatch, x: Tensor) -> BranchDigraphModel.Output:
        raise NotImplementedError

    def __call__(self, batch: BranchDigraphBatch, x: Tensor) -> BranchDigraphModel.Output:
        return super().__call__(batch, x)


class SimpleClassifHead(ClassifHead):
    def __init__(self, out_channels: int, affinity_embedding_dim: int = 128, oriented_affinity: bool = True):
        super().__init__()
        self.oriented_affinity = oriented_affinity

        # === Classification layers ===
        self.lin_fp = Linear(out_channels, 1, weight_initializer="glorot")
        self.lin_av = Linear(out_channels, 1, weight_initializer="glorot")
        self.lin_dir = Linear(out_channels, 1, weight_initializer="glorot")
        self.lin_root = Linear(out_channels, 1, weight_initializer="glorot")
        affinity_embedding_dim *= 2 if oriented_affinity else 1
        self.lin_affinity = Linear(out_channels, affinity_embedding_dim, weight_initializer="glorot")

    def forward(self, batch: BranchDigraphBatch, x: Tensor) -> BranchDigraphModel.Output:
        branch_fp = self.lin_fp(x).squeeze(-1)
        branch_av = self.lin_av(x).squeeze(-1)
        branch_dir = self.lin_dir(x).squeeze(-1)
        branch_affinity_v = self.lin_affinity(x)
        branch_root_score = self.lin_root(x).squeeze(-1)

        if self.oriented_affinity:
            b0_embedding, b1_embedding = branch_affinity_v.view(x.shape[0], -1, 2).unbind(-1)
        else:
            b0_embedding = b1_embedding = branch_affinity_v

        return BranchDigraphModel.Output(
            batch=batch,
            fp_logit=branch_fp,
            av_logit=branch_av,
            dir_logit=branch_dir,
            b0_embedding=b0_embedding,
            b1_embedding=b1_embedding,
            root_logit=branch_root_score,
        )


class PolarizedClassifHead(ClassifHead):
    def __init__(
        self,
        branch_channels: int,
        tip_channels: int,
        affinity_embedding_dim: int = 128,
        oriented_affinity: bool = True,
    ):
        super().__init__()
        self.oriented_affinity = oriented_affinity
        self.branch_channels = branch_channels
        self.tip_channels = tip_channels

        # === Classification layers ===
        both_channels = branch_channels + tip_channels
        self.lin_fp = Linear(both_channels, 1, weight_initializer="glorot")
        self.lin_av = Linear(branch_channels, 1, weight_initializer="glorot")
        self.lin_dir = Linear(both_channels, 1, weight_initializer="glorot")
        self.lin_root = Linear(both_channels, 1, weight_initializer="glorot")
        affinity_embedding_dim *= 2 if oriented_affinity else 1
        self.lin_affinity = Linear(both_channels, affinity_embedding_dim, weight_initializer="glorot")

    def forward(self, batch: BranchDigraphBatch, x: Tensor) -> BranchDigraphModel.Output:
        B, T = self.branch_channels, self.tip_channels
        x_branch, x_tip0, x_tip1 = x[:, :B], x[:, B : B + T], x[:, B + T :]
        x_tip = (x_tip0 + x_tip1) * 0.5
        x_branch_tip = torch.cat([x_branch, x_tip], dim=-1)
        branch_fp = self.lin_fp(x_branch_tip).squeeze(-1)
        branch_av = self.lin_av(x_branch).squeeze(-1)
        branch_dir = self.lin_dir(torch.cat([x_branch, (x_tip0 - x_tip1) / 2], dim=-1)).squeeze(-1)
        branch_root_score = self.lin_root(x_branch_tip).squeeze(-1)
        tip0_emb = self.lin_affinity(torch.cat([x_branch, x_tip0], dim=-1))
        tip1_emb = self.lin_affinity(torch.cat([x_branch, x_tip1], dim=-1))

        if self.oriented_affinity:
            b0_tip0_emb, b1_tip0_emb = tip0_emb.view(x_branch.shape[0], -1, 2).unbind(-1)
            b0_tip1_emb, b1_tip1_emb = tip1_emb.view(x_branch.shape[0], -1, 2).unbind(-1)
            b0_embedding = torch.stack([b0_tip0_emb, b0_tip1_emb], dim=-2)
            b1_embedding = torch.stack([b1_tip0_emb, b1_tip1_emb], dim=-2)
        else:
            b0_embedding = b1_embedding = torch.stack([tip0_emb, tip1_emb], dim=-2)

        return BranchDigraphModel.Output(
            batch=batch,
            fp_logit=branch_fp,
            av_logit=branch_av,
            dir_logit=branch_dir,
            b0_embedding=b0_embedding,
            b1_embedding=b1_embedding,
            root_logit=branch_root_score,
        )


###############################
# === Features Extractors === #
###############################
class BranchFeaturesEfficientNetV2S(torch.nn.Module):
    N_FEATURES = 392

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


######################
# === GCN Models === #
######################
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


###########################
# === Utils functions === #
###########################
def _max_parent(lines: DigraphLines, lines_score: Tensor, n_branch: int):
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

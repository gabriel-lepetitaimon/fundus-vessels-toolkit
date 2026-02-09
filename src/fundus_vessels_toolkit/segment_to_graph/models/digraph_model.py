from copy import copy
from functools import partial
from typing import Optional, TypedDict

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.conv import GATv2Conv, MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import softmax
from torchvision.models.efficientnet import FusedMBConv, efficientnet_v2_s

from fundus_vessels_toolkit.utils.torch import torch_interp_bilinear

from .convbn import ConvBN, NormSpec, PaddingSpec


class BranchFeaturesEfficientNetV2S(torch.nn.Module):
    def __init__(self):
        super().__init__()
        net = efficientnet_v2_s()
        self._features_cache = [None] * 4

        def hook(module, input, output, idx):
            self._features_cache[idx] = output

        self.net = nn.Sequential(*net.features[:-1])  # type: ignore
        for i, f_idx in enumerate([1, 2, 3, 6]):
            self.net[f_idx].register_forward_hook(partial(hook, idx=i))

    def forward(self, x):
        self.net(x)
        return copy(self._features_cache)


class Gatv2GCN(torch.nn.Module):
    def __init__(self, n_in: int = 512, n_out: int = 512):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out

        self.gat0 = GATv2Conv(n_in, 64, heads=8, dropout=0.1)
        self.gat1a = GATv2Conv(64 * 8, 128, heads=8, dropout=0.1)
        self.gat1b = GATv2Conv(128 * 8, 256, heads=4, dropout=0.1)
        self.gat2a = GATv2Conv(256 * 4, 128, heads=8, dropout=0.1)
        self.gat2b = GATv2Conv(128 * 8, 256, heads=4, dropout=0.1)
        self.gat3a = GATv2Conv(256 * 4, 512, heads=2, dropout=0.1)
        self.gat3b = GATv2Conv(512 * 2, n_out, heads=1, dropout=0.1)

    def forward(self, x, edge_index):
        x = self.gat0(x, edge_index).relu()
        x = self.gat1a(x, edge_index).relu()
        x = self.gat1b(x, edge_index).relu()
        x = self.gat2a(x, edge_index).relu()
        x = self.gat2b(x, edge_index).relu()
        x = self.gat3a(x, edge_index).relu()
        x = self.gat3b(x, edge_index)

        return x


class BranchDigraphModel(torch.nn.Module):
    def __init__(self, img_feature_extractor: nn.Module, gnn: nn.Module, gnn_out_channels: Optional[int] = None):
        super().__init__()

        self.img_feature_extractor = img_feature_extractor.half()
        self.gnn = gnn

        if gnn_out_channels is None:
            assert hasattr(gnn, "n_out"), "gnn_out_channels must be specified if gnn does not have n_out attribute"
            gnn_out_channels = int(gnn.n_out)  # type: ignore

        # === Classification layers ===
        self.lin_av = Linear(gnn_out_channels, 3, weight_initializer="glorot")
        self.lin_dir = Linear(gnn_out_channels, 1, weight_initializer="glorot")
        self.lin_parent = Linear(gnn_out_channels, 128, weight_initializer="glorot")
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
            features_map = sorted(features_map, key=lambda x: x.shape[-1], reverse=True)
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

    def forward(self, data):
        img = data.img.reshape(data.batch_size, 3, *data.img.shape[1:]).half()
        img_features = self.img_feature_extractor(img)
        branch_features = self.sample_features(img_features, data.branch_curves, data.batch, img.shape[-2:])

        edge_index = data.edge_index[:, ~root_lines_mask(data)]
        x = self.gnn(branch_features, edge_index)

        branch_av_p = self.lin_av(x)
        branch_dir = torch.sigmoid(self.lin_dir(x).squeeze(-1))
        branch_parent_v = self.lin_parent(x)
        branch_root_p = self.lin_root(x).squeeze(-1)

        dir = (data.branch_dir if self.training else branch_dir) > 0.5
        incident_edges_mask = incident_mask(data, dir)
        incident_edges = data.edge_index[:, incident_edges_mask]
        edge_scores = torch.zeros(data.edge_index.shape[1], device=edge_index.device)
        if incident_edges.numel() > 0:
            branch_out = branch_parent_v[incident_edges[0]]
            branch_in = branch_parent_v[incident_edges[1]]
            edge_scores[incident_edges_mask] = F.cosine_similarity(branch_out, branch_in, dim=-1)
            root_mask, root_target_branches = root_lines_index(data, dir)
            assert (incident_edges_mask & root_mask).sum() == 0, "Root edges should not be incident edges"
            if root_mask.any():
                edge_scores[root_mask] = branch_root_p[root_target_branches]
            edge_scores = softmax(edge_scores, groupby_incident_edges(data, dir))

        return branch_av_p, branch_dir, edge_scores


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

import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import GATv2Conv, MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import softmax

from .convbn import ConvBN


class SimpleGATGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # === Convolutional layers ===
        self.conv0 = ConvBN(3, 16, kernel=5, bn=True)

        self.conv1a = [ConvBN(16, 16, kernel=5, bn=True, dilation=_) for _ in [1, 2, 4]]
        self.conv1b = ConvBN(64, 64, kernel=3, bn=True)
        self.conv2a = [ConvBN(64, 32, kernel=5, bn=True, dilation=_) for _ in [1, 2, 4]]
        self.conv2b = ConvBN(128, 128, kernel=3, bn=True)
        self.conv3a = [ConvBN(128, 64, kernel=5, bn=True, dilation=_) for _ in [1, 2, 4]]
        self.conv3b = ConvBN(256, 256, kernel=3, bn=True)

        self.last_conv = ConvBN(256, 256, kernel=3, bn=True)

        # === GATv2 layers ===
        self.gat0 = GATv2Conv(512, 64, heads=8, dropout=0.1)
        self.gat1a = GATv2Conv(64 * 8, 128, heads=8, dropout=0.1)
        self.gat1b = GATv2Conv(128 * 8, 256, heads=4, dropout=0.1)
        self.gat2a = GATv2Conv(256 * 4, 128, heads=8, dropout=0.1)
        self.gat2b = GATv2Conv(128 * 8, 256, heads=4, dropout=0.1)
        self.gat3a = GATv2Conv(256 * 4, 512, heads=2, dropout=0.1)
        self.gat3b = GATv2Conv(512 * 2, 512, heads=1, dropout=0.1)

        self.lin_av = Linear(512, 3, weight_initializer="glorot")
        self.lin_dir = Linear(512, 1, weight_initializer="glorot")
        self.lin_parent = Linear(512, 128, weight_initializer="glorot")

    def forward_conv(self, x):
        x = self.conv0(x)

        x = torch.cat([x] + [conv(x) for conv in self.conv1a], dim=1)
        x = self.conv1b(x)

        x = torch.cat([x] + [conv(x) for conv in self.conv2a], dim=1)
        x = self.conv2b(x)

        x = torch.cat([x] + [conv(x) for conv in self.conv3a], dim=1)
        x = self.conv3b(x)

        x = self.last_conv(x)
        return x

    def sample_fundus_features(self, y_fundus, data):
        # Sample fundus features
        batch = data.batch[:, None]
        C = data.branch_curves.shape[1]
        curves_y_tip0, curves_x_tip0 = data.branch_curves[:, : C // 2].unbind(-1)
        curves_y_tip1, curves_x_tip1 = data.branch_curves[:, C // 2 :].unbind(-1)

        b_features0 = y_fundus[batch, :, curves_y_tip0, curves_x_tip0].mean(dim=-1)  # (B, 256)
        b_features1 = y_fundus[batch, :, curves_y_tip1, curves_x_tip1].mean(dim=-1)  # (B, 256)
        return torch.cat([b_features0, b_features1], dim=-1)  # (B, 512)

    def forward_gat(self, data, b_features):
        edge_index = data.edge_index

        x = b_features
        x = self.gat0(x, edge_index).relu()
        x = self.gat1a(x, edge_index).relu()
        x = self.gat1b(x, edge_index).relu()
        x = self.gat2a(x, edge_index).relu()
        x = self.gat2b(x, edge_index).relu()
        x = self.gat3a(x, edge_index).relu()
        x = self.gat3b(x, edge_index)

        return x

    def forward(self, data):
        fundus_img = data.fundus_img
        y_fundus = self.forward_conv(fundus_img)

        x = self.sample_fundus_features(y_fundus, data)
        x = self.forward_gat(data, x)

        branch_av_p = self.lin_av(x)
        branch_dir = torch.sigmoid(self.lin_dir(x).squeeze(-1))
        branch_parent_v = self.lin_parent(x)

        dir = (data.branch_dir if self.training else branch_dir) > 0.5
        incident_edges = data.edge_index[incident_mask(data, dir)]
        edge_scores = torch.zeros(data.edge_index.shape[1], device=data.edge_index.device)
        if incident_edges.numel() > 0:
            branch_out = branch_parent_v[incident_edges[0]]
            branch_in = branch_parent_v[incident_edges[1]]
            edge_scores[incident_edges] = F.cosine_similarity(branch_out, branch_in, dim=-1)
            edge_probs = softmax(edge_scores, data.edge_index[1])

        return branch_av_p, branch_dir, branch_parent


def incident_mask(data, branch_dir):
    edge_index = data.edge_index
    edge_first_tip = data.edge_first_tip

    source_branches = edge_index[0]
    target_branches = edge_index[1]

    source_first_tip = edge_first_tip[:, 0]
    target_first_tip = edge_first_tip[:, 1]

    source_dirs = branch_dir[source_branches]
    target_dirs = branch_dir[target_branches]

    source_out = source_first_tip ^ source_dirs  # if dir is True, the edge should be emitted from the second tip
    target_in = target_first_tip == target_dirs  # if dir is True, the edge should be incident to the first tip

    return source_out & target_in


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

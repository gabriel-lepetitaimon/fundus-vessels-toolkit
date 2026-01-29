import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear


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

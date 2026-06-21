import math
from typing import Annotated, Literal, Optional

import torch
import torch.nn.functional as F
from pydantic import Field, StringConstraints
from torch import Tensor
from torch.nn import ModuleDict
from torch_geometric import nn as pyg_nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import OptTensor
from torch_geometric.utils import softmax

from ...utils.nnet.experiment import ExpCfgBaseModel
from .positionnal_embedding import RoPE, SupportPattern, TransformerConvWithPosEncoding

type SupportPatternOrNone = SupportPattern | Literal["none"]


class TransformerGCNOpt(ExpCfgBaseModel):
    architecture: Annotated[
        str,
        StringConstraints(
            pattern=r"^(?:InstNorm|BatchNorm|Conv\d+(?:x\d+)?(?:-DropOut)?)(?:\s+(?:InstNorm|BatchNorm|Conv\d+(?:x\d+)?(?:-DropOut)?))*$"
        ),
    ] = Field(default="InstNorm Conv64x8-DropOut InstNorm Conv128x8-DropOut Conv256x4 Conv128x8 Conv256x4 Conv512x2")
    """Model architecture as a string. 
    The syntax is a sequence of layers separated by spaces and using the following format:
     - "InstNorm" for instance normalization
     - "BatchNorm" for batch normalization
     - "Conv{out}x{heads}[-DropOut]" for a transformer convolution layer with {out} output channels and {heads} attention heads. If "x{heads}" is omitted, it defaults to 1 head. If "-DropOut" is present, dropout with the specified rate will be applied after the convolution.
     """  # noqa: E501

    dropout: float = Field(default=0.1, ge=0.0, le=1.0)
    """Dropout rate to apply after convolution layers that have the "-DropOut" suffix in the architecture string."""

    bipolar_node: bool = Field(default=True)
    """Whether to use bipolar nodes extending the state of every node with two additional feature vectors representing their two poles. If True, the model will use BipolarTransformerConv layers and the output dimension will be split between nodes and poles features."""  # noqa: E501

    total_out_features: int = Field(default=512, ge=1)
    """The total number of output features for the GNN. If bipolar_node is False, this will be the dimension of the node features output by the GNN. If bipolar_node is True, this will be the sum of the dimensions of the node features and the two pole features output by the GNN."""  # noqa: E501

    pole_features_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    """Ratio of the number of features dedicated to pole over the total number of features (including both pole and node). Only relevant if bipolar_node is True. For example, if total_n_out=100 and pole_features_ratio=0.66, then 66 features will be dedicated to poles (33 for each) and 33 features will be dedicated to nodes."""  # noqa: E501

    pos_encoding: SupportPatternOrNone = Field(default="spiral")
    """The type of positional encoding to use. If "none", no positional encoding will be used. Otherwise, should be a support pattern supported by RoPESupportPattern, which will be used to compute RoPE positional encodings based on the relative positions of the nodes' poles."""  # noqa: E501

    @property
    def n_out(self) -> int:
        if self.bipolar_node:
            return int(self.total_out_features * (1 - self.pole_features_ratio))
        else:
            return self.total_out_features

    @property
    def n_out_pole(self) -> int:
        return int(self.total_out_features * self.pole_features_ratio / 2) if self.bipolar_node else 0


class TransformerGCN(torch.nn.Module):
    def __init__(
        self,
        n_in: int = 512,
        edge_attr_dim: Optional[int] = None,
        opt: Optional[TransformerGCNOpt] = None,
    ):
        super().__init__()
        # --- Save hyperparameters ---
        if opt is None:
            opt = TransformerGCNOpt()

        self.n_in = n_in
        self.edge_attr_dim = edge_attr_dim
        self.opt = opt

        # --- Create layers based on architecture specification string ---
        def ConvBlock(in_channels, out_channels, heads, dropout: float = 0, first=False):
            if opt.bipolar_node:
                if first:
                    in_channels_pole = in_channels_node = in_channels // 3
                else:
                    in_channels_pole = int((in_channels * opt.pole_features_ratio) / 2)
                    in_channels_node = in_channels - 2 * in_channels_pole
                out_channels_pole = int((out_channels * opt.pole_features_ratio) / 2)
                out_channels_node = out_channels - 2 * out_channels_pole
                conv = BipolarTransformerConv(
                    in_channels_node=in_channels_node,
                    in_channels_pole=in_channels_pole,
                    out_channels_node=out_channels_node,
                    out_channels_pole=out_channels_pole,
                    heads=heads,
                    dropout=dropout,
                    edge_dim=edge_attr_dim,
                    pos_encoding=opt.pos_encoding,
                    beta=True,
                )
            else:
                conv = TransformerConvWithPosEncoding(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=heads,
                    dropout=dropout,
                    edge_dim=edge_attr_dim,
                    pos_encoding=opt.pos_encoding,
                    beta=True,
                )
            return conv

        self.layers = ModuleDict()
        layers_count = {}
        f = n_in
        for layer_spec in opt.architecture.split():
            if layer_spec == "InstNorm":
                layers_count["in"] = i = layers_count.setdefault("in", -1) + 1
                self.layers[f"in{i}"] = pyg_nn.InstanceNorm(f)
            elif layer_spec == "BatchNorm":
                layers_count["bn"] = i = layers_count.setdefault("bn", -1) + 1
                self.layers[f"bn{i}"] = pyg_nn.BatchNorm(f)
            elif layer_spec.startswith("Conv"):
                l_out = layer_spec[4:]
                if dropout := l_out.endswith("-DropOut"):
                    l_out = l_out[: -len("-DropOut")]
                try:
                    if "x" in l_out:
                        out_channels, heads = l_out.split("x")
                        out_channels, heads = int(out_channels), int(heads)
                    else:
                        out_channels, heads = int(l_out), 1
                except ValueError:
                    raise ValueError(f"Invalid layer specification: {layer_spec}") from None

                layers_count["conv"] = i = layers_count.setdefault("conv", -1) + 1
                dropout = opt.dropout if dropout else 0
                self.layers[f"conv{i}"] = ConvBlock(f, out_channels, heads, dropout=dropout, first=(i == 0))
                f = out_channels * heads
            else:
                raise ValueError(f"Invalid layer specification: {layer_spec}")

        self.last_conv = ConvBlock(f, opt.total_out_features, heads=1, dropout=0)

        if opt.bipolar_node:
            self.n_out = int(self.last_conv.out_channels_node)  # type: ignore
            self.n_out_pole = int(self.last_conv.out_channels_pole)  # type: ignore
        else:
            self.n_out = int(opt.total_out_features)
            self.n_out_pole = 0

    def forward(self, x, edge_index, edge_pole, batch_idx, batch_size, edge_attr=None, pos=None):
        if pos is not None:
            assert pos.shape == (x.shape[0], 2, 2), "Pos must have shape [N, 2, 2]"
            pos = pos.unbind(1)
        for name, layer in list(self.layers.items()):
            match layer:
                case pyg_nn.InstanceNorm():
                    x = layer(x, batch_idx, batch_size=batch_size)
                case BipolarTransformerConv():
                    x = layer(x, edge_index, edge_pole, edge_attr=edge_attr, pos=pos).relu()
                case TransformerConvWithPosEncoding():
                    x = layer(x, edge_index, edge_attr=edge_attr, pos=pos).relu()

        match self.last_conv:
            case BipolarTransformerConv():
                x = self.last_conv(x, edge_index, edge_pole, edge_attr=edge_attr, pos=pos)
            case TransformerConvWithPosEncoding():
                x = self.last_conv(x, edge_index, edge_attr=edge_attr, pos=pos)

        return x


class BipolarTransformerConv(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels_node: int,
        in_channels_pole: int,
        out_channels_node: int,
        out_channels_pole: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        *,
        pos_encoding: RoPE | SupportPattern | Literal["none"] = "none",  # Dimension of positional encoding
        **kwargs,
    ):
        super().__init__(node_dim=0, aggr="add", **kwargs)

        self.in_channels_node = in_channels_node
        self.in_channels_pole = in_channels_pole
        self.out_channels_node = out_channels_node
        self.out_channels_pole = out_channels_pole
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        if isinstance(pos_encoding, str) and pos_encoding != "none":
            self.pos_encoding = RoPE(out_channels_node + out_channels_pole, support_pattern=pos_encoding)
        else:
            self.pos_encoding = pos_encoding

        # === PARAMETERS ===
        in_node, in_pole = in_channels_node, in_channels_pole
        H, out_node, out_pole = heads, out_channels_node, out_channels_pole

        self.lin_node_key = Linear(in_node, H * (out_node + out_pole), bias=bias)
        self.lin_pole_key = Linear(in_pole, H * (out_node + out_pole), bias=bias)
        self.lin_node_query = Linear(in_node, H * (out_node + out_pole), bias=bias)
        self.lin_pole_query = Linear(in_pole, H * (out_node + out_pole), bias=bias)
        self.lin_node_value_node = Linear(in_node, H * out_node, bias=bias)
        self.lin_node_value_pole = Linear(in_pole, H * out_node, bias=bias)
        self.lin_pole_value_node = Linear(in_node, H * out_pole, bias=bias)
        self.lin_pole_value_pole = Linear(in_pole, H * out_pole, bias=bias)

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, H * (out_node + out_pole), bias=False)
        else:
            self.lin_edge = self.register_parameter("lin_edge", None)

        concat_heads = heads if concat else 1
        if root_weight:
            self.lin_node_skip = Linear(in_node, out_node * concat_heads, bias=bias)
            # self.lin_pole_skip_node = Linear(in_node, out_pole * concat_heads, bias=bias)
            self.lin_pole_skip_pole = Linear(in_pole, out_pole * concat_heads, bias=bias)
        else:
            self.lin_node_skip = self.register_parameter("lin_skip_node", None)
            # self.lin_pole_skip_node = self.register_parameter("lin_skip_node", None)
            self.lin_pole_skip_pole = self.register_parameter("lin_skip_pole", None)
        if self.beta:
            self.lin_beta_node = Linear(3 * out_node * concat_heads, 1, bias=False)
            self.lin_beta_pole = Linear(3 * out_pole * concat_heads, 1, bias=False)
        else:
            self.lin_beta_node = self.register_parameter("lin_beta_node", None)
            self.lin_beta_pole = self.register_parameter("lin_beta_pole", None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()

        self.lin_node_key.reset_parameters()
        self.lin_pole_key.reset_parameters()
        self.lin_node_query.reset_parameters()
        self.lin_pole_query.reset_parameters()
        self.lin_node_value_node.reset_parameters()
        self.lin_node_value_pole.reset_parameters()
        self.lin_pole_value_node.reset_parameters()
        self.lin_pole_value_pole.reset_parameters()

        n, p = self.in_channels_node, self.in_channels_pole
        for param in [self.lin_node_key, self.lin_node_query, self.lin_node_value_node, self.lin_pole_value_node]:
            param.weight.data *= n / (n + p)
        for param in [self.lin_pole_key, self.lin_pole_query, self.lin_node_value_pole, self.lin_pole_value_pole]:
            param.weight.data *= p / (n + p)

        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()

        if self.lin_node_skip is not None:
            self.lin_node_skip.reset_parameters()
        if self.lin_pole_skip_pole is not None:
            self.lin_pole_skip_pole.reset_parameters()
            # self.lin_pole_skip_pole.weight.data *= 2 * p / (n + 2 * p)
        # if self.lin_pole_skip_node is not None:
        #    self.lin_pole_skip_node.reset_parameters()
        #    self.lin_pole_skip_node.weight.data *= n / (n + 2 * p)
        if self.lin_beta_node is not None:
            self.lin_beta_node.reset_parameters()
        if self.lin_beta_pole is not None:
            self.lin_beta_pole.reset_parameters()

    def stack_x(self, x_branch: Tensor, x_tip0: Tensor, x_tip1: Tensor) -> Tensor:
        return torch.cat([x_branch, x_tip0, x_tip1], dim=-1)

    def unstack_x(self, x: Tensor, out=False) -> tuple[Tensor, Tensor, Tensor]:
        if out:
            N = self.out_channels_node * (self.heads if self.concat else 1)
            P = self.out_channels_pole * (self.heads if self.concat else 1)
        else:
            N, P = self.in_channels_node, self.in_channels_pole
        assert x.shape[-1] == N + 2 * P, f"Last dimension of x should be {N + 2 * P}, but got {x.shape[-1]}"
        return x[..., :N], x[..., N : N + P], x[..., N + P :]

    def forward(  # noqa: F811
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_pole: Tensor,
        edge_attr: OptTensor = None,
        pos: Optional[tuple[Tensor, Tensor]] = None,
    ) -> Tensor:
        r"""Runs the forward pass of the module.  # noqa: E501

        Parameters
        ----------
        x : Tensor [N, in_channels + 2*in_channels_tip]
            Stacked features of the nodes and their two poles.
        edge_index : Tensor [2, E]
            The edge indices.
        edge_pole: Tensor [E, 2]
            The indices of the source and target poles for each edge.
        edge_attr : Tensor [E, edge_dim], optional
            The edge features.
        pos : (Tensor, Tensor) [N, 2], optional
            The two pole positions for each node, used to computing positional encodings.
        """
        H, N, P = self.heads, self.out_channels_node, self.out_channels_pole
        x_node, x_pol0, x_pol1 = self.unstack_x(x)

        query_node = self.lin_node_query(x_node).view(-1, H, N + P)
        query_p0 = self.lin_pole_query(x_pol0).view(-1, H, N + P) + query_node
        query_p1 = self.lin_pole_query(x_pol1).view(-1, H, N + P) + query_node

        key_node = self.lin_node_key(x_node).view(-1, H, N + P)
        key_p0 = self.lin_pole_key(x_pol0).view(-1, H, N + P) + key_node
        key_p1 = self.lin_pole_key(x_pol1).view(-1, H, N + P) + key_node

        node_value_node = self.lin_node_value_node(x_node).view(-1, H, N)
        node_value_p0 = self.lin_node_value_pole(x_pol0).view(-1, H, N) + node_value_node
        node_value_p1 = self.lin_node_value_pole(x_pol1).view(-1, H, N) + node_value_node
        pole_value_node = self.lin_pole_value_node(x_node).view(-1, H, P)
        pole_value_p0 = self.lin_pole_value_pole(x_pol0).view(-1, H, P) + pole_value_node
        pole_value_p1 = self.lin_pole_value_pole(x_pol1).view(-1, H, P) + pole_value_node

        edge_value = self.lin_edge(edge_attr).view(-1, H, N + P) if self.lin_edge is not None else None

        if pos is not None and isinstance(self.pos_encoding, RoPE):
            assert pos[0].shape == pos[1].shape == (x.shape[0], 2), "pos[0] and pos[1] must have shape [N, 2]"
            pos_emb = self.pos_encoding.compute_freqs_from_pos(pos[0])
            query_p0 = self.pos_encoding.apply_rot_emb(query_p0, pos_emb)
            key_p0 = self.pos_encoding.apply_rot_emb(key_p0, pos_emb)

            pos_emb = self.pos_encoding.compute_freqs_from_pos(pos[1])
            query_p1 = self.pos_encoding.apply_rot_emb(query_p1, pos_emb)
            key_p1 = self.pos_encoding.apply_rot_emb(key_p1, pos_emb)

        y = self.propagate(
            edge_index,
            edge_pole=edge_pole,
            query=torch.stack([query_p0, query_p1], dim=1),
            key=torch.stack([key_p0, key_p1], dim=1),
            node_value=torch.stack([node_value_p0, node_value_p1], dim=1),
            pole_value=torch.stack([pole_value_p0, pole_value_p1], dim=1),
            edge_value=edge_value,
        )
        if self.concat:
            y = y.permute(0, 2, 1).reshape(-1, (N + 2 * P) * H)
        else:
            y = y.mean(dim=1)

        if self.root_weight:
            y_node, y_pol0, y_pol1 = self.unstack_x(y, out=True)

            if self.lin_node_skip:
                # yr_node = self.lin_node_skip(torch.cat([x_node, (x_pol0 + x_pol1) / 2], dim=-1))
                yr_node = self.lin_node_skip(x_node)
                if self.lin_beta_node is not None:
                    beta_node = self.lin_beta_node(torch.cat([y_node, yr_node, y_node - yr_node], dim=-1)).sigmoid()
                    y_node = beta_node * yr_node + (1 - beta_node) * y_node
                else:
                    y_node = y + yr_node

            if self.lin_pole_skip_pole:  # and self.lin_pole_skip_node:
                # yr_pole_node = self.lin_pole_skip_node(x_node)
                yr_pol0 = self.lin_pole_skip_pole(x_pol0)  # + yr_pole_node
                yr_pol1 = self.lin_pole_skip_pole(x_pol1)  # + yr_pole_node
                if self.lin_beta_pole is not None:
                    beta_pol1 = self.lin_beta_pole(torch.cat([y_pol0, yr_pol0, y_pol0 - yr_pol0], dim=-1)).sigmoid()
                    beta_pol2 = self.lin_beta_pole(torch.cat([y_pol1, yr_pol1, y_pol1 - yr_pol1], dim=-1)).sigmoid()
                    y_pol0 = beta_pol1 * yr_pol0 + (1 - beta_pol1) * y_pol0
                    y_pol1 = beta_pol2 * yr_pol1 + (1 - beta_pol2) * y_pol1
                else:
                    y_pol0 = y_pol0 + yr_pol0
                    y_pol1 = y_pol1 + yr_pol1

            y = self.stack_x(y_node, y_pol0, y_pol1)

        return y

    def message(
        self,
        query_i: Tensor,
        key_j: Tensor,
        pole_value_j: Tensor,
        node_value_j: Tensor,
        edge_pole: Tensor,
        edge_value: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> Tensor:
        # For each edge j->i, select the query, key and value corresponding poles of j and i
        E = torch.arange(edge_pole.size(0), device=edge_pole.device)
        query_i = query_i[E, edge_pole[:, 1]]  # [E, H, N+P]
        key_j = key_j[E, edge_pole[:, 0]]  # [E, H, N+P]
        node_value_j = node_value_j[E, edge_pole[:, 0]]  # [E, H, N]
        pole_value_i = torch.zeros_like(pole_value_j)  # [E, 2, H, P]
        pole_value_j = pole_value_j[E, edge_pole[:, 0]]  # [E, H, P]

        if self.lin_edge is not None and edge_value is not None:
            key_j = key_j + edge_value

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels_node + self.out_channels_pole)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        if edge_value is not None:
            node_value_j = node_value_j + edge_value[..., : self.out_channels_node]
            pole_value_j = pole_value_j + edge_value[..., self.out_channels_node :]

        pole_value_i[E, edge_pole[:, 1]] = pole_value_j
        y = torch.cat([node_value_j, pole_value_i[:, 0], pole_value_i[:, 1]], dim=2)  # [E, H, N+2P]

        y = y * alpha.view(-1, self.heads, 1)
        return y

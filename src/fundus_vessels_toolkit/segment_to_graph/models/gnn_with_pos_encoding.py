import math
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.conv import GATv2Conv, MessagePassing, TransformerConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import OptTensor
from torch_geometric.utils import softmax


class APE(torch.nn.Module):
    freqs_support: Tensor

    def __init__(self, head_dim: int, *, support_pattern: Literal["axial", "spiral"] = "axial", max_pos: int = 1000):
        super().__init__()
        self.head_dim = head_dim
        self.support_pattern = support_pattern
        self.max_pos = max_pos

        self.register_buffer("freqs_support", self.compute_freqs_support(), persistent=False)

    def compute_freqs_support(self) -> Tensor:
        """
        Compute the unitary vector along which the 2d position will be projected to get the positional encoding.
        """
        N = self.head_dim // 2

        thetas = self.max_pos ** ((-2 / self.head_dim) * torch.arange(0, N))
        if self.support_pattern == "axial":
            v = torch.tensor([[1, 0], [0, 1]]).repeat(math.ceil(N / 2), 1)
            return v[:N] * thetas[:, None]
        elif self.support_pattern == "spiral":
            angle = torch.linspace(0, torch.pi / 2, steps=math.ceil(N / 2) + 1)[:-1]
            cos, sin = torch.cos(angle), torch.sin(angle)
            v = torch.tensor([[cos, -sin], [sin, cos]]).permute((2, 0, 1)).reshape(-1, 2)
            return v[:N] * thetas[:, None]
        else:
            raise ValueError(f"Unknown support pattern: {self.support_pattern}")

    def compute_pos_encoding(self, pos: Tensor) -> Tensor:
        """
        Compute positional encoding for given positions and embedding dimension.

        Parameters
        ----------
        pos : Tensor
            A tensor of shape (n_pos, 2) containing the 2d positions for which to compute the encoding.

        Returns
        -------
        Tensor
            A tensor of shape (n_pos, head_dim) containing the positional encodings for each position.
        """
        freqs = pos @ self.freqs_support.T  # [n_pos, 2] @ [2, head_dim // 2] -> [n_pos, head_dim // 2]
        pos_emb = torch.stack((torch.sin(freqs), torch.cos(freqs)), dim=1)  # [n_pos, 2, head_dim // 2]
        return pos_emb.reshape(-1, self.head_dim)  # n_pos, head_dim


class RoPE(torch.nn.Module):
    freqs_support: Tensor

    def __init__(self, head_dim: int, *, support_pattern: Literal["axial", "spiral"] = "axial", max_pos: int = 1000):
        super().__init__()
        self.head_dim = head_dim
        self.support_pattern = support_pattern
        self.max_pos = max_pos
        self._last_pos: Optional[Tensor] = None
        self._last_freqs: Optional[Tensor | tuple] = None

        self.register_buffer("freqs_support", self.compute_freqs_support(), persistent=False)

    def compute_freqs_support(self) -> Tensor:
        """
        Compute the unitary vector along which the 2d position will be projected to get the positional encoding.
        """
        half_head_dim = self.head_dim // 2
        qurt_head_dim = math.ceil(half_head_dim / 2)

        thetas = self.max_pos ** ((-2 / self.head_dim) * torch.arange(0, half_head_dim))
        if self.support_pattern == "axial":
            v = torch.tensor([[1, 0], [0, 1]]).repeat(qurt_head_dim, 1)
            return v[:half_head_dim] * thetas[:, None]
        elif self.support_pattern == "spiral":
            angle = torch.linspace(0, torch.pi / 2, steps=qurt_head_dim + 1)[:-1]
            cos, sin = torch.cos(angle), torch.sin(angle)
            v = torch.stack([cos, -sin, sin, cos], dim=1).reshape(-1, 2)
            return v[:half_head_dim] * thetas[:, None]
        else:
            raise ValueError(f"Unknown support pattern: {self.support_pattern}")

    def get_freqs_cis(self, theta, n_embd, n_heads, ctx_size) -> Tensor:
        head_dim = n_embd // n_heads
        i = torch.arange(head_dim // 2)
        thetas = theta ** (-2 * i / head_dim)  # head_dim // 2
        pos = torch.arange(ctx_size)  # pos
        freqs = torch.outer(pos, thetas)  # pos, head_dim // 2
        return torch.complex(torch.cos(freqs), torch.sin(freqs))

    def apply_rot_emb(self, x: Tensor, pos: Tensor) -> Tensor:
        """Apply RoPE positional encoding to the input tensor x based on the positions pos.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (n_branch, n_heads, head_dim) containing the query/key/value vectors.

        pos : Tensor
            Tensor of shape (n_branch, 2) containing the 2d positions corresponding to each branch.

        Examples
        --------
        >>> x = torch.randn(4, 8, 64)  # n_branch=2, n_heads=8, head_dim=64
        >>> pos = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)  # n_branch*2=4
        >>> rope = RoPE(head_dim=64, support_pattern="axial", max_pos=1000)
        >>> x_float32 = rope.apply_rot_emb(x, pos)
        >>> x_float16 = rope.to(torch.bfloat16).apply_rot_emb(x.to(torch.bfloat16), pos.to(torch.bfloat16)).to(torch.float32)
        >>> torch.allclose(x_float16, x_float32, atol=5e-2)
        True
        """  # noqa: E501
        n_pos, n_heads, head_dim = x.shape
        half = head_dim // 2
        assert head_dim == self.head_dim, f"Input head dim ({head_dim}) should be {self.head_dim}"

        if self._last_pos is pos and self._last_freqs is not None:
            freqs = self._last_freqs
        else:
            freqs = pos @ self.freqs_support.T  # [n_pos, 2] @ [2, half] -> [n_pos, half]
            if freqs.dtype == torch.bfloat16:  # === Complex computation is not implemented for bfloat16 ===
                freqs = (torch.cos(freqs), torch.sin(freqs))
            else:
                freqs = torch.complex(torch.cos(freqs), torch.sin(freqs))  # [n_pos, half]
            self._last_pos = pos
            self._last_freqs = freqs

        x = x.reshape(n_pos, n_heads, half, 2)
        if isinstance(freqs, tuple):
            cos, sin = freqs
            v = torch.stack([cos, -sin, sin, cos], dim=-1).view(n_pos, 1, half, 2, 2)
            x_real = torch.sum(v * x.view(n_pos, n_heads, half, 1, 2), dim=-1)  # n_pos, n_heads, half, 2
        else:
            x_rot = torch.view_as_complex(x) * freqs.view(n_pos, 1, half)  # n_pos, n_heads, half
            x_real = torch.view_as_real(x_rot)  # bsz, n_heads, seq_len, head_dim // 2, 2
        return x_real.reshape(n_pos, n_heads, head_dim)


class TransformerConvWithPosEncoding(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: int | tuple[int, int],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        *,
        pos_encoding: Optional[RoPE | Literal["axial", "spiral"]] = None,  # Dimension of positional encoding
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        if isinstance(pos_encoding, str):
            pos_encoding = RoPE(out_channels, support_pattern=pos_encoding)
        self.pos_encoding = pos_encoding

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels, bias=bias)
        self.lin_query = Linear(in_channels[1], heads * out_channels, bias=bias)
        self.lin_value = Linear(in_channels[0], heads * out_channels, bias=bias)
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter("lin_edge", None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter("lin_beta", None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter("lin_beta", None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.lin_beta is not None:
            self.lin_beta.reset_parameters()

    def forward(  # noqa: F811
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_pole: OptTensor = None,
        edge_attr: OptTensor = None,
        pos: OptTensor = None,
    ) -> Tensor:
        r"""Runs the forward pass of the module.

        Parameters
        ----------
        x : Tensor [N, in_channels]
            The node features
        edge_index : Tensor [2, E]
            The edge indices.
        edge_attr : Tensor [E, edge_dim], optional
            The edge features. (default: :obj:`None`)
        pos : Tensor [N, 2] or [2, N, 2], optional
            The node positions for computing positional encodings. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        query = self.lin_query(x).view(-1, H, C)
        key = self.lin_key(x).view(-1, H, C)
        value = self.lin_value(x).view(-1, H, C)

        if pos is not None and self.pos_encoding is not None:
            assert pos.shape[-2] == x.shape[0] and pos.shape[-1] == 2, "Pos must have shape [N, 2] or [2, N, 2]"
            if pos.dim() == 3:
                pos0, pos1 = pos.unbind(dim=0)
            elif pos.dim() == 2:
                pos0 = pos1 = pos
            else:
                raise ValueError("Pos must have shape [N, 2] or [2, N, 2]")
            query = self.pos_encoding.apply_rot_emb(query, pos0)
            key = self.pos_encoding.apply_rot_emb(key, pos1)

        out = self.propagate(edge_index, query=query, key=key, value=value, edge_attr=edge_attr)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x)
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out = out + x_r

        return out

    def message(
        self,
        query_i: Tensor,
        key_j: Tensor,
        value_j: Tensor,
        edge_attr: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> Tensor:
        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            key_j = key_j + edge_attr  # type: ignore

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        if edge_attr is not None:
            out = out + edge_attr

        out = out * alpha.view(-1, self.heads, 1)
        return out


class PolarizedTransformerConvWithPosEncoding(MessagePassing):
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
        pos_encoding: Optional[RoPE | Literal["axial", "spiral"]] = None,  # Dimension of positional encoding
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)

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
        if isinstance(pos_encoding, str):
            pos_encoding = RoPE(out_channels_node + out_channels_pole, support_pattern=pos_encoding)
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
        pos: OptTensor = None,
    ) -> Tensor:
        r"""Runs the forward pass of the module.

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
        pos : Tensor [2, N, 2], optional
            The pole positions for each node, used to computing positional encodings.
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

        if pos is not None and self.pos_encoding is not None:
            assert pos.shape == (2, x.shape[0], 2), "Pos must have shape [2, N, 2]"
            pos0, pos1 = pos.unbind(dim=0)
            query_p0 = self.pos_encoding.apply_rot_emb(query_p0, pos0)
            query_p1 = self.pos_encoding.apply_rot_emb(query_p1, pos1)
            key_p0 = self.pos_encoding.apply_rot_emb(key_p0, pos0)
            key_p1 = self.pos_encoding.apply_rot_emb(key_p1, pos1)

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

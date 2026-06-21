import math
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import OptTensor
from torch_geometric.utils import softmax

SupportPattern = Literal["axial", "spiral"]


class APE(torch.nn.Module):
    freqs_support: Tensor

    def __init__(self, head_dim: int, *, support_pattern: SupportPattern = "axial", max_pos: int = 1000):
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

    def __init__(self, head_dim: int, *, support_pattern: SupportPattern = "axial", max_pos: int = 1000):
        super().__init__()
        self.head_dim = head_dim
        self.support_pattern = support_pattern
        self.max_pos = max_pos

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

    # def get_freqs_cis(self, theta, n_embd, n_heads, ctx_size) -> Tensor:
    #     head_dim = n_embd // n_heads
    #     i = torch.arange(head_dim // 2)
    #     thetas = theta ** (-2 * i / head_dim)  # head_dim // 2
    #     pos = torch.arange(ctx_size)  # pos
    #     freqs = torch.outer(pos, thetas)  # pos, head_dim // 2
    #     return torch.complex(torch.cos(freqs), torch.sin(freqs))

    def compute_freqs_from_pos(self, pos: Tensor) -> Tensor:
        COMPLEX_IMPLEMENTATION = pos.dtype != torch.bfloat16
        freqs = pos @ self.freqs_support.T  # [n_pos, 2] @ [2, head_dim // 2] -> [n_pos, head_dim // 2]
        if not COMPLEX_IMPLEMENTATION:  # Complex computation is not implemented for bfloat16
            return torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=0)  # [2, n_pos, head_dim // 2]
        else:
            return torch.complex(torch.cos(freqs), torch.sin(freqs))  # [n_pos, head_dim // 2]

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
        x = x.reshape(n_pos, n_heads, half, 2)

        COMPLEX_IMPLEMENTATION = pos.dtype != torch.bfloat16

        if pos.shape == ((n_pos, half) if COMPLEX_IMPLEMENTATION else (2, n_pos, half)):
            freqs = pos
        else:
            freqs = self.compute_freqs_from_pos(pos)
        if not COMPLEX_IMPLEMENTATION:  # Complex computation is not implemented for bfloat16
            cos, sin = freqs[0], freqs[1]  # [n_pos, head_dim // 2]
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
        pos_encoding: RoPE | SupportPattern | Literal["none"] = "none",  # Dimension of positional encoding
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
        if isinstance(pos_encoding, str) and pos_encoding != "none":
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
        pos : Tensor or tuple[Tensor, Tensor] [N, 2], optional
            The node positions for computing positional encodings. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        query = self.lin_query(x).view(-1, H, C)
        key = self.lin_key(x).view(-1, H, C)
        value = self.lin_value(x).view(-1, H, C)

        ### --- MODIFICATION FOR POSITIONAL ENCODING
        if pos is not None and isinstance(self.pos_encoding, RoPE):
            if isinstance(pos, tuple):
                pos0, pos1 = pos
            else:
                pos0 = pos1 = pos
            assert pos0.shape == pos1.shape == (x.shape[0], 2), "Pos must be of shape [N, 2]"
            query = self.pos_encoding.apply_rot_emb(query, pos0)
            key = self.pos_encoding.apply_rot_emb(key, pos1)
        ### ---

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

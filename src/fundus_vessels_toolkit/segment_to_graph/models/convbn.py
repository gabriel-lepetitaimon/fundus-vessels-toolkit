from __future__ import annotations

from typing import Literal, TypeAlias

import torch

NormSpec: TypeAlias = Literal[False, "batch", "instance", "mixed"]


class ConvBN(torch.nn.Module):
    def __init__(
        self,
        n_in,
        n_out=None,
        kernel=3,
        stride=1,
        relu=True,
        padding: PaddingSpec = 0,
        dilation=1,
        norm: NormSpec = False,
    ):
        super().__init__()

        if n_out is None:
            n_out = n_in
        if isinstance(kernel, int):
            kernel = kernel, kernel
        padding = compute_padding(padding, kernel, dilation)

        self.kernel = kernel
        self.n_in = n_in
        self.n_out = n_out
        self._relu = relu
        self._norm = norm

        conv = torch.nn.Conv2d(
            n_in, n_out, kernel_size=kernel, stride=stride, padding=padding, dilation=dilation, bias=not self._bn
        )
        torch.nn.init.kaiming_normal_(
            conv.weight, mode="fan_out", nonlinearity=("relu" if self._bn else "selu") if self._relu else "linear"
        )
        if conv.bias is not None:
            torch.nn.init.constant_(conv.bias, 0)

        model = [conv]
        if norm:
            if norm == "batch":
                model += [torch.nn.BatchNorm2d(n_out)]
            elif norm == "instance":
                model += [torch.nn.InstanceNorm2d(n_out)]
            elif norm == "mixed":
                model += [MixedNorm(n_out)]
            if relu:
                model += [torch.nn.ReLU()]
        elif relu:
            model += [torch.nn.SELU()]
        self.model = torch.nn.Sequential(*model)

        self.stride = stride
        self.padding = padding
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)

    def forward(self, x):
        return self.model(x)

    @property
    def conv(self):
        return self.model[0]

    @property
    def bn(self):
        if self._bn:
            return self.model[1]
        return None

    @property
    def relu(self):
        return self.model[2 if self._bn else 1]

    def __getattr__(self, item):
        if item in ("stride", "padding", "dilation"):
            return getattr(self.conv, item)
        return super().__getattr__(item)

    def __setattr__(self, key, value):
        if key in ("stride", "padding", "dilation"):
            setattr(self.conv, key, value)
        else:
            super().__setattr__(key, value)


# --- Utils function ---
PaddingSpec: TypeAlias = Literal["same", "auto", "true", "valid", "full"] | int | tuple[int, int]


def compute_padding(padding: PaddingSpec, kernel_shape, dilation: int = 1) -> tuple[int, int]:
    if padding == "same" or padding == "auto":
        hW, wW = kernel_shape[-2:]
        padding = ((hW // 2) * dilation, (wW // 2) * dilation)
    elif padding == "true" or padding == "valid":
        padding = (0, 0)
    elif padding == "full":
        hW, wW = kernel_shape[-2:]
        hW = hW + (hW - 1) * (dilation - 1)
        wW = wW + (wW - 1) * (dilation - 1)
        padding = (hW - hW % 2, wW - wW % 2)
    elif isinstance(padding, int):
        padding = (padding, padding)
    return padding


def compute_conv_outputs_dim(
    input_shape, weight_shape, padding=0, output_padding=0, stride=1, dilation=1, transpose=False
):
    h, w = input_shape[-2:]
    n, m = weight_shape[-2:]

    if not isinstance(padding, tuple):
        padding = compute_padding(padding, weight_shape, dilation)
    if not isinstance(output_padding, tuple):
        output_padding = compute_padding(output_padding, weight_shape, dilation=dilation)
    if isinstance(stride, int):
        stride = stride, stride
    if isinstance(dilation, int):
        dilation = dilation, dilation
    if not transpose:
        h = int((h + 2 * padding[0] - dilation[0] * (n - 1) - 1) / stride[0] + 1)
        w = int((w + 2 * padding[1] - dilation[1] * (m - 1) - 1) / stride[1] + 1)
    else:
        h = int((h - 1) * stride[0] - 2 * padding[0] + dilation[0] * (n - 1) + output_padding[0] + 1)
        w = int((w - 1) * stride[1] - 2 * padding[1] + dilation[1] * (m - 1) + output_padding[1] + 1)
    return h, w


class MixedNorm(torch.nn.Module):
    def __init__(self, n_channels, eps=1e-5):
        super().__init__()
        self.n_channels = n_channels
        self.eps = eps
        self.alpha = torch.nn.Parameter(torch.empty(n_channels))
        self.batch_norm = torch.nn.BatchNorm2d(n_channels, eps=eps)
        self.instance_norm = torch.nn.InstanceNorm2d(n_channels, eps=eps)
        self.reset_parameter()

    def forward(self, x):
        alpha = self.alpha.view(1, -1, 1, 1)
        return alpha * self.batch_norm(x) + (1 - alpha) * self.instance_norm(x)

    def reset_parameter(self):
        torch.nn.init.uniform_(self.alpha, 0, 1)

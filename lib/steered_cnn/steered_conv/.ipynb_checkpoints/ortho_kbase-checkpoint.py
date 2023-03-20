import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from typing import Union

from ..kbase_conv import KernelBase
from .steerable_filters import radial_steerable_filter


class OrthoKernelBase(KernelBase):
    def __init__(self, base: Union[np.ndarray, torch.Tensor], autonormalize=True):
        """

        Args:
            base: The vertical kernels of the orthogonal base (horizontal kernels are automatically derived)

        Shape:
            base: (K,n,m)
        """
        if isinstance(base, np.ndarray):
            base = torch.from_numpy(base).to(dtype=torch.float)
        base = torch.cat((base, torch.rot90(base, 1, (-2, -1))), dim=0)

        super(OrthoKernelBase, self).__init__(base, autonormalize=autonormalize)
        self.K = base.shape[0] // 2

    @property
    def idx_vertical(self):
        return slice(None, self.K)

    @property
    def idx_horizontal(self):
        return slice(self.K, None)

    @property
    def base_vertical(self):
        return self.base[:self.K]

    @property
    def base_horizontal(self):
        return self.base[self.K:]

    def create_weights(self, n_in, n_out, nonlinearity='linear', nonlinearity_param=None, dist='normal'):
        from torch.nn.init import calculate_gain
        import math

        w = torch.empty((n_out, n_in, 2*self.K))

        gain = calculate_gain(nonlinearity, nonlinearity_param)
        std = gain*math.sqrt(1/(n_in*self.K))
        if dist == 'normal':
            nn.init.normal_(w, std=std)
        elif dist =='uniform':
            bound = std*math.sqrt(3)
            nn.init.uniform_(w, -bound, bound)
        else:
            raise NotImplementedError(f'Unsupported distribution for the random initialization of weights: "{dist}". \n'
                                      f'(Supported distribution are "normal" or "uniform"')
        return w

    @staticmethod
    def from_steerable(R: int, std=.5, size=None, autonormalize=True):
        """


        Args:
            kr: A specification of which steerable filters should be included in the base.
                This parameter can be one of:
                    - a dictionary mapping harmonics order to a list of radius:
                      {k0: [r0, r1, ...], k1: [r2, r3, ...], ...}
                    - an integer interpreted as the wanted kernel size:
                      for every r <= kr/2, k will be set to be the maximum number of harmonics
                      before the apparition of aliasing artefact
            std: The standard deviation of the gaussian distribution which weights the kernels radially.
            size:
            autonormalize:

        Returns: A SteerableKernelBase parametrized by the corresponding kernels.

        """
        if size is None:
            size = int(np.ceil(2 * (R + std)))
            size += int(1 - (size % 2))
        kernels = [np.real(radial_steerable_filter(size=size, k=1, r=r, std=std))
                   for r in range(1, R+1)]
        return OrthoKernelBase(np.stack(kernels), autonormalize=autonormalize)

    def ortho_conv2d(self, input: torch.Tensor, weight: torch.Tensor,
               stride=1, padding='same', dilation=1) -> torch.Tensor:
        """
        Compute the convolution of `input` with the vertical and horizontal kernels of this base given the provided
         weigths.

        Args:
            input: Input tensor.
            weight: Weight for each couples of in and out features and for each kernels of this base.
            stride: The stride of the convolving kernel. Can be a single number or a tuple (sH, sW).
                    Default: 1
            padding:  Implicit paddings on both sides of the input. Can be a single number, a tuple (padH, padW) or
                      one of 'true', 'same' and 'full'.
                      Default: 'same'
            dilation: The spacing between kernel elements. Can be a single number or a tuple (dH, dW).
                      Default: 1

        Shape:
            input: (b, n_in, h, w)
            weight: (n_out, n_in, 2K)
            return: (2, b, n_out, ~h, ~w)

        """
        conv_opts, _ = self._prepare_conv(input, weight, stride=stride, padding=padding, dilation=dilation)

        W = KernelBase.composite_kernels(weight[..., self.idx_vertical], self.base_vertical)
        out_ver = F.conv2d(input, W, **conv_opts)

        W = KernelBase.composite_kernels(weight[..., self.idx_horizontal], self.base_horizontal)
        out_hor = F.conv2d(input, W, **conv_opts)

        return torch.stack((out_ver, out_hor))

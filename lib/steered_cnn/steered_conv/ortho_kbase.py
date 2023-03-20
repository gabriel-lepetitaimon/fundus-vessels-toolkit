import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from typing import Union

from ..kbase_conv import KernelBase
from .steerable_filters import radial_steerable_filter


class OrthoKernelBase(KernelBase):
    def __init__(self, base: Union[np.ndarray, torch.Tensor]):
        """

        Args:
            base: The vertical kernels of the orthogonal base (horizontal kernels are automatically derived)

        Shape:
            base: (K,n,m)
        """
        if isinstance(base, np.ndarray):
            base = torch.from_numpy(base).to(dtype=torch.float)
        base = torch.cat((base, torch.rot90(base, 1, (-2, -1))), dim=0)

        super(OrthoKernelBase, self).__init__(base)
        self.K = base.shape[0] // 2
        self.kernels_label = None
        self.kernels_info = None

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

    def init_weights(self, n_in, n_out, nonlinearity='linear', nonlinearity_param=None, dist='normal'):
        """
        Create and randomly initialize a weight tensor accordingly to this kernel base.

        Args:
            n_in (int): Number of channels in the input image
            n_out (int): Number of channels produced by the convolution
            nonlinearity: Type of nonlinearity used after the convolution.
                          See torch.nn.init.calculate_gain() documentation for more detail.
            nonlinearity_param: Optional parameter for the non-linear function.
                                See torch.nn.init.calculate_gain() documentation for more detail.
            dist: Distribution used for the random initialization of the weights. Can be one of: 'normal' or 'uniform'.
                    Default: 'normal'

        Returns:
            A weight tensor of shape (n_out, n_in, self.K).
        """
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
    def parse(spec, default=None):
        if isinstance(spec, OrthoKernelBase):
            return spec
        if spec is None or spec is True:
            return default
        elif spec is False:
            return False
        if isinstance(spec, int):
            return OrthoKernelBase.create_radial(spec)
        if isinstance(spec, dict):
            if 'R' in spec:
                return OrthoKernelBase.create_radial(**spec)
        raise ValueError(f'Invalid OrthoKernelBase specs: {spec}.')

    @staticmethod
    def create_radial(kernel_size: int, std=.5, size=-1, phase=None, oversample=100):
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
            oversample:

        Returns: A SteerableKernelBase parametrized by the corresponding kernels.

        """
        from ..utils.rotequivariance_toolbox import polar_space

        if isinstance(kernel_size, int):
            # --- Automatically generate kr to cover a kernel of size kr ---
            if not kernel_size % 2 and phase is None:
                phase = np.pi/4  # Shift phase by 45° when kernel size is even.

            if size == -1:
                size = int(np.round(kernel_size*np.sqrt(2)))
                if (size % 2) ^ (kernel_size % 2):
                    size += 1

            r, _ = polar_space(kernel_size)
            r = r.flatten()
            R = []
            for i in np.arange(0.5, kernel_size/np.sqrt(2)+1):
                r_in_interval = (i-1 <= r) & (r < i)
                if r_in_interval.sum():
                    R += [r[r_in_interval].mean()]
        else:
            R = kernel_size

        if phase is None:
            phase = 0
        if size == -1:
            r_max = max(R)
            size = int(np.ceil(2*(r_max+std)))

        kernels = [np.real(radial_steerable_filter(size=size, k=1, r=r, phase=phase,
                                                   std=std, oversampling=oversample, normalize=True))
                   for r in R]
        base = OrthoKernelBase(np.stack(kernels))
        base.kernels_label = [f'r{r}R' for r in R] + [f'r{r}I' for r in R]
        base.kernels_info = [{'r': r, 'type': 'R'} for r in R] + [{'r': r, 'type': 'I'} for r in R]
        return base

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

        W = KernelBase.composite_kernels(weight, self.base)
        out_ver = F.conv2d(input, W, **conv_opts)

        W = KernelBase.composite_kernels(weight[..., self.idx_vertical], self.base_horizontal)
        W -= KernelBase.composite_kernels(weight[..., self.idx_horizontal], self.base_vertical)
        out_hor = F.conv2d(input, W, **conv_opts)

        return torch.stack((out_ver, out_hor))

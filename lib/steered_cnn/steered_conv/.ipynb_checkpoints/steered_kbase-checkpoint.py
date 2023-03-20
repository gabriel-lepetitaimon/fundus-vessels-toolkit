import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ..kbase_conv import KernelBase
from .steerable_filters import max_steerable_harmonics, radial_steerable_filter, cos_sin_ka
from ..utils import clip_pad_center, normalize_vector, torch_norm2d

from collections import OrderedDict
from typing import Union, Dict, List


class SteerableKernelBase(KernelBase):
    def __init__(self, base: 'list(np.array) [K,n,m]', n_kernel_by_k: 'dict {k -> n_k}'):
        """

        Args:
            base: kernels are assumed to be sorted by ascending k
            n_kernel_by_k:
        """
        super(SteerableKernelBase, self).__init__(base, autonormalize=False)

        # Sorting n_filter_by_k and removing invalid values
        self.n_filter_by_k = OrderedDict()
        for k in sorted(n_kernel_by_k.keys()):
            if n_kernel_by_k[k] > 0:
                self.n_filter_by_k[k] = n_kernel_by_k[k]
        n_kernel_by_k = self.n_filter_by_k

        # Statically store the maximum harmonic order (max_k) and all harmonics orders values (k_values)
        self.k_max = max(n_kernel_by_k.keys())
        self.k_values = list(sorted(n_kernel_by_k.keys()))
        self.r_values = [n_kernel_by_k[k] for k in self.k_values]
        self.k_len = len(self.k_values)

        # Store the number of kernel for k=0
        self._n_k0 = n_kernel_by_k.get(0, 0)

        # Store list and cumulative list of kernel count for each k
        self._n_kernel_by_k = []
        self._start_idx_by_k = [0]
        c = 0
        for k in range(0, self.k_max + 1):
            n_kernel = n_kernel_by_k.get(k, 0)
            self._n_kernel_by_k.append(n_kernel)
            c += n_kernel
            self._start_idx_by_k.append(c)

        self.K_equi = self._n_k0
        self.K_steer = self._start_idx_by_k[-1] - self.K_equi
        self.K = self.K_equi + 2*self.K_steer

        assert self.K == base.shape[0], 'The sum of n_kernel_by_k must be equal ' \
                                        'to the number of kernel in base (base.shape[0]).\n ' \
                                        f'(base.shape: {base.shape}, n_kernel_by_k sum: {self._start_idx_by_k[-1]})'

        self.kernels_info = []
        self.kernels_label = []

    def idx(self, k, real=True):
        return self.idx_real(k) if real else self.idx_imag(k)

    def idx_equi(self):
        return slice(None, self.K_equi)

    def idx_real(self, k=None):
        if k is None:
            return slice(self.K_equi, self.K_equi+self.K_steer)
        if k > self.k_max:
            return slice(self.K, self.K)    # Empty slice at the end of the list of kernels
        return slice(self._start_idx_by_k[k], self._start_idx_by_k[k+1])

    def idx_imag(self, k=None):
        if k is None:
            return slice(self.K_equi+self.K_steer, None)
        if k > self.k_max or k <= 0:
            return slice(self.K, self.K)    # Empty slice at the end of the list of kernels
        return slice(self.K_steer+self._start_idx_by_k[k], self.K_steer+self._start_idx_by_k[k+1])

    @property
    def base_equi(self):
        return self.base[self.idx_equi()]

    @property
    def base_real(self):
        return self.base[self.idx_real()]

    @property
    def base_imag(self):
        return self.base[self.idx_imag()]

    @property
    def base_complex(self):
        return torch.cat([self.base_equi,
                          self.base_real+1j*self.base_imag], dim=0)

    def complex_weights(self, weights):
        return torch.cat([weights[self.idx_equi()],
                          weights[self.idx_real()]+1j*weights[self.idx_imag()]], dim=0)

    def expand_r(self, arr, dim=0):
        l = []
        arr = torch.split(arr, 1, dim=dim)
        for ki, r in enumerate(self.r_values):
            if self.k_values[ki] == 0:
                continue
            l += [arr[ki]]*r
        return torch.cat(l, dim=dim)

    def init_weights(self, n_in, n_out, nonlinearity='relu', nonlinearity_param=None, dist='normal', std_theta=0):
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
            std_theta: By default, all radial components of a given weight ω_nmk share the same phase
                       (uniformly sampled between 0 and 2π). This parameter allow the addition of gaussian noise on
                       those phases of standard deviation std_theta. (Default: 0)

        Returns:
            A weight tensor of shape (n_out, n_in, self.K).
        """
        from torch.nn.init import calculate_gain
        import math

        w_equi = torch.empty((n_out, n_in, self.K_equi))

        w_steer_theta = self.expand_r(torch.rand((n_out, n_in, self.k_len))*(2*math.pi), dim=2)
        if std_theta:
            w_steer_theta += torch.normal(0, std=std_theta, size=(n_out, n_in, self.K_steer))

        gain = calculate_gain(nonlinearity, nonlinearity_param)
        std = gain*math.sqrt(1/(n_in*(self.K_equi+self.K_steer)))

        if dist == 'normal':
            nn.init.normal_(w_equi, std=std)
            w_steer_rho = torch.abs(torch.randn((n_out, n_in, self.K_steer)))*std*math.sqrt(2)
            w_steer = w_steer_rho * torch.exp(1j*w_steer_theta)
        elif dist == 'uniform':
            bound = std * math.sqrt(3)
            nn.init.uniform_(w_equi, 0, bound)
            w_steer_rho = torch.rand((n_out, n_in, self.K_steer))*bound*math.sqrt(2)
            w_steer = w_steer_rho * torch.exp(1j*w_steer_theta)
        else:
            raise NotImplementedError(f'Unsupported distribution for the random initialization of weights: "{dist}". \n'
                                      f'(Supported distribution are "normal" or "uniform"')
        w = torch.cat((w_equi, w_steer.real, w_steer.imag), dim=2).contiguous()
        return w

    @staticmethod
    def parse(info, default=None):
        if isinstance(info, SteerableKernelBase):
            return info
        if info is None:
            return default
        if isinstance(info, int):
            return SteerableKernelBase.create_radial(info)
        if isinstance(info, dict):
            if 'kr' in info:
                return SteerableKernelBase.create_radial(**info)
            else:
                return SteerableKernelBase.create_radial(info)

    @staticmethod
    def create_radial(kr: Union[int, Dict[int, List[int]]], std=.5, size=-1, oversample=16,
                      phase=None, max_k=None, cap_k=True):
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
            oversample:
            max_k:

        Returns: A SteerableKernelBase parametrized by the corresponding kernels.

        """
        from ..utils.rotequivariance_toolbox import polar_space

        if size == -1:
            if isinstance(kr, int):
                size = int(np.round(kr*np.sqrt(2)))
                if (size % 2) ^ (kr % 2):
                    size += 1
            else:
                r_max = max(R if np.isscalar(R) else max(R) for R in kr.values())
                size = int(np.ceil(2*(r_max+std)))

        if isinstance(kr, int):
            # --- Automatically generate kr to cover a kernel of size kr ---
            if not kr % 2 and phase is None:
                phase = np.pi/4  # Shift phase by 45° when kernel size is even.
                
            r, _ = polar_space(kr) if cap_k else polar_space(size)
            r = r.flatten()
            rk = {}
            for i in np.arange(1, kr/np.sqrt(2)+1):
                r_in_interval = (i-1 <= r) & (r < i)
                if r_in_interval.sum():
                    k = (r_in_interval.sum())//2
                    rk[r[r_in_interval].mean()] = int(k)

            kr = {}
            for r, K in rk.items():
                for k in range(K+1):
                    if max_k is not None and k > max_k:
                        break
                    if k in kr:
                        kr[k].append(r)
                    else:
                        kr[k] = [r]

        if phase is None:
            phase = 0
        
        kernels_real, kernels_imag = [], []
        labels_real, labels_imag = [], []
        info_real, info_imag = [], []
        n_kernel_by_k = {}

        for k in sorted(kr.keys()):
            R = kr[k]
            if np.isscalar(R):
                R = (R,)
            for r in sorted(R):
                if k in n_kernel_by_k:
                    n_kernel_by_k[k] += 1
                else:
                    n_kernel_by_k[k] = 1

                psi = radial_steerable_filter(size, k, r, std=std, oversampling=oversample, phase=phase, normalize=True)

                labels_real += [f'k{k}r{r:.4g}'+('R' if k > 0 else '')]
                info_real += [{'k': k, 'r': r, 'type': 'R'}]
                kernels_real += [psi.real]
                if k > 0:
                    labels_imag += [f'k{k}r{r:.4g}I']
                    info_imag += [{'k': k, 'r': r, 'type': 'I'}]
                    kernels_imag += [psi.imag]

        K = np.stack(kernels_real + kernels_imag)

        B = SteerableKernelBase(K, n_kernel_by_k=n_kernel_by_k)
        B.kernels_label = labels_real + labels_imag
        B.kernels_info = info_real + info_imag
        return B

    def conv2d(self, input: torch.Tensor, weight: torch.Tensor,
               alpha: Union[int, float, torch.Tensor] = None,
               rho: Union[int, float, torch.Tensor] = 1, rho_nonlinearity: str = None,
               stride=1, padding='same', dilation=1, ) -> torch.Tensor:
        """
        Compute the convolution of `input` and this base's kernels steered by the angle `alpha`
        and premultiplied by `weight`.

        Args:
            input: Input tensor.
            weight: Weight for each couples of in and out features and for each kernels of this base.
            alpha: The angle by which the kernels are steered. (If None then alpha=0.)
                    To enhance the efficiency of the steering computation, α should not be provided as an angle but as a
                      vertical and horizontal projection: cos(α) and sin(α).
                    Please note that for computational efficiency: if rho is not set to None alpha is
                      assumed to be unitary!! In this case, providing a not unitary vector field for alpha will produce
                      unknown behaviour!! (rho default value is 1)
                    This parameter can either be:
                        - A 4D tensor where  alpha=α
                        - A 5D tensor where  alpha[0]=cos(α) and alpha[1]=sin(α)
                        - A 6D tensor where  alpha[0,k-1]=cos(kα) and alpha[1,k-1]=sin(kα) for k=1:k_max
                        - None: the steering is disabled and this function acts like traditional 2D convolution.
                    The last 4 dimensions allow to specify a different alpha for each output pixels [n, n_out, h, w].
                        (If any of these dimensions has a length of 1, alpha will be broadcast along.)
                    Default: None
            rho:    The norm of the attention vector multiplying the outputs features.
                    This parameter can be:
                        - A number: especially if rho=1 computations are simplified to ignore the norm.
                        - A 3D tensor: (b, h, w)
                        - A 4D tensor: (b, n_out, h, w)
                        - None: The value of rho is derived from the norm of alpha.
                        If the vector field alpha is not unitary rho must be explicitly set to None.
                    (If any of these dimensions has a length of 1, rho will be broadcast along.)
                    Default: 1
            rho_nonlinearity: Apply a non-linearity on rho (or the norm of alpha if rho is None).
                              Can be one of:
                                - None: rho is left unchanged (identity function);
                                - 'tanh': hyperbolic tangent non linear function;
                                - 'normalize': rho is set to 1 everywhere, only the angle information is kept.
                              Default: None
            stride: The stride of the convolving kernel. Can be a single number or a tuple (sH, sW).
                    Default: 1
            padding:  Implicit paddings on both sides of the input. Can be a single number, a tuple (padH, padW) or
                      one of 'true', 'same' and 'full'.
                      Default: 'same'
            dilation: The spacing between kernel elements. Can be a single number or a tuple (dH, dW).
                      Default: 1

        Shape:
            input: (b, n_in, h, w)
            weight: (n_out, n_in, K)
            alpha: ([2, [k_max]], b, n_out, ~h, ~w)     (The last four dimensions are broadcastable
                                                         by replacing b, n_out, h or w by 1)
            rho: (b, [n_out], ~h, ~w)          (b, h and w are broadcastable)

            return: (b, n_out, ~h, ~w)

        """
        conv_opts = dict(input=input, weight=weight, alpha=alpha, rho=rho, rho_nonlinearity=rho_nonlinearity,
                         stride=stride, padding=padding, dilation=dilation)
        return self.composite_kernels_conv2d(**conv_opts)
        # return self.preconvolved_base_conv2d(**conv_opts)
        
    def conv_transpose2d(self, input: torch.Tensor, weight: torch.Tensor,
                         alpha: Union[int, float, torch.Tensor] = None,
                         rho: Union[int, float, torch.Tensor] = 1, rho_nonlinearity: str = None,
                         stride=None, padding='same', output_padding=0, dilation=1) -> torch.Tensor:
        """
        Compute the transposed convolution of `input` and kernels described by this base premultiplied by `weight` and
        steered by the angle `alpha`. The resulting features are multiplied by the attention map `rho`.

        Args:
            input: Input tensor.
            weight: Weight for each couples of in and out features and for each kernels of this base.
            alpha: The angle by which the kernels are steered. (If None then alpha=0.)
                    To enhance the efficiency of the steering computation, α should not be provided as an angle but as a
                        vertical and horizontal projection: cos(α) and sin(α).
                    Please note that for computational efficiency: if rho is not set to None alpha is
                        assumed to be unitary! In this case, providing a not unitary vector field for alpha will produce
                        unknown behaviour!! (rho default value is 1)
                    This parameter can either be:
                        - A 4D tensor where  alpha=α
                        - A 5D tensor where  alpha[0]=cos(α) and alpha[1]=sin(α)
                        - A 6D tensor where  alpha[0,k-1]=cos(kα) and alpha[1,k-1]=sin(kα) for k=1:k_max
                        - None: the steering is disabled and this function acts like traditional 2D transposed conv.
                    The last 4 dimensions allow to specify a different alpha for each output pixels [n, n_out, h, w].
                        (If any of these dimensions has a length of 1, alpha will be broadcast along.)
                    Default: None
            rho:    The norm of the attention vector multiplying the outputs features.
                    This parameter can be:
                        - A number: especially if rho=1 computations are simplified to ignore the norm.
                        - A 3D tensor: (b, h, w)
                        - A 4D tensor: (b, n_out, h, w)
                        - None: The value of rho is derived from the norm of alpha.
                        If the vector field alpha is not unitary rho must be explicitly set to None.
                    (If any of these dimensions has a length of 1, rho will be broadcast along.)
                    Default: 1
            rho_nonlinearity: Apply a non-linearity on rho (or the norm of alpha if rho is None).
                              Can be one of:
                                - None: rho is left unchanged (identity function);
                                - 'tanh': hyperbolic tangent non linear function;
                                - 'normalize': rho is set to 1 everywhere, only the angle information is kept.
                              Default: None
            stride: The stride of the transposed convolution (the upsampling factor). Can be a single number or a tuple (sH, sW).
                    Default: None, the kernel size
            padding:  Implicit paddings on both sides of the input. Can be a single number, a tuple (padH, padW) or
                      one of 'true', 'same' and 'full'.
                      Default: 'same'
            output_padding: Additional size added to one side of each dimension in the output shape.
                                Can be a single number or a tuple (padH, padW).
                            Default: 0
            dilation: The spacing between kernel elements. Can be a single number or a tuple (dH, dW).
                      Default: 1

        Shape:
            input: (b, n_in, h, w)
            weight: (n_out, n_in, K)
            alpha: ([2, [k_max]], b, n_out, ~h, ~w)     (The last four dimensions are broadcastable
                                                         by replacing b, n_out, h or w by 1)
            rho: (b, [n_out], ~h, ~w)          (b, h and w are broadcastable)

            return: (b, n_out, ~h*stride, ~w*stride)

        """
        conv_opts = dict(input=input, weight=weight, alpha=alpha, rho=rho, rho_nonlinearity=rho_nonlinearity,
                         stride=stride, padding=padding, dilation=dilation,
                         transpose=True, output_padding=output_padding)
        return self.composite_kernels_conv2d(**conv_opts)
        # return self.preconvolved_base_conv2d(**conv_opts)

    def _prepare_steered_conv(self, input, weight, alpha, rho, rho_nonlinearity,
                              stride, padding, dilation, transpose=False, output_padding=0):
        """
        Prepare module for the convolution operation.
        Returns alpha, conv_opts, (b, n_in, n_out, k, h, w)
        """
        conv_opts, shapes = self._prepare_conv(input=input, weight=weight, transpose=transpose, stride=stride,
                                               padding=padding, output_padding=output_padding, dilation=dilation)
        b, n_in, n_out, k, h, w = shapes
        if isinstance(alpha, (int, float)):
            if alpha == 0:
                alpha = None
            else:
                alpha = torch.Tensor([alpha]).to(device=input.device)
                alpha = torch.stack((torch.cos(alpha), torch.sin(alpha)))[:, None, None, None]

        # --- ALPHA ---
        if alpha is not None:
            alpha = clip_pad_center(alpha, (h, w), broadcastable=True)

            assert 4 <= alpha.dim() <= 6, f'Invalid number of dimensions for alpha: alpha.shape={alpha.shape}.\n' \
                                          'alpha shape should be like ([2, [k_max]], b, n_out, h, w)'
            if alpha.dim() == 4:
                b_a, n_out_a, h_a, w_a = alpha.shape
                alpha = torch.stack((torch.cos(alpha), torch.sin(alpha)))
            elif alpha.dim() == 5:
                cossin, b_a, n_out_a, h_a, w_a = alpha.shape
                assert cossin == 2, f'Invalid first dimensions for alpha: alpha.shape[0]={cossin} but should be 2.\n' \
                                    f'(if alpha is a 5D matrix its shape should be: (2, b, n_out, h, w),' \
                                    f' with alpha[0]=cos(α) and alpha[1]=sin(α)\n' \
                                    f'alpha.shape={alpha.shape})'
            else:
                cossin, k_max_a, b_a, n_out_a, h_a, w_a = alpha.shape
                assert cossin == 2, f'Invalid first dimensions for alpha: alpha.shape[0]={cossin} but should be 2.\n' \
                                    f'(if alpha is a 6D matrix its shape should be: (2, k_max, b, n_out, h, w),' \
                                    f' with alpha[0]=cos(α) and alpha[1]=sin(α)\n' \
                                    f'alpha.shape={alpha.shape})'
                assert k_max_a >= self.k_max, f'Invalid k dimension for alpha: alpha.shape[1]={k_max_a} but should be equal to k_max={self.k_max}.\n' \
                                            f'(alpha.shape={alpha.shape})'
                if k_max_a < self.k_max:
                    alpha = alpha[:, :self.k_max]
            assert b_a == 1 or b == b_a, f'Invalid batch size for alpha: alpha.shape[{alpha.dim()-4}]={b_a} but should be {b} (or  1 for broadcast)\n' \
                                         f'(alpha.shape={alpha.shape}, input.shape={input.shape}'
            assert n_out_a == 1 or n_out == n_out_a, f'Invalid number of output features for alpha: ' \
                                                     f'alpha.shape[{alpha.dim() - 3}]={n_out_a} but should be {n_out} (or  1 for broadcast)\n' \
                                                     f'(alpha.shape={alpha.shape}, input.shape={input.shape}'
            assert h_a == 1 or h_a == h, f'Invalid height for alpha: alpha.shape[{alpha.dim()-2}]={h_a} but should be {h} (or  1 for broadcast)\n' \
                                         f'(alpha.shape={alpha.shape}, input.shape={input.shape}'
            assert w_a == 1 or w_a == w, f'Invalid width for alpha: alpha.shape[{alpha.dim() - 1}]={w_a} but should be {w} (or  1 for broadcast)\n' \
                                         f'(alpha.shape={alpha.shape}, input.shape={input.shape}'

            if rho is None:
                if alpha.dim() == 6:
                    rho = torch_norm2d(alpha[:, 0])
                    alpha /= rho[None, None, :, :, :, :]+1e-8
                else:
                    alpha, rho = normalize_vector(alpha)

        # --- RHO ---
        if isinstance(rho, (int, float)):
            if rho == 1:
                rho = None
            else:
                rho = torch.Tensor([rho]).to(device=input.device)[:, None, None, None]
        elif rho is not None:
            rho = clip_pad_center(rho, (h, w), broadcastable=True)

            assert 3 <= rho.dim() <= 4, f'Invalid number of dimensions for rho: rho.shape={rho.shape}.\n' \
                                          'rho shape should be like (b, [n_out], h, w)'
            if rho.dim() == 3:
                b_r, h_r, w_r = rho.shape
                n_out_r = 1
                rho = rho[:, None, :, :]
            else:
                b_r, n_out_r, h_r, w_r = rho.shape
                assert b_r == 1 or b == b_r, f'Invalid batch size for rho: rho.shape[0]={b_r} but should be {b} (or  1 for broadcast)\n' \
                                             f'(rho.shape={rho.shape}, input.shape={input.shape}'
            assert n_out_r == 1 or n_out == n_out_r, f'Invalid number of output features for rho: ' \
                                                     f'rho.shape[{rho.dim() - 3}]={n_out_r} but should be {n_out} (or  1 for broadcast)\n' \
                                                     f'(rho.shape={rho.shape}, input.shape={input.shape}'
            assert h_r == 1 or h_r == h, f'Invalid height for rho: rho.shape[{rho.dim()-2}]={h_r} but should be {h} (or  1 for broadcast)\n' \
                                         f'(rho.shape={rho.shape}, input.shape={input.shape}'
            assert w_r == 1 or w_r == w, f'Invalid width for rho: rho.shape[{rho.dim() - 1}]={w_r} but should be {w} (or  1 for broadcast)\n' \
                                         f'(rho.shape={rho.shape}, input.shape={input.shape}'
        if rho is not None:
            if rho_nonlinearity == 'normalize':
                rho = 1
            elif rho_nonlinearity == 'tanh':
                rho = torch.tanh(rho)

        return alpha, rho, conv_opts, shapes

    # --- Composite Kernels ---
    def composite_equi_kernels(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Compute the sum of all kernels for a the polar harmonic k=0,
        based on the provided weight and self.base (shape: [K, n, m], Ψ=[Ψ_0r, ΨR_1r, ΨR_2r, ..., ΨI_1r. ΨI_2r, ...]).

        Args:
            weight: The weight of each kernels in self.base
                    [ω_ji0r, ωR_ji1r, ωR_ji2r, ..., ωI_ji1r, ωI_ji2r, ...]

        Shapes:
            weight: (n_out, n_in, K)
            return: (n_out, n_in, n, m)

        Returns: The composite kernel.
                 φ_ji0 = Σr[ ωR_ji0r ΨR_0r]
        """
        idx = self.idx_equi()
        return KernelBase.composite_kernels(weight[..., idx], self.base[idx])

    def composite_steerable_kernels_real(self, weight: torch.Tensor, k) -> torch.Tensor:
        """
        Compute φR_jik: the real part of the sum of all kernels for a specific polar harmonic k,
        based on the provided weight and self.base (shape: [K, n, m], Ψ=[Ψ_0r, ΨR_1r, ΨR_2r, ..., ΨI_1r. ΨI_2r, ...]).

        Args:
            weight: The weight of each kernels in self.base
                    [ω_ji0r, ωR_ji1r, ωR_ji2r, ..., ωI_ji1r, ωI_ji2r, ...]
            k: The desired polar harmonic. (0 <= k <= self.max_k)

        Shapes:
            weight: (n_out, n_in, K)
            return: (n_out, n_in, n, m)

        Returns: The composite kernel.
                 φR_jik = Σr[ ωR_jikr ΨR_kr + ωI_jikr ΨI_kr]
        """
        if k == 0:
            return self.composite_equi_kernels(weight)

        real_idx = self.idx_real(k)
        imag_idx = self.idx_imag(k)
        w_real, w_imag = weight[..., real_idx], weight[..., imag_idx]
        psi_real, psi_imag = self.base[real_idx], self.base[imag_idx]

        return KernelBase.composite_kernels(w_real, psi_real) + KernelBase.composite_kernels(w_imag, psi_imag)

    def composite_steerable_kernels_imag(self, weight: torch.Tensor, k) -> torch.Tensor:
        """
        Compute φR_jik: the imaginary part of the sum of all kernels for a specific polar harmonic k,
        based on the provided weight and self.base (shape: [K, n, m], Ψ=[Ψ_0r, ΨR_1r, ΨR_2r, ..., ΨI_1r. ΨI_2r, ...]).

        Args:
            weight: The weight of each kernels in self.base
                    [ω_ji0r, ωR_ji1r, ωR_ji2r, ..., ωI_ji1r, ωI_ji2r, ...]
            k: The desired polar harmonic. (0 <= k <= self.max_k)

        Shapes:
            weight: (n_out, n_in, K)
            return: (n_out, n_in, n, m)

        Returns: The composite kernel.
                 φI_jik = Σr[ ωR_jikr ΨI_kr - ωI_jikr ΨR_kr]
        """
        if k == 0:
            n_out, n_in, K = weight.shape
            K, n, m = self.base.shape
            return torch.zeros((n_out, n_in, n, m), device=self.base.device, dtype=self.base.dtype)
        real_idx = self.idx_real(k)
        imag_idx = self.idx_imag(k)
        w_real, w_imag = weight[..., real_idx], weight[..., imag_idx]
        psi_real, psi_imag = self.base[real_idx], self.base[imag_idx]

        return KernelBase.composite_kernels(w_real, psi_imag) - KernelBase.composite_kernels(w_imag, psi_real)

    def composite_kernels_conv2d(self, input: torch.Tensor, weight: torch.Tensor,
                                 alpha: Union[int, float, torch.Tensor] = None,
                                 rho: Union[int, float, torch.Tensor] = None, rho_nonlinearity: str = None,
                                 stride=1, padding='same', output_padding=0, dilation=1, transpose=False):
        alpha, rho, conv_opts, (b, n_in, n_out, K, h, w) = self._prepare_steered_conv(input, weight, alpha,
                                                                                      rho, rho_nonlinearity,
                                                                                      stride, padding, dilation,
                                                                                      transpose, output_padding)
        if alpha is None:
            return super(SteerableKernelBase, self).composite_kernels_conv2d(input, weight, transpose=transpose,
                                                                             **conv_opts)
        
        conv2d = F.conv2d if not transpose else F.conv_transpose2d

        # f = X⊛K_equi + Σk[ cos(kα)(X⊛K_kreal) + sin(kα) (X⊛K_kimag)]
        # computing f = X ⊛ K_equi ...
        if self._n_k0:
            f = conv2d(input, self.composite_equi_kernels(weight), **conv_opts)
        else:
            f = torch.zeros((b, n_out, h, w),
                            device=self.base.device, dtype=self.base.dtype)

        # then: f += Σk[ cos(kα)(X⊛K_kreal) + sin(kα) (X⊛K_kimag)]
        for k in self.k_values:
            if k == 0:
                continue
            if alpha.dim() == 5:
                if k == 1:
                    cos_sin_kalpha = alpha
                else:
                    cos_sin_kalpha = cos_sin_ka(alpha, cos_sin_kalpha)
            else:
                cos_sin_kalpha = alpha[:, k-1]
            f.addcmul_(cos_sin_kalpha[0], conv2d(input, self.composite_steerable_kernels_real(weight, k=k), **conv_opts))
            f.addcmul_(cos_sin_kalpha[1], conv2d(input, self.composite_steerable_kernels_imag(weight, k=k), **conv_opts))

        if rho is not None:
            f *= rho
        return f

    # --- Preconvolve Kernels ---
    @staticmethod
    def _preconvolved_KW(xbase, weight, idx_x, idx_w=None):
        if idx_w is None:
            idx_w = idx_x
        K = torch.flatten(xbase[..., idx_x], start_dim=-2)
        W = torch.flatten(weight[..., idx_w], start_dim=-2).transpose(0, 1)
        return K, W

    def preconvolved_base_conv2d(self, input: torch.Tensor, weight: torch.Tensor,
                                 alpha: Union[int, float, torch.Tensor] = None,
                                 rho: Union[int, float, torch.Tensor] = None, rho_nonlinearity: str = None,
                                 stride=1, padding='same', output_padding=0, dilation=1, transpose=False):
        alpha, rho, conv_opts, (b, n_in, n_out, K, h, w) = self._prepare_steered_conv(input, weight, alpha,
                                                                                      rho, rho_nonlinearity,
                                                                                      stride, padding, dilation,
                                                                                      transpose, output_padding)
        if alpha is None:
            return super(SteerableKernelBase, self).preconvolved_base_conv2d(input, weight, transpose=transpose, **conv_opts)

        # alpha shape: (2, [k_max], b, n_out, ~h, ~w)
        if alpha.dim() == 5:
            alpha = alpha.permute(0, 1, 3, 4, 2)    # (2, b, ~h, ~w, n_out)
        else:   #alpha.dim() == 6
            alpha = alpha.permute(0, 1, 2, 4, 5, 3)       # (2, k_max, b, ~h, ~w, n_out)

        xbase = KernelBase.preconvolve_base(input, self.base, transpose=transpose, **conv_opts)
        # f = X⊛K_equi + Σk[ cos(kα)(X⊛K_kreal) + sin(kα) (X⊛K_kimag)]
        # computing f = X ⊛ K_equi ...
        if self._n_k0:
            K, W = self._preconvolved_KW(xbase, weight, self.idx_equi())
            f = torch.matmul(K, W)  # K:[b,h,w,n_in*k] x W:[n_in*k, n_out] -> [b,h,w,n_out]
        else:
            f = torch.zeros((b, h, w, n_out),
                            device=self.base.device, dtype=self.base.dtype)

        # then: f += Σk[ cos(kα)(X⊛K_kreal) + sin(kα) (X⊛K_kimag)]
        for k in self.k_values:
            if k == 0:
                continue
            if alpha.dim() == 5:
                if k == 1:
                    cos_sin_kalpha = alpha
                    alpha_norm = torch.linalg.norm(alpha, dim=0)+1e-8
                else:
                    cos_sin_kalpha = cos_sin_ka(alpha, cos_sin_kalpha) / alpha_norm
            else:
                cos_sin_kalpha = alpha[:, k-1]
            KR, WR = self._preconvolved_KW(xbase, weight, self.idx_real(k=k))
            KI, WI = self._preconvolved_KW(xbase, weight, self.idx_imag(k=k))
            f.addcmul_(cos_sin_kalpha[0], (torch.matmul(KR, WR) + torch.matmul(KI, WI)))
            f.addcmul_(cos_sin_kalpha[1], torch.matmul(KI, WR) - torch.matmul(KR, WI))

        f = f.permute(0, 3, 1, 2)    # [b,n_out,h,w]
        if rho is not None:
            f *= rho
        return f

    # --- Analyse Weights ---
    def format_weights(self, weights, mean=True):
        from pandas import DataFrame
        import numpy as np
        from ..utils import iter_index
        data = dict(r=[_['r'] for _ in self.kernels_info],
                    k=[_['k'] for _ in self.kernels_info],
                    type=[_['type'] for _ in self.kernels_info])
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()
        s = weights.shape[:-1]
        if not len(s) or np.prod(s) == 1:
            data['weight'] = weights.flatten()
        else:
            for idx in iter_index(weights.shape):
                data[f'weights{list(idx)}'] = weights[idx]
            if mean:
                data['weights_mean'] = weights.mean(axis=tuple(range(len(s))))
                data['weights_std'] = weights.std(axis=tuple(range(len(s))))
        return DataFrame(data=data)

    def complex_kernels_couple(self):
        infos = list(self.kernels_info)
        couples = []
        couples_info = []
        while sum(_ is not None for _ in infos):
            info1 = infos.pop(-1)
            if info1 is None:
                continue
            i1 = len(infos)
            for i2, info2 in enumerate(infos):
                if i2 is None:
                    continue
                if info1['r'] == info2['r'] and info1['k'] == info2['k']:
                    infos[i2] = None
                    couples += [(i2, i1)]
                    break
            else:
                couples += [(i1, None)]
            couples_info += [{'r_name': info1['r'], 'k':info1['k'], 'r': info1['r'],
                              'name': f'k={info1["k"]} r={info1["r"]}'}]
        return tuple(reversed(couples)), tuple(reversed(couples_info))

    def flatten_weights(self, weights, complex=False):
        import numpy as np
        if complex:
            complex_couples, w_infos = self.complex_kernels_couple()
        else:
            w_infos = [{'r_name': f'r={_["r"]} {("R" if _["type"]=="R" else " I") if _["k"]>0 else ""}',
                        'name': f'k={_["k"]} r={_["r"]} {("R" if _["type"]=="R" else " I") if _["k"]>0 else ""}',
                        'k': _['k'], 'r': _['r']} for _ in self.kernels_info]
            complex_couples = None

        def flatten_weight(w, w_infos):
            if isinstance(w, torch.Tensor):
                w = w.detach().cpu().numpy()
            w = w.reshape(-1, w.shape[-1])
            if complex is False:
                return w, w_infos

            infos = []
            l = []
            for w1, w2 in complex_couples:
                if w2 is None:
                    if complex == 'angle':
                        continue
                    w0 = w[:, w1]
                else:
                    w0 = w[:, w1]+1j*w[:, w2]
                if complex == 'angle':
                    l += [np.angle(w0)]
                else:
                    l += [np.abs(w0)]
                infos += [w_infos[w1]]
            w = np.stack(l, axis=1)
            return w, infos

        if isinstance(weights, (list, tuple)):
            flat_weights = []
            for w in weights:
                flat_w, infos = flatten_weight(w, w_infos)
                flat_weights += [flat_w]
            return np.concatenate(flat_weights, axis=0), infos
        else:
            return flatten_weight(weights, w_infos)

    def weights_dist(self, weights, Q=3, complex=False):
        import numpy as np

        weights, infos = self.flatten_weights(weights, complex=complex)
        data = {k: [_[k] for _ in infos] for k, v in infos[0].items()}
        if isinstance(Q, int):
            Q = [(i+1)/(Q+1) for i in range(Q)]
        q = np.array(Q)*100
        q = q/2
        q = np.concatenate([50-q[::-1], [50], 50+q]).flatten()

        perc = np.percentile(weights, q, axis=0)
        data['median'] = perc[len(Q)]

        for i, q in enumerate(Q):
            data[f'q{i}'] = perc[-i-1]
            data[f'-q{i}'] = perc[i]
        return data

    def plot_weights_dist(self, weights, Q=5, complex='norm', scale_type='linear', wrange=None):
        import pandas as pd
        import altair as alt
        N = len(weights) if isinstance(weights, dict) else 1
        def plot_dist(weights, offset=0, color=alt.Undefined, domain=None):
            df = pd.DataFrame(data=self.weights_dist(weights, Q=Q, complex=complex))
            chart = alt.Chart(data=df, width=70)

            if domain is None:
                domain = df.loc[:, 'w'].min(), df.loc[:, 'w'].max()

            if scale_type == 'log':
                axis = alt.Axis(format='e', values=[10**_
                                                    for _ in range(int(np.floor(np.log10(domain[0]))),
                                                                   int(np.ceil(np.log10(domain[1]))))])
            else:
                axis = alt.Axis()

            plot = chart.mark_tick(width=10, thickness=2, xOffset=offset, color=color
                                   ).encode(
                x='r_name:N',
                y=alt.Y('median:Q', title='Weights '+complex,
                        scale=alt.Scale(type=scale_type, domain=domain),
                        axis=axis)
            )

            for q in range(Q):
                plot += chart.mark_bar(width=10, opacity=.2 if q < Q-2 else .3, xOffset=offset, color=color
                                       ).encode(
                    x='r_name:N',
                    y=alt.Y(f'-q{q}:Q', title='',
                            #axis=alt.Axis(tickCount=7)
                            ),
                    y2=f'q{q}:Q',
                )
            return plot

        if isinstance(weights, dict):
            i = 0
            tableau10 = '#4E79A7 #F28E2B #E15759 #76B6B2 #59A14F #EDC948 #B07AA1 #FF9DA7 #9C755F #BAB0AC'.split(' ')
            plots = []
            if wrange is None:
                wrange = [float(min(_.min() for _ in weights.values())), float(max(_.max() for _ in weights.values()))]
            for k, w in weights.items():
                plots += [plot_dist(weights=w, offset=10*i, color=tableau10[i], domain=wrange)]
                i += 1
            plot = alt.LayerChart(plots)
        else:
            if wrange is None:
                wrange = [float(min(_.min() for _ in weights)), float(max(_.max() for _ in weights))]
            plot = plot_dist(weights=weights, domain=wrange)

        return plot.facet(column='k').resolve_scale(x='independent').interactive(bind_x=False)

    def weights_hist(self, weights, bins=100, wrange=None, binspace='linear', complex=False, norm='sum',
                     symlog_C=None, displayable=True):
        import pandas as pd
        weights, w_infos = self.flatten_weights(weights, complex=complex)
        if isinstance(bins, int):
            if wrange is None:
                wrange = weights.min(), weights.max()
            if binspace == 'linear':
                bins = np.linspace(wrange[0], wrange[1], num=bins)
            elif binspace == 'log':
                bins = np.logspace(np.log10(wrange[0]), np.log10(wrange[1]), num=bins)
            elif binspace == 'symlog':
                if symlog_C is None:
                    symlog_C = np.abs(np.diff(wrange))*2/bins
                wrange = np.sign(wrange)*np.log10(1 + np.abs(wrange)/symlog_C)
                bins = np.linspace(wrange[0], wrange[1], endpoint=True)
                bins = np.sign(bins)*symlog_C*(np.power(10, np.abs(bins))-1)
            else:
                raise NotImplementedError(f'{binspace} scale is not implemented yet. '
                                          f'Valid binspace is "linear", "log" or "symlog".')
        else:
            wrange = np.min(bins), np.max(bins)

        hists = []
        for i in range(weights.shape[1]):
            hist, bins_edge = np.histogram(weights[:, i], bins=bins, range=wrange)
            if norm == 'max':
                hist = hist / hist.max()
            elif norm == 'sum':
                hist = hist / hist.sum()
            hists += [hist]
        bins = (bins_edge[:-1] + bins_edge[1:])/2

        if displayable:
            idx = pd.MultiIndex.from_tuples([(_['k'], _['r_name']) for _ in w_infos], names=['k', 'r'])
            return pd.DataFrame(data=hists, index=idx, columns=bins)
        else:
            dfs = [pd.DataFrame(h, index=pd.Index(bins, name='w'), columns=['density']) for h in hists]
            return pd.concat(dfs, keys=[(_['k'], _['r'], _['name']) for _ in w_infos], names=['k', 'r', 'name'])

    def plot_weights_hist(self, weights, bins=100, binspace=None, wrange=None, complex='norm', norm='max'):
        import altair as alt
        if binspace is None:
            if complex is False or complex=='angle':
                binspace = 'linear'
            else:
                binspace = 'symlog'
        elif binspace == 'log' and complex != 'norm':
            binspace = 'symlog'
        if not isinstance(bins, int):
            binspace = 'linear'
        df = self.weights_hist(weights, bins=bins, binspace=binspace, wrange=wrange, norm=norm,
                               complex=complex, displayable=False)\
                 .reset_index(['r', 'k', 'name', 'w'])
        wrange = df.loc[:, 'w'].min(), df.loc[:, 'w'].max()
        chart = alt.Chart(data=df, width=50)

        plot = chart.mark_area(orient='horizontal').encode(
            y=alt.Y('w:Q', scale=alt.Scale(type=binspace, domain=wrange, constant=(wrange[1]-wrange[0])/20),
                    axis=alt.Axis(title="Weights Distribution")),
            x=alt.X(
                'density:Q',
                stack='center',
                title=None,
                impute=None,
                axis=alt.Axis(labels=False, values=[0], grid=False, ticks=True),
            ),
            column=alt.Column(
                'name:N',
                title="Basis Kernel",
                header=alt.Header(
                    titleOrient='bottom',
                    labelOrient='bottom',
                    labelPadding=0,
                ),
            ))
        return plot.configure_facet(
            spacing=0
        ).configure_view(
            stroke=None
        ).interactive(bind_x=False)

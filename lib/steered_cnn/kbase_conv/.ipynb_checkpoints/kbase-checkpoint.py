import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from ..utils import clip_pad_center, compute_padding


class KernelBase:
    def __init__(self, base: 'torch.Tensor [n_k,h,w]', autonormalize=True, epsilon=1e-5):
        if isinstance(base, np.ndarray):
            base = torch.from_numpy(base).to(dtype=torch.float)
        if autonormalize:
            base = KernelBase.normalize_base(base)
        self.base = base

        self.r = []
        n_k, h, w = base.shape
        c = (min(h, w)+1)/2
        normed_base = torch.abs(base)
        normed_base /= torch.amax(base, dim=(1, 2), keepdim=True)
        for b in normed_base:
            for r in torch.arange(c % 1, c, 1):
                i0, i1 = int(c-r-1), int(c+r)
                if torch.max(b - clip_pad_center(b[i0:i1, i0:i1], b.shape)) < epsilon:
                    self.r += [r]
                    break
            else:
                self.r += [c]

    @staticmethod
    def normalize_base(base: 'torch.Tensor [n_k,h,w]'):
        b = torch.empty_like(base)
        for i in range(base.shape[0]):
            b[i] = base[i]/(torch.sqrt(torch.square(base[i]).sum())+1e-8)
        return b

    @staticmethod
    def cardinal_base(size=3):
        base = []
        for k in range(1, size+1, 2):
            if k == 1:
                base.append(np.ones((1, 1, 1)))
            else:
                kernels = []
                for i in range(k-1):
                    K = np.zeros((k, k))
                    K[0, i] = 1
                    kernels.append(K)
                for i in range(k-1):
                    K = np.zeros((k, k))
                    K[i, -1] = 1
                    kernels.append(K)
                for i in range(k-1):
                    K = np.zeros((k, k))
                    K[-1, -1-i] = 1
                    kernels.append(K)
                for i in range(k-1):
                    K = np.zeros((k, k))
                    K[-1-i, 0] = 1
                    kernels.append(K)
                base.append(np.stack(kernels))
        return KernelBase(base)

    def init_weights(self, n_in, n_out, nonlinearity=' relu', nonlinearity_param=None, dist='normal'):
        from torch.nn.init import calculate_gain
        import math

        K = self.base.shape[0]
        w = torch.empty((n_out, n_in, K))

        gain = calculate_gain(nonlinearity, nonlinearity_param)
        std = gain * math.sqrt(1 / (n_in * K))
        if dist == 'normal':
            nn.init.normal_(w, std=std)
        elif dist == 'uniform':
            bound = std * math.sqrt(3)
            nn.init.uniform_(w, -bound, bound)
        else:
            raise NotImplementedError(f'Unsupported distribution for the random initialization of weights: "{dist}". \n'
                                      f'(Supported distribution are "normal" or "uniform"')

        return w

    def approximate_weights(self, kernels: 'torch.Tensor [n_out, n_in, h, w]',  info=None, bias=False, ridge_alpha=.1):
        from sklearn.linear_model import Ridge
        n_out, n_in, h, w = kernels.shape
        base = self.base.cpu().numpy()
        k, n, m = base.shape

        device = kernels.device
        kernels = kernels.detach().cpu()
        kernels = clip_pad_center(kernels, (n, m))

        kernels = kernels.numpy()
        X = base.reshape((k, n*m)).T
        Y = kernels.reshape((n_out*n_in, n*m)).T

        if bias:
            X = np.concatenate([X, np.ones((n*m, 1))], axis=1)

        regr = Ridge(alpha=ridge_alpha, fit_intercept=False)
        regr.fit(X, Y)
        if info is not None:
            from sklearn.metrics import mean_squared_error, r2_score
            y_approx = regr.predict(X)
            mse = mean_squared_error(Y, y_approx)
            info['mse'] = mse
            info['r2'] = 1 if mse < 1e-12 else r2_score(Y, y_approx)
            info['y_approx'] = torch.from_numpy(y_approx.T.reshape((n_out, n_in, n, m))).to(device=device)

        coef = regr.coef_   # [n_out*n_in, k]
        coef = torch.from_numpy(coef).to(device=device, dtype=torch.float).reshape(n_out, n_in, -1)
        if not bias:
            return coef
        else:
            return coef[..., :-1], coef[..., -1]

    @property
    def device(self):
        return self.base.device

    def to(self, *args, **kwargs):
        self.base = self.base.to(*args, **kwargs)
        return self

    def _prepare_conv(self, input, weight, stride, padding, dilation, transpose=False, output_padding=0):
        from ..utils import compute_conv_outputs_dim, compute_padding
        if self.device != input.device:
            self.to(input.device)
        padding = compute_padding(padding, self.base.shape)
        output_padding = compute_padding(output_padding, self.base.shape)

        b, n_in, h, w = input.shape
        n_out, n_in_w, k_w = weight.shape
        k, n, m = self.base.shape
        if transpose:
            assert n_in == n_out, 'Incoherent number of input neurons between the provided input and transposed weight:\n' \
                                   f'input.shape={input.shape} (n_in={n_in}), weight.shape={weight.shape} (n_in={n_out}).'
        else:
            assert n_in == n_in_w, 'Incoherent number of input neurons between the provided input and weight:\n' \
                                   f'input.shape={input.shape} (n_in={n_in}), weight.shape={weight.shape} (n_in={n_in_w}).'
        assert k == k_w, f"The provided weights have an incorrect number of kernels:\n " + \
                         f"weight.shape={weight.shape} (k={k_w}), but should be {k}."
        conv_opts = dict(padding=padding, dilation=dilation, stride=stride)
        if output_padding not in (0, (0,0)):
            if transpose:
                conv_opts['output_padding'] = output_padding
            else:
                raise ValueError('Parameter output_padding should equal 0 for not transpose conv, '
                                 f'but {output_padding} was provided.')
        h, w = compute_conv_outputs_dim(input.shape, self.base.shape, transpose=transpose, **conv_opts)

        return conv_opts, (b, n_in, n_out, k, h, w)

    # --- Composite Kernels ---
    @staticmethod
    def composite_kernels(weight: 'torch.Tensor [n_out, n_in, n_k]', base: 'torch.Tensor [n_k, n, m]'):
        W = weight
        n_out, n_in, n_k = W.shape
        n_k, n, m = base.shape

        K = base.reshape(n_k, n*m)

        # W: [n_out,n_in,k0:k0+k] * F: [f,n*m] -> [n_out, n_in, n*m]
        return torch.matmul(W, K).reshape(n_out, n_in, n, m)

    def composite_kernels_conv2d(self, input: 'torch.Tensor [b,i,w,h]', weight: 'np.array [n_out, n_in, k]',
                                 stride=1, padding='same',  output_padding=0, dilation=1, groups=1, transpose=False):
        W = KernelBase.composite_kernels(weight, self.base)
        padding = compute_padding(padding, W.shape)
        if not transpose:
            return F.conv2d(input, W, stride=stride, padding=padding, dilation=dilation, groups=groups)
        else:
            output_padding = compute_padding(output_padding, W.shape)
            return F.conv_transpose2d(input, W, stride=stride, padding=padding, output_padding=output_padding,
                                      dilation=dilation, groups=groups)

    # --- Preconvolve Base ---
    @staticmethod
    def preconvolve_base(input: 'torch.Tensor [b,i,w,h]', base: 'torch.Tensor [n_k, n, m]',
                         stride=1, padding='same', output_padding=0, dilation=1, transpose=False):
        b, n_in, h, w = input.shape
        n_k, n, m = base.shape
        pad = compute_padding(padding, (n, m))

        input = input.reshape(b * n_in, 1, h, w)
        base = base.reshape(n_k, 1, n, m)
        if not transpose:
            K = F.conv2d(input, base, stride=stride, padding=pad, dilation=dilation)
        else:
            output_padding = compute_padding(output_padding, (n, m))
            K = F.conv_tranpose2d(input, base, stride=stride, padding=pad, dilation=dilation,
                                  output_padding=output_padding)
        h, w = K.shape[-2:]     # h and w after padding, K.shape: [b*n_in, n_k, ~h, ~w]
        return K.reshape(b, n_in, n_k, h, w).permute(0, 3, 4, 1, 2).reshape(b, h, w, n_in, n_k)

    @staticmethod
    def multiply_preconvolved_base_weigth(preconvolved_base, weight):
        K = torch.flatten(preconvolved_base, start_dim=-2)
        W = torch.flatten(weight, start_dim=1).transpose(0, 1)
        f = torch.matmul(K, W)  # K:[b,h,w,n_in*k] x W:[n_in*k, n_out] -> [b,h,w,n_out]
        return f.permute(0, 3, 1, 2)    # [b,n_out,h,w]

    def preconvolved_base_conv2d(self, input: 'torch.Tensor [b,i,w,h]', weight: 'np.array [n_out, n_in, k]',
                                 stride=1, padding='same', output_padding=0, dilation=1, transpose=False):
        bases = KernelBase.preconvolve_base(input, self.base, stride=stride, padding=padding, dilation=dilation,
                                            transpose=transpose, output_padding=output_padding)
        b, h, w, n_in, k = bases.shape
        n_out, n_in_w, k_w = weight.shape

        assert n_in == n_in_w, f"The provided inputs and weights have different number of input neuron:\n " +\
                               f"x.shape[1]=={n_in}, weigth.shape[1]=={n_in_w}."
        assert k == k_w, f"The provided weights have an incorrect number of kernels:\n " +\
                         f"weight.shape[2]=={k_w}, but should be {k}."

        return KernelBase.multiply_preconvolved_base_weigth(bases, weight)

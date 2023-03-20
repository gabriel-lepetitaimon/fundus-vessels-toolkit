import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from ..utils import clip_pad_center, compute_padding


class KernelBase:
    def __init__(self, base: 'np.array [n_k,h,w]'):
        if isinstance(base, np.ndarray):
            base = torch.from_numpy(base).to(dtype=torch.float)
        self.base = base

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

    def create_weights(self, n_in, n_out):
        R = len(self.base)
        K = sum([self.base[r].shape[0] for r in range(R)])
        w = torch.empty((n_out, n_in, K))
        b = np.sqrt(3/(n_in*K))
        nn.init.uniform_(w, -b, b)
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
                                 stride=1, padding='same', dilation=1, groups=1):
        W = KernelBase.composite_kernels(weight, self.base)
        padding = compute_padding(padding, W.shape)
        return F.conv2d(input, W, stride=stride, padding=padding, dilation=dilation, groups=groups)

    # --- Preconvolve Base ---
    @staticmethod
    def preconvolve_base(input: 'torch.Tensor [b,i,w,h]', base: 'torch.Tensor [n_k, n, m]',
                         stride=1, padding='same', dilation=1):
        b, n_in, h, w = input.shape
        n_k, n, m = base.shape
        pad = compute_padding(padding, (n, m))

        input = input.reshape(b * n_in, 1, h, w)
        base = base.reshape(n_k, 1, n, m)
        K = F.conv2d(input, base, stride=stride, padding=pad, dilation=dilation)
        h, w = K.shape[-2:]     # h and w after padding, K.shape: [b*n_in, n_k, ~h, ~w]
        return K.reshape(b, n_in, n_k, h, w)

    def preconvolved_base_conv2d(self, input: 'torch.Tensor [b,i,w,h]', weight: 'np.array [n_out, n_in, k]',
                                 stride=1, padding='same', dilation=1):
        bases = KernelBase.preconvolve_base(input, self.base, stride=stride, padding=padding, dilation=dilation)
        b, n_in, k, h, w = bases.shape
        n_out, n_in_w, k_w = weight.shape

        assert n_in == n_in_w, f"The provided inputs and weights have different number of input neuron:\\ " +\
                               f"x.shape[1]=={n_in}, weigth.shape[1]=={n_in_w}."
        assert k == k_w, f"The provided weights have an incorrect number of kernels:\\ " +\
                         f"weight.shape[2]=={k_w}, but should be {k}."

        K = bases.permute(0, 3, 4, 1, 2).reshape(b, h, w, n_in*k)
        W = weight.reshape(n_out, n_in*k).transpose(0, 1)
        f = torch.matmul(K, W)  # K:[b,h,w,n_in*k] x W:[n_in*k, n_out] -> [b,h,w,n_out]
        return f.permute(0, 3, 1, 2)    # [b,n_out,h,w]

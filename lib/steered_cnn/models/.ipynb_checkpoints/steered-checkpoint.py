import torch
from torch import nn
from ..utils import cat_crop, pyramid_pool2d, normalize_vector
from ..steered_conv import SteeredConvBN, SteeredConvTranspose2d, SteerableKernelBase, OrthoKernelBase
from ..steered_conv.steerable_filters import cos_sin_ka_stack
from .backbones import UNet, HemelingNet

DEFAULT_STEERABLE_BASE = SteerableKernelBase.create_radial(5, max_k=5)
DEFAULT_ATTENTION_BASE = OrthoKernelBase.create_radial(5)
DEFAULT_STEERABLE_RESAMPLING_BASE = SteerableKernelBase.create_radial(2)
DEFAULT_ATTENTION_RESAMPLING_BASE = OrthoKernelBase.create_radial(3)


class SteeredUNet(UNet):
    def __init__(self, n_in, n_out, nfeatures=6, depth=2, nscale=5, padding='same',
                 p_dropout=0, batchnorm=True, downsampling='maxpooling', upsampling='conv',
                 base=DEFAULT_STEERABLE_BASE, rho_nonlinearity=False,
                 attention_mode=False, attention_base=DEFAULT_ATTENTION_BASE):
        """

        Args:
            n_in (int): Number of channels in the input image.
            n_out (int): Number of channels produced by the convolution.
            nfeatures (int or tuple, optional): Base number of features for each layer. Namely the number of features of the first scale.
            depth (int, optional): 
            nscale (int, optional): 
            
            
            padding (int, tuple or str, optional):
                Implicit paddings on both sides of the input. Can be a single number, a tuple (padH, padW) or
                one of 'true', 'same' and 'full'.
                      Default: 'same'
            p_dropout (float, optional): Probability of dropping out features during training. Drop out is disable if null.
                  Default: 0
            batchnorm (bool, optional): If True, adds batch normalization between the convolution layers and the activation functions.
                                        It also disables bias on convolution layers.
                   Default: True
            downsampling (str, optional): 
                Specify how the downsampling is performed. Can be one of:
                   - 'maxpooling': Maxpooling Layer.
                   - 'averagepooling': Average Pooling.
                   - 'conv': Stride on the last convolution.
                   Default: 'maxpooling'
           upsampling (str, optional): 
                Specify how the downsampling is performed. Can be one of:
                   - 'conv': Deconvolution with stride
                   - 'bilinear': Bilinear upsampling
                   - 'nearest': Nearest upsampling
                   Default: 'maxpooling'
            base (SteerableKernelBase, int or dict, optional):
                Steerable base which parametrize the steerable kernels. The weights are initialized accordingly.
                Can be a SteerableKernelBase, an integer interpreted as the desired equivalent kernel size or
                a dictionary with the specs of the base.
                See the documentation of SteerableKernelBase.parse() for more details on SteerableKernelBase specs.
                    Default: SteerableKernelBase.create_radial(5, max_k=5)
            rho_nonlinearity (str, optional):
                Apply a non-linearity on rho (or the norm of alpha if rho is None).
                Can be one of:
                  - None: rho is left unchanged (identity function);
                  - 'tanh': hyperbolic tangent non linear function;
                  - 'normalize': rho is set to 1 everywhere, only the angle information is kept.
                    Default: None
            attention_mode (str or bool, optional):
                Define how the attention submodule responsible of the kernels steering affects the output.
                Can be one of:
                  - 'shared': all the output features are steered with the same angles;
                  - 'feature': each output feature is steered with different angles;
                  - False: the attention submodules is disabled, alpha must be provided when calling forward().
                    Default: False
            attention_base (OrthoKernelBase, int or dict, optional):
                Orthogonal base which parametrize the attention sub-modules.
                Can be a SteerableKernelBase, an integer interpreted as the desired equivalent kernel size or
                a dictionary with the specs of the base.
                This parameter is ignored if `attention_mode` is set to False.
                See the documentation of OrthoKernelBase.parse() for more details on OrthoKernelBase specs.
                    Default: SteerableKernelBase.create_radial(5)
        """
        
        self.base = SteerableKernelBase.parse(base, default=DEFAULT_STEERABLE_BASE)
        self.attention_base = OrthoKernelBase.parse(attention_base, default=DEFAULT_ATTENTION_BASE)
        
        super(SteeredUNet, self).__init__(n_in, n_out, nfeatures=nfeatures, depth=depth,
                                          nscale=nscale, padding=padding, p_dropout=p_dropout, batchnorm=batchnorm,
                                          downsampling=downsampling, upsampling=upsampling,
                                          attention_mode=attention_mode, rho_nonlinearity=rho_nonlinearity)

    def setup_convbn(self, n_in, n_out, kernel=None, stride=1):
        if kernel is None:
            kernel = self.kernel
        opts = dict(attention_mode=self.attention_mode, rho_nonlinearity=self.rho_nonlinearity, stride=stride,
                    relu=True, bn=self.batchnorm, padding=self.padding)
        if kernel == 2:
            return SteeredConvBN(n_in, n_out,
                                 steerable_base=DEFAULT_STEERABLE_RESAMPLING_BASE,
                                 attention_base=DEFAULT_ATTENTION_RESAMPLING_BASE,
                                 **opts)
        else:
            return SteeredConvBN(n_in, n_out,
                                 steerable_base=self.base,
                                 attention_base=self.attention_base,
                                 **opts)

    def setup_convtranspose(self, n_in, n_out):
        return SteeredConvTranspose2d(n_in, n_out, stride=2,
                                      steerable_base=DEFAULT_STEERABLE_RESAMPLING_BASE,
                                      attention_base=DEFAULT_ATTENTION_RESAMPLING_BASE,
                                      attention_mode=self.attention_mode, rho_nonlinearity='normalize')

    def forward(self, x, alpha=None, rho=None):
        """
        Args:
            x: The input tensor.
            alpha: The angle by which the network is steered. (If None then alpha=0.)
                    This parameter can either be:
                        - a scalar: α
                        - 3D tensor: alpha[b, h, w]=α
                        - 4D tensor: alpha[b, 0, h, w]= ρ cos(α) and alpha[b, 1, h, w]= ρ sin(α).
                    (Alpha can be broadcasted along b, h or w, if these dimensions are of length 1.
                     It can also be a simple scalar.)
                    Default: None
            rho: The norm of the attention vector field. If None, the norm of alpha is used (the norm is set to 1
                 if alpha has only 3 dimensions). If provided, it will supplant the norm of alpha.
                 This parameter can either be:
                        - a scalar: ρ
                        - 3D tensor: rho[b, h, w]=ρ
                 Default: None

        Shape:
            input: (b, n_in, h, w)
            alpha: (b, [2,] ~h, ~w)     (b, h and w are broadcastable)
            rho:   (b, ~h, ~w)          (b, h and w are broadcastable)
            return: (b, n_out, ~h, ~w)

        Returns: The prediction of the network (without the sigmoid).

        """
        alpha_pyramid, rho_pyramid = attention_pyramid(alpha, rho, self, x.device)

        xscale = []
        for i, (conv_stack, downsample) in enumerate(zip(self.down_conv[:-1], self.downsample)):
            x = self.reduce_stack(conv_stack, x, alpha=alpha_pyramid[i], rho=rho_pyramid[i])
            xscale += [self.dropout(x)] if self.dropout_mode == 'shortcut' else [x]
            x = downsample(x, alpha=alpha_pyramid[i], rho=rho_pyramid[i])\
                if self.downsampling == 'conv' else downsample(x)

        x = self.reduce_stack(self.down_conv[-1], x, alpha=alpha_pyramid[-1], rho=rho_pyramid[-1])
        x = self.dropout(x)

        for i, conv_stack in enumerate(self.up_conv):
            x = cat_crop(xscale.pop(), self.upsample(x))
            x = self.reduce_stack(conv_stack, x, alpha=alpha_pyramid[-i], rho=rho_pyramid[-i])

        return self.final_conv(x)


class SteeredHemelingNet(HemelingNet):
    def __init__(self, n_in, n_out, nfeatures=6, depth=2, nscale=5, padding='same',
                 p_dropout=0, batchnorm=True, downsampling='maxpooling', upsampling='conv',
                 base=DEFAULT_STEERABLE_BASE, attention_base=False, attention_mode='shared', rho_nonlinearity=False):
        self.base = SteerableKernelBase.parse(base, default=DEFAULT_STEERABLE_BASE)
        self.attention_base = OrthoKernelBase.parse(attention_base, default=DEFAULT_ATTENTION_BASE)
        super(SteeredHemelingNet, self).__init__(n_in, n_out, nfeatures=nfeatures, depth=depth,
                                          nscale=nscale, padding=padding, p_dropout=p_dropout, batchnorm=batchnorm,
                                          downsampling=downsampling, upsampling=upsampling,
                                          attention_mode=attention_mode, rho_nonlinearity=rho_nonlinearity)

    def setup_convbn(self, n_in, n_out, kernel, stride=1):
        opts = dict(attention_mode=self.attention_mode, rho_nonlinearity=self.rho_nonlinearity, stride=stride,
                    relu=True, bn=self.batchnorm, padding=self.padding)
        if kernel == 2:
            return SteeredConvBN(n_in, n_out,
                                 steerable_base=DEFAULT_STEERABLE_RESAMPLING_BASE,
                                 attention_base=DEFAULT_ATTENTION_RESAMPLING_BASE,
                                 **opts)
        else:
            return SteeredConvBN(n_in, n_out,
                                 steerable_base=self.base,
                                 attention_base=self.attention_base,
                                 **opts)

    def setup_convtranspose(self, n_in, n_out):
        return SteeredConvTranspose2d(n_in, n_out, stride=2,
                                      steerable_base=DEFAULT_STEERABLE_RESAMPLING_BASE,
                                      attention_base=DEFAULT_ATTENTION_RESAMPLING_BASE,
                                      attention_mode=self.attention_mode, rho_nonlinearity='normalize')

    def forward(self, x, alpha=None, rho=None):
        """
        Args:
            x: The input tensor.
            alpha: The angle by which the network is steered. (If None then alpha=0.)
                    This parameter can either be:
                        - a scalar: α
                        - 3D tensor: alpha[b, h, w]=α
                        - 4D tensor: alpha[b, 0, h, w]= ρ cos(α) and alpha[b, 1, h, w]= ρ sin(α).
                    (Alpha can be broadcasted along b, h or w, if these dimensions are of length 1.
                     It can also be a simple scalar.)
                    Default: None
            rho: The norm of the attention vector field. If None, the norm of alpha is used (the norm is set to 1
                 if alpha has only 3 dimensions). If provided, it will supplant the norm of alpha.
                 This parameter can either be:
                        - a scalar: ρ
                        - 3D tensor: rho[b, h, w]=ρ
                 Default: None

        Shape:
            input: (b, n_in, h, w)
            alpha: (b, [2,] ~h, ~w)     (b, h and w are broadcastable)
            rho:   (b, ~h, ~w)          (b, h and w are broadcastable)
            return: (b, n_out, ~h, ~w)

        Returns: The prediction of the network (without the sigmoid).

        """
        alpha_pyramid, rho_pyramid = attention_pyramid(alpha, rho, self, x.device)

        xscale = []
        for i, (conv_stack, downsample) in enumerate(zip(self.down_conv[:-1], self.downsample)):
            x = self.reduce_stack(conv_stack, x, alpha=alpha_pyramid[i], rho=rho_pyramid[i])
            xscale += [self.dropout(x)] if self.dropout_mode == 'shortcut' else [x]
            x = downsample(x, alpha=alpha_pyramid[i], rho=rho_pyramid[i]) \
                if self.downsampling == 'conv' else downsample(x)

        x = self.reduce_stack(self.down_conv[-1], x, alpha=alpha_pyramid[-1], rho=rho_pyramid[-1])
        x = self.dropout(x)

        for conv_stack, upsample in zip(self.up_conv, self.upsample):
            x = upsample(x, alpha=alpha_pyramid[i], rho=rho_pyramid[i]) \
                if self.upsampling == 'conv' else upsample(x)
            x = cat_crop(xscale.pop(), x)
            x = self.reduce_stack(conv_stack, x, alpha=alpha_pyramid[-i], rho=rho_pyramid[-i])

        return self.final_conv(x)


def attention_pyramid(alpha, rho, module, device=None):
    k_max = module.base.k_max
    N = module.nscale
    if alpha is None:
        if module.attention_base is None:
            raise ValueError('If no attention base is specified, a steering angle alpha should be provided.')
        return [None]*N, [None]*N
    else:
        with torch.no_grad():
            if isinstance(alpha, (int, float)):
                if alpha == 0:
                    alpha = None
                else:
                    alpha = torch.Tensor([alpha]).to(device=device)
                    alpha = torch.stack((torch.cos(alpha), torch.sin(alpha)))[:, None, None, None]

            alpha_rho = 1
            if alpha.dim() == 3:
                cos_sin_kalpha = cos_sin_ka_stack(torch.cos(alpha), torch.sin(alpha), k=k_max)
            elif alpha.dim() == 4 and alpha.shape[1] == 2:
                alpha = alpha.transpose(0, 1)
                alpha, alpha_rho = normalize_vector(alpha)
                cos_sin_kalpha = cos_sin_ka_stack(alpha[0], alpha[1], k=k_max)
            else:
                raise ValueError(f'alpha shape should be either [b, h, w] or [b, 2, h, w] '
                                 f'but provided tensor shape is {alpha.shape}.')
            cos_sin_kalpha = cos_sin_kalpha.unsqueeze(3)
            alpha_pyramid = pyramid_pool2d(cos_sin_kalpha, n=N)

            if rho is None:
                rho = alpha_rho
            elif isinstance(rho, (int, float)):
                rho = torch.Tensor([rho]).to(device=x.device)
                rho = torch.stack((torch.cos(rho), torch.sin(rho)))[:, None, None, None]

            if module.rho_nonlinearity == 'normalize':
                rho = 1
            elif module.rho_nonlinearity == 'tanh':
                rho = torch.tanh(rho)
            rho_pyramid = [rho]*N if not isinstance(rho, torch.Tensor) else pyramid_pool2d(rho, n=N)
        return alpha_pyramid, rho_pyramid



class SteeredHemelingNetOld(nn.Module):
    def __init__(self, n_in, n_out=1, nfeatures_base=6, depth=2, base=None, attention=None,
                 p_dropout=0, padding='same', batchnorm=True, upsample='conv'):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.upsample = upsample

        # --- MODEL ---
        n1 = nfeatures_base
        n2 = nfeatures_base * 2
        n3 = nfeatures_base * 4
        n4 = nfeatures_base * 8
        n5 = nfeatures_base * 16

        # Down
        self.conv1 = nn.ModuleList(
            [SteeredConvBN(n_in, n1, relu=True, bn=batchnorm, padding=padding, steerable_base=base, attention_base=attention)]
            + [SteeredConvBN(n1, n1, relu=True, bn=batchnorm, padding=padding, steerable_base=base, attention_base=attention)
               for _ in range(depth - 1)])
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.ModuleList(
            [SteeredConvBN(n1, n2, relu=True, bn=batchnorm, padding=padding, steerable_base=base, attention_base=attention)]
            + [SteeredConvBN(n2, n2, relu=True, bn=batchnorm, padding=padding, steerable_base=base, attention_base=attention)
               for _ in range(depth - 1)])
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.ModuleList(
            [SteeredConvBN(n2, n3, relu=True, bn=batchnorm, padding=padding, steerable_base=base, attention_base=attention)]
            + [SteeredConvBN(n3, n3, relu=True, bn=batchnorm, padding=padding, steerable_base=base, attention_base=attention)
               for _ in range(depth - 1)])
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.ModuleList(
            [SteeredConvBN(n3, n4, relu=True, bn=batchnorm, padding=padding, steerable_base=base, attention_base=attention)]
            + [SteeredConvBN(n4, n4, relu=True, bn=batchnorm, padding=padding, steerable_base=base, attention_base=attention)
               for _ in range(depth - 1)])
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = nn.ModuleList(
            [SteeredConvBN(n4, n5, relu=True, bn=batchnorm, padding=padding, steerable_base=base, attention_base=attention)]
            + [SteeredConvBN(n5, n5, relu=True, bn=batchnorm, padding=padding, steerable_base=base, attention_base=attention)
               for _ in range(depth - 1)])

        # Up
        if upsample == 'nearest':
            self.upsample1 = nn.Sequential(nn.Conv2d(n5, n4, kernel_size=(1, 1)), nn.Upsample(scale_factor=2))
        else:
            self.upsample1 = nn.ConvTranspose2d(n5, n4, kernel_size=(2, 2), stride=(2, 2))
        self.conv6 = nn.ModuleList(
            [SteeredConvBN(2 * n4, n4, relu=True, bn=batchnorm, padding=padding, steerable_base=base, attention_base=attention)]
            + [SteeredConvBN(n4, n4, relu=True, bn=batchnorm, padding=padding, steerable_base=base, attention_base=attention)
               for _ in range(depth - 1)])

        if upsample == 'nearest':
            self.upsample2 = nn.Sequential(nn.Conv2d(n4, n3, kernel_size=(1, 1)), nn.Upsample(scale_factor=2))
        else:
            self.upsample2 = nn.ConvTranspose2d(n4, n3, kernel_size=(2, 2), stride=(2, 2))
        self.conv7 = nn.ModuleList(
            [SteeredConvBN(2 * n3, n3, relu=True, bn=batchnorm, padding=padding, steerable_base=base, attention_base=attention)]
            + [SteeredConvBN(n3, n3, relu=True, bn=batchnorm, padding=padding, steerable_base=base, attention_base=attention)
               for _ in range(depth - 1)])

        if upsample == 'nearest':
            self.upsample3 = nn.Sequential(nn.Conv2d(n3, n2, kernel_size=(1, 1)), nn.Upsample(scale_factor=2))
        else:
            self.upsample3 = nn.ConvTranspose2d(n3, n2, kernel_size=(2, 2), stride=(2, 2))
        self.conv8 = nn.ModuleList(
            [SteeredConvBN(2 * n2, n2, relu=True, bn=batchnorm, padding=padding, steerable_base=base, attention_base=attention)]
            + [SteeredConvBN(n2, n2, relu=True, bn=batchnorm, padding=padding, steerable_base=base, attention_base=attention)
               for _ in range(depth - 1)])

        if upsample == 'nearest':
            self.upsample4 = nn.Sequential(nn.Conv2d(n2, n1, kernel_size=(1, 1)), nn.Upsample(scale_factor=2))
        else:
            self.upsample4 = nn.ConvTranspose2d(n2, n1, kernel_size=(2, 2), stride=(2, 2))
        self.conv9 = nn.ModuleList(
            [SteeredConvBN(2 * n1, n1, relu=True, bn=batchnorm, padding=padding, steerable_base=base, attention_base=attention)]
            + [SteeredConvBN(n1, n1, relu=True, bn=batchnorm, padding=padding, steerable_base=base, attention_base=attention)
               for _ in range(depth - 1)])

        # End
        self.final_conv = nn.Conv2d(n1, 1, kernel_size=(1, 1))

        self.dropout = torch.nn.Dropout(p_dropout) if p_dropout else identity

    @property
    def attention(self):
        return self.conv1[0].conv.attention_base

    @property
    def base(self):
        return self.conv1[0].conv.steerable_base

    def forward(self, x, alpha=None, **kwargs):
        """
        Args:
            x: The input tensor.
            alpha: The angle by which the network is steered. (If None then alpha=0.)
                    To enhance the efficiency of the steering computation, α should not be provided as an angle but as a
                    vertical and horizontal projection: cos(α) and sin(α).
                    Hence this parameter should be a 4D tensor: alpha[b, 0, h, w]=cos(α) and alpha[b, 1, h, w]=sin(α).
                    (Alpha can be broadcasted along b, h or w, if these dimensions are of length 1.)
                    Default: None

        Shape:
            input: (b, n_in, h, w)
            alpha: (b, 2, ~h, ~w)     (b, h and w are broadcastable)
            return: (b, n_out, ~h, ~w)

        Returns: The prediction of the network (without the sigmoid).

        """
        from functools import reduce

        N = 5
        if alpha is None:
            if self.attention is None:
                raise NotImplementedError()
            else:
                alpha_pyramid = [None]*N
                rho_pyramid = [None]*N
        else:
            with torch.no_grad():
                k_max = self.base.k_max

                rho = 1
                if alpha.dim() == 3:
                    cos_sin_kalpha = cos_sin_ka_stack(torch.cos(alpha), torch.sin(alpha), k=k_max)
                elif alpha.dim() == 4 and alpha.shape[1] == 2:
                    alpha = alpha.transpose(0, 1)
                    alpha, rho = normalize_vector(alpha)
                    cos_sin_kalpha = cos_sin_ka_stack(alpha[0], alpha[1], k=k_max)
                else:
                    raise ValueError(f'alpha shape should be either [b, h, w] or [b, 2, h, w] '
                                     f'but provided tensor shape is {alpha.shape}.')
                cos_sin_kalpha = cos_sin_kalpha.unsqueeze(3)

                alpha_pyramid = pyramid_pool2d(cos_sin_kalpha, n=N)
                rho_pyramid = [rho]*N if not isinstance(rho, torch.Tensor) else pyramid_pool2d(rho, n=N)

        # Down
        x1 = reduce(lambda X, conv: conv(X, alpha=alpha_pyramid[0], rho=rho_pyramid[0]), self.conv1, x)

        x2 = self.pool1(x1)
        x2 = reduce(lambda X, conv: conv(X, alpha=alpha_pyramid[1], rho=rho_pyramid[1]), self.conv2, x2)

        x3 = self.pool2(x2)
        x3 = reduce(lambda X, conv: conv(X, alpha=alpha_pyramid[2], rho=rho_pyramid[2]), self.conv3, x3)

        x4 = self.pool3(x3)
        x4 = reduce(lambda X, conv: conv(X, alpha=alpha_pyramid[3], rho=rho_pyramid[3]), self.conv4, x4)

        x5 = self.pool4(x4)
        x5 = reduce(lambda X, conv: conv(X, alpha=alpha_pyramid[4], rho=rho_pyramid[4]), self.conv5, x5)
        x5 = self.dropout(x5)

        # Up
        x4 = cat_crop(x4, self.upsample1(x5))
        del x5
        x4 = reduce(lambda X, conv: conv(X, alpha=alpha_pyramid[3], rho=rho_pyramid[3]), self.conv6, x4)

        x3 = cat_crop(x3, self.upsample2(x4))
        del x4
        x3 = reduce(lambda X, conv: conv(X, alpha=alpha_pyramid[2], rho=rho_pyramid[2]), self.conv7, x3)

        x2 = cat_crop(x2, self.upsample3(x3))
        del x3
        x2 = reduce(lambda X, conv: conv(X, alpha=alpha_pyramid[1], rho=rho_pyramid[1]), self.conv8, x2)

        x1 = cat_crop(x1, self.upsample4(x2))
        del x2
        x1 = reduce(lambda X, conv: conv(X, alpha=alpha_pyramid[0], rho=rho_pyramid[0]), self.conv9, x1)

        # End
        return self.final_conv(x1)

    @property
    def p_dropout(self):
        return self.dropout.p

    @p_dropout.setter
    def p_dropout(self, p):
        self.dropout.p = p

        
def identity(x):
    return x
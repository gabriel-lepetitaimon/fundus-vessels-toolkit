from torch import nn
from .steered_conv import SteeredConv2d


class SteeredConvBN(nn.Module):
    def __init__(self, n_in, n_out=None, stride=1, padding='same', dilation=1, bn=False, relu=True,
                 steerable_base=None, rho_nonlinearity=None,
                 attention_mode=False, attention_base=None):
        """

        Args:
            n_in (int): Number of channels in the input image.
            n_out (int): Number of channels produced by the convolution.
            stride (int or tuple, optional): Stride of the convolution. Default: 1.

            padding (int, tuple or str, optional):
                Implicit paddings on both sides of the input. Can be a single number, a tuple (padH, padW) or
                one of 'true', 'same' and 'full'.
                      Default: 'same'
            dilation (int or tuple, optional):
                The spacing between kernel elements. Can be a single number or a tuple (dH, dW).
                  Default: 1
            bn (bool, optional): If True, adds batch normalization between the convolution and the activation function.
                                 It also disables bias.
                   Default: True
            relu (bool, optional): If True, adds a ReLU activation function if batch normalization is active or a SeLU
                                   function if batch normalization is disabled.
                   Default: True
            steerable_base (SteerableKernelBase, int or dict, optional):
                Steerable base which parametrize the steerable kernels. The weights are initialized accordingly.
                Can be a SteerableKernelBase, an integer interpreted as the desired equivalent kernel size or
                a dictionary with the specs of the base.
                See the documentation of SteerableKernelBase.parse() for more details on SteerableKernelBase specs.
                    Default: SteerableKernelBase.create_radial(3)
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
                    Default: SteerableKernelBase.create_radial(3)
        """
        super().__init__()

        self._bn = bn
        self._relu = relu
        if n_out is None:
            n_out = n_in
        self.n_out = n_out
        self.conv = SteeredConv2d(n_in, n_out, steerable_base=steerable_base, stride=stride,
                                  padding=padding, bias=not bn, dilation=dilation, attention_base=attention_base,
                                  attention_mode=attention_mode, rho_nonlinearity=rho_nonlinearity,
                                  nonlinearity=('relu' if bn else 'selu') if relu else 'linear')
        bn_relu = []
        if bn:
            bn_relu += [nn.BatchNorm2d(n_out)]
            if relu:
                bn_relu += [nn.ReLU()]
        elif relu:
            bn_relu += [nn.SELU()]

        self.bn_relu = nn.Sequential(*bn_relu)

    def forward(self, x, alpha=None, rho=None):
        x = self.conv(x, alpha=alpha, rho=rho)
        return self.bn_relu(x)

    @property
    def bn(self):
        if self._bn:
            return self.bn_relu[0]
        return None

    @property
    def relu(self):
        if self._relu:
            return self.bn_relu[1 if self._bn else 0]
        return None

    def __getattr__(self, item):
        if item in ('stride', 'padding', 'dilation'):
            return getattr(self.conv, item)
        return super().__getattr__(item)

    def __setattr__(self, key, value):
        if key in ('stride', 'padding', 'dilation'):
            setattr(self.conv, key, value)
        else:
            super().__setattr__(key, value)

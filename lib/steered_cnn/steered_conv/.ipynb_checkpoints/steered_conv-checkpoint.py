import torch
from torch import nn
import math

from .steered_kbase import SteerableKernelBase
from .ortho_kbase import OrthoKernelBase
from ..utils.clip_pad import normalize_vector

DEFAULT_STEERABLE_BASE = SteerableKernelBase.create_radial(3)
DEFAULT_ATTENTION_BASE = OrthoKernelBase.create_radial(3)


class SteeredConv2d(nn.Module):
    def __init__(self, n_in, n_out=None, stride=1, padding='same', dilation=1, bias=True,
                 steerable_base: (SteerableKernelBase, int, dict) = None, rho_nonlinearity=None,
                 attention_mode=False, attention_base: (OrthoKernelBase, int, dict) = None,
                 nonlinearity='relu', nonlinearity_param=None):
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
            bias (bool, optional): If True, adds a learnable bias to the output.
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
            nonlinearity: Type of nonlinearity used after the convolution.
                          See torch.nn.init.calculate_gain() documentation for more details.
                          Default: 'relu'
            nonlinearity_param: Optional parameter for the non-linear function.
                                See torch.nn.init.calculate_gain() documentation for more details.
        """
        super(SteeredConv2d, self).__init__()

        if n_out is None:
            n_out = n_in
        
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.steerable_base = SteerableKernelBase.parse(steerable_base, DEFAULT_STEERABLE_BASE)
        self.attention_base = OrthoKernelBase.parse(attention_base, default=DEFAULT_ATTENTION_BASE)

        # Weight
        self.weights = nn.Parameter(self.steerable_base.init_weights(n_in, n_out, nonlinearity, nonlinearity_param),
                                    requires_grad=True)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(n_out), requires_grad=True) if bias else None
            torch.nn.init.constant_(self.bias, 0)
            
        self.attention_mode = attention_mode
        self.rho_nonlinearity = rho_nonlinearity
        if attention_mode:
            w = self.attention_base.init_weights(n_in, n_out if attention_mode == 'feature' else 1,
                                                 nonlinearity='linear')
            self.attention_weights = nn.Parameter(w, requires_grad=True)
        else:
            self.attention_base = None
            self.attention_weights = None

    def forward(self, x, alpha=None, rho=None):
        """

        Args:
            x: Input images.
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
                        - None: alpha is computed using the attention submodule (attention_mode must be specified).
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

        Returns:

        """
        if alpha is None:
            if self.attention_base:
                alpha = self.attention_base.ortho_conv2d(x, self.attention_weights,
                                                         stride=self.stride, padding=self.padding)
                alpha, rho = normalize_vector(alpha)
            else:
                raise ValueError('Either attention_base or alpha should be provided when computing the result of a '
                                 'SteeredConv2d module.')

        out = self.steerable_base.conv2d(x, self.weights, alpha=alpha, rho=rho, rho_nonlinearity=self.rho_nonlinearity,
                                         stride=self.stride, padding=self.padding, dilation=self.dilation)

        # Bias
        if self.bias is not None:
            out += self.bias[None, :, None, None]
        return out


class SteeredConvTranspose2d(nn.Module):
    def __init__(self, n_in, n_out=None, stride=2, padding='same', output_padding=0, dilation=1, bias=True,
                 steerable_base: SteerableKernelBase = None, rho_nonlinearity=None,
                 attention_mode='feature', attention_base: (OrthoKernelBase, int, dict) = None,
                 nonlinearity='linear', nonlinearity_param=None):
        """

        Args:
            n_in (int): Number of channels in the input image.
            n_out (int): Number of channels produced by the convolution.
            stride (int or tuple, optional): Stride of the convolution. Default: 1.

            padding (int, tuple or str, optional):
                Implicit paddings on both sides of the input. Can be a single number, a tuple (padH, padW) or
                one of 'true', 'same' and 'full'.
                      Default: 'same'
            output_padding (int, tuple or str, optional):
                Implicit paddings on both sides of the input. Can be a single number, a tuple (padH, padW) or
                one of 'true', 'same' and 'full'.
                      Default: 'same'
            dilation (int or tuple, optional):
                The spacing between kernel elements. Can be a single number or a tuple (dH, dW).
                      Default: 1
            bias (bool, optional): If True, adds a learnable bias to the output.
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
            nonlinearity: Type of nonlinearity used after the convolution.
                          See torch.nn.init.calculate_gain() documentation for more details.
                          Default: 'linear'
            nonlinearity_param: Optional parameter for the non-linear function.
                                See torch.nn.init.calculate_gain() documentation for more details.
        """
        super(SteeredConvTranspose2d, self).__init__()

        if n_out is None:
            n_out = n_in
        
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.steerable_base = SteerableKernelBase.parse(steerable_base, SteerableKernelBase.create_radial(stride))
        self.attention_base = OrthoKernelBase.parse(attention_base, OrthoKernelBase.create_radial(stride))

        # Weight
        self.weights = nn.Parameter(self.steerable_base.init_weights(n_out, n_in, nonlinearity, nonlinearity_param),
                                    requires_grad=True)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(n_out), requires_grad=True) if bias else None
            # b = 1*math.sqrt(1/n_out)
            # nn.init.uniform_(self.bias, -b, b)

        if attention_mode:
            self.attention_mode = attention_mode
            self.rho_nonlinearity = rho_nonlinearity
            w = self.attention_base.init_weights(n_in, n_out if attention_mode == 'feature' else 1,
                                                 nonlinearity='linear')
            self.attention_weights = nn.Parameter(w, requires_grad=True)

    def forward(self, x, alpha=None, rho=None):
        """

        Args:
            x: Input images.
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
                        - None: alpha is computed using the attention submodule (attention_mode must be specified).
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

        Returns:

        """
        if alpha is None:
            if self.attention_base:
                alpha = self.attention_base.ortho_conv2d(x, self.attention_weights,
                                                         stride=self.stride, padding=self.padding)
                alpha, rho = normalize_vector(alpha)
            else:
                raise ValueError('Either attention_base or alpha should be provided when computing the result of a '
                                 'SteeredConv2d module.')

        out = self.steerable_base.conv_transpose2d(x, self.weights, alpha=alpha, rho=rho, stride=self.stride,
                                                   padding=self.padding, output_padding=self.output_padding,
                                                   dilation=self.dilation)

        # Bias
        if self.bias is not None:
            out += self.bias[None, :, None, None]
        return out
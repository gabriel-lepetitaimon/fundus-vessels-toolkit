import torch
from torch import nn
from collections import OrderedDict
from ...utils import cat_crop
from ...utils.convbn import ConvBN
from .model import Model


class HemelingNet(Model):
    def __init__(self, n_in, n_out=1, nfeatures=64, kernel=3, depth=2, nscale=5, padding='same',
                 p_dropout=0, dropout_mode='shortcut', batchnorm=True, downsampling='maxpooling', upsampling='conv',
                 **kwargs):
        """
        :param n_in: Number of input features (or channels).
        :param n_out: Number of output features (or classes).
        :param nfeatures: Number of features of each convolution block for the first stage of the UNet.
                          The number of features doubles at every stage.
        :param kernel: Height and Width of the convolution kernels
        :param depth: Number of convolution block in each stage.
        :param nscale: Number of downsampling stage.
        :param padding: Padding configuration [One of 'same' or 'auto'; 'true' or 'valid'; 'full'; number of padded pixel].
        :param p_dropout: Dropout probability during training.
        :param dropout_mode: Decide where to apply dropout. [One of 'shortcut' or 'bottom']
        :param batchnorm: Adds batchnorm layers before each convolution layer.
        :param downsampling:
            - maxpooling: Maxpooling Layer.
            - averagepooling: Average Pooling.
            - conv: Stride on the last convolution.
        :param upsampling:
            - conv: Deconvolution with stride
            - bilinear: Bilinear upsampling
            - nearest: Nearest upsampling
        """
        super().__init__(n_in=n_in, n_out=n_out, nfeatures=nfeatures, depth=depth, nscale=nscale,
                         kernel=kernel, padding=padding, p_dropout=p_dropout, dropout_mode=dropout_mode.lower(),
                         batchnorm=batchnorm, downsampling=downsampling.lower(), upsampling=upsampling.lower(),
                         **kwargs)

        if isinstance(nfeatures, int):
            nfeatures = [nfeatures*(2**scale) for scale in range(nscale)]
        if isinstance(nfeatures, (tuple, list)):
            if len(nfeatures) == nscale:
                nfeatures = nfeatures + list(reversed(nfeatures[:-1]))
            elif len(nfeatures) != nscale*2-1:
                raise ValueError(f'Invalid length for nfeatures: {nfeatures}.\n '
                                 f'Should be nscale={nscale} or nscale*2-1={nscale*2-1}.')

        # Down
        self.down_conv = []
        self.downsample = []
        if downsampling.lower() not in ('maxpooling', 'averagepooling', 'conv'):
            raise ValueError(f'downsampling must be one of: "maxpooling", "averagepooling", "conv". '
                             f'Provided: {downsampling}.')
        for i in range(nscale):
            nf_prev = n_in if i == 0 else nfeatures[i-1]
            nf_scale = nfeatures[i]
            conv_stack = [self.setup_convbn(nf_prev, nf_scale, kernel)]
            conv_stack += [self.setup_convbn(nf_scale, nf_scale, kernel) for _ in range(depth-1)]
            self.down_conv += [conv_stack]
            for j, mod in enumerate(conv_stack):
                self.add_module(f'downconv{i}-{j}', mod)

            if i != nscale-1:
                if downsampling.lower() == 'conv':
                    downsample = self.setup_convbn(nf_scale, nf_scale, kernel=2, stride=2)
                else:
                    downsample = nn.MaxPool2d(2) if downsampling.lower()=='maxpooling' else nn.AvgPool2d(2)
                self.downsample += [downsample]
                self.add_module(f'downsample{i}', downsample)

        # Up Conv
        self.up_conv = []
        self.upsample = []
        if upsampling.lower() not in ('conv', 'linear', 'bilinear', 'bicubic', 'nearest'):
            raise ValueError(f'upsampling must be one of: "linear", "bilinear", "bicubic", "nearest", "conv". '
                             f'Provided: {upsampling}.')
        for i in range(nscale, 2*nscale-1):
            nf_prev = nfeatures[i-1]
            nf_scale = nfeatures[i]
            nf_concat = nf_scale + nfeatures[2*nscale-i-2]
            if upsampling.lower() == 'conv':
                upsample = self.setup_convtranspose(nf_prev, nf_scale)
            else:
                upsample = nn.Sequential(OrderedDict({
                        'conv': ConvBN(1, nf_prev, nf_scale, bn=False, relu=False),
                        'upsample': torch.nn.Upsample(scale_factor=2, mode=upsampling.lower())}))
            self.upsample += [upsample]
            self.add_module(f'upsample{i}', upsample)

            conv_stack = [self.setup_convbn(nf_concat, nf_scale, kernel)]
            conv_stack += [self.setup_convbn(nf_scale, nf_scale, kernel) for _ in range(depth-1)]
            self.up_conv += [conv_stack]
            for j, mod in enumerate(conv_stack):
                self.add_module(f'upconv{i}-{j}', mod)

        # End
        self.final_conv = ConvBN(1, nfeatures[-1], n_out, relu=False, bn=False)
        self.dropout = torch.nn.Dropout(p_dropout) if p_dropout else identity

    def setup_convbn(self, n_in, n_out, kernel, stride=1):
        return ConvBN(kernel, n_in, n_out, relu=True, bn=self.batchnorm, padding=self.padding, stride=stride)

    def setup_convtranspose(self, n_in, n_out):
        return torch.nn.ConvTranspose2d(n_in, n_out, kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x):
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
        xscale = []
        for conv_stack, downsample in zip(self.down_conv[:-1], self.downsample):
            x = self.reduce_stack(conv_stack, x)
            xscale += [self.dropout(x)] if self.dropout_mode == 'shortcut' else [x]
            x = downsample(x)

        x = self.reduce_stack(self.down_conv[-1], x)
        x = self.dropout(x)

        for conv_stack, upsample in zip(self.up_conv, self.upsample):
            x = cat_crop(xscale.pop(), upsample(x))
            x = self.reduce_stack(conv_stack, x)

        return self.final_conv(x)

    @property
    def p_dropout(self):
        return self.dropout.p

    @p_dropout.setter
    def p_dropout(self, p):
        self.dropout.p = p


def identity(x):
    return x

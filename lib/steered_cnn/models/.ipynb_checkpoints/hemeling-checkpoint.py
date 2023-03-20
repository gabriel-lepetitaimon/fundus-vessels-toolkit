import torch
from torch import nn
from ..utils import ConvBN, cat_crop


class HemelingNet(nn.Module):
    def __init__(self, n_in, n_out=1, nfeatures_base=16, depth=2, half_kernel_height=3,
                 p_dropout=0, padding='same', batchnorm=True):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.kernel_height = half_kernel_height*2-1

        # --- MODEL ---
        n1 = nfeatures_base
        n2 = nfeatures_base * 2
        n3 = nfeatures_base * 4
        n4 = nfeatures_base * 8
        n5 = nfeatures_base * 16

        kernel = self.kernel_height

        # Down
        self.conv1 = nn.ModuleList(
            [ConvBN(kernel, n_in, n1, relu=True, bn=batchnorm, padding=padding)]
            + [ConvBN(kernel, n1, n1, relu=True, bn=batchnorm, padding=padding)
               for _ in range(depth - 1)])
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.ModuleList(
            [ConvBN(kernel, n1, n2, relu=True, bn=batchnorm, padding=padding)]
            + [ConvBN(kernel, n2, n2, relu=True, bn=batchnorm, padding=padding)
               for _ in range(depth - 1)])
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.ModuleList(
            [ConvBN(kernel, n2, n3, relu=True, bn=batchnorm, padding=padding)]
            + [ConvBN(kernel, n3, n3, relu=True, bn=batchnorm, padding=padding)
               for _ in range(depth - 1)])
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.ModuleList(
            [ConvBN(kernel, n3, n4, relu=True, bn=batchnorm, padding=padding)]
            + [ConvBN(kernel, n4, n4, relu=True, bn=batchnorm, padding=padding)
               for _ in range(depth - 1)])
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = nn.ModuleList(
            [ConvBN(kernel, n4, n5, relu=True, bn=batchnorm, padding=padding)]
            + [ConvBN(kernel, n5, n5, relu=True, bn=batchnorm, padding=padding)
               for _ in range(depth - 1)])

        # Up
        self.upsample1 = nn.ConvTranspose2d(n5, n4, kernel_size=(2, 2), stride=(2, 2))
        self.conv6 = nn.ModuleList(
            [ConvBN(kernel, 2 * n4, n4, relu=True, bn=batchnorm, padding=padding)]
            + [ConvBN(kernel, n4, n4, relu=True, bn=batchnorm, padding=padding)
               for _ in range(depth - 1)])

        self.upsample2 = nn.ConvTranspose2d(n4, n3, kernel_size=(2, 2), stride=(2, 2))
        self.conv7 = nn.ModuleList(
            [ConvBN(kernel, 2 * n3, n3, relu=True, bn=batchnorm, padding=padding)]
            + [ConvBN(kernel, n3, n3, relu=True, bn=batchnorm, padding=padding)
               for _ in range(depth - 1)])

        self.upsample3 = nn.ConvTranspose2d(n3, n2, kernel_size=(2, 2), stride=(2, 2))
        self.conv8 = nn.ModuleList(
            [ConvBN(kernel, 2 * n2, n2, relu=True, bn=batchnorm, padding=padding)]
            + [ConvBN(kernel, n2, n2, relu=True, bn=batchnorm, padding=padding)
               for _ in range(depth - 1)])

        self.upsample4 = nn.ConvTranspose2d(n2, n1, kernel_size=(2, 2), stride=(2, 2))
        self.conv9 = nn.ModuleList(
            [ConvBN(kernel, 2 * n1, n1, relu=True, bn=batchnorm, padding=padding)]
            + [ConvBN(kernel, n1, n1, relu=True, bn=batchnorm, padding=padding)
               for _ in range(depth - 1)])

        # End
        self.final_conv = nn.Conv2d(n1, 1, kernel_size=(1, 1))

        self.dropout = torch.nn.Dropout(p_dropout) if p_dropout else identity

    def forward(self, x, **kwargs):
        from functools import reduce
        
        # Down
        x1 = reduce(lambda X, conv: conv(X), self.conv1, x)

        x2 = self.pool1(x1)
        x2 = reduce(lambda X, conv: conv(X), self.conv2, x2)

        x3 = self.pool2(x2)
        x3 = reduce(lambda X, conv: conv(X), self.conv3, x3)

        x4 = self.pool3(x3)
        x4 = reduce(lambda X, conv: conv(X), self.conv4, x4)

        x5 = self.pool4(x4)
        x5 = reduce(lambda X, conv: conv(X), self.conv5, x5)
        x5 = self.dropout(x5)

        # Up
        x4 = cat_crop(x4, self.upsample1(x5))
        del x5
        x4 = reduce(lambda X, conv: conv(X), self.conv6, x4)

        x3 = cat_crop(x3, self.upsample2(x4))
        del x4
        x3 = reduce(lambda X, conv: conv(X), self.conv7, x3)

        x2 = cat_crop(x2, self.upsample3(x3))
        del x3
        x2 = reduce(lambda X, conv: conv(X), self.conv8, x2)

        x1 = cat_crop(x1, self.upsample4(x2))
        del x2
        x1 = reduce(lambda X, conv: conv(X), self.conv9, x1)

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
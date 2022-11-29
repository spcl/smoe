"""A convolution layer followed by a locally-connected layer."""

import torch

from . lcn import LocallyConnected2d


class ConvLCN2d(torch.nn.Module):
    """A convolution followed by a locally-connected layer."""

    def __init__(self, in_planes, out_planes, kernel_size, input_shape,
                 inner_planes=3, stride=1, padding=0, bias=False):
        # TODO: Support passing inner planes from args.
        super().__init__()
        if inner_planes is None:
            inner_planes = out_planes
        self.conv = torch.nn.Conv2d(
            in_planes, inner_planes, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=bias)
        self.lcn = LocallyConnected2d(
            inner_planes, out_planes, kernel_size=1, input_shape=input_shape,
            stride=1, padding=0, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.lcn(x)
        return x

"""Coordinate-augmented convolution layer."""

import torch


class CoordConv2d(torch.nn.Module):
    """
    Coordinate-augmented convolution.

    Adapted from some of the public PyTorch ports:
    https://github.com/mkocabas/CoordConv-pytorch
    https://github.com/walsvid/CoordConv
    https://github.com/Wizaron/coord-conv-pytorch/blob/master/coord_conv.py

    """

    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
                 padding=0, groups=1, dilation=1, bias=False, with_r=False):
        super().__init__()
        self.with_r = with_r
        in_planes += 3 if with_r else 2
        self.conv = torch.nn.Conv2d(
            in_planes, out_planes, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=groups, dilation=dilation,
            bias=bias)
        # Used to cache coordinates to avoid reconstruction.
        self.saved_coords = None

    def construct_coords(self, x):
        """Construct coordinate planes to match x."""
        batch_size, _, x_dim, y_dim = x.size()
        xx_channel = torch.arange(x_dim, dtype=x.dtype, device=x.device).repeat(1, y_dim, 1)
        xx_channel /= x_dim - 1
        xx_channel = xx_channel * 2 - 1
        yy_channel = torch.arange(y_dim, dtype=x.dtype, device=x.device).repeat(1, x_dim, 1).transpose(1, 2)
        yy_channel /= y_dim - 1
        yy_channel = yy_channel * 2 - 1
        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        if self.with_r:
            rr_channel = torch.sqrt(torch.pow(xx_channel - 0.5, 2)
                                    + torch.pow(yy_channel - 0.5, 2))
            self.saved_coords = torch.cat([xx_channel, yy_channel, rr_channel], dim=1)
        else:
            self.saved_coords = torch.cat([xx_channel, yy_channel], dim=1)

    def add_coords(self, x):
        """Insert coordinate planes into x."""
        if (self.saved_coords is None
            or (x.size(0),) + x.size()[2:] != (self.saved_coords.size(0),) + self.saved_coords.size()[2:]):
            self.construct_coords(x)
        return torch.cat([x, self.saved_coords], dim=1)

    def forward(self, x):
        x = self.add_coords(x)
        x = self.conv(x)
        return x

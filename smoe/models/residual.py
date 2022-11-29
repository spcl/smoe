"""Generic support for building residual networks."""

import torch


class ResidualBlock(torch.nn.Module):
    """Implements a residual block with various options for resizing"""

    def __init__(self, layers, resize_block=None, downsample_slice=None):
        """Define layers and downsampling method (if needed).

        layers is an iterable of (name, layer)s.

        resize_block is a module that can be used to resize the
        input.

        downsample_slice is a slice that is used to cut from the input
        channel-wise to downsample.

        resize_block and downsample_slice are mutually exclusive.

        """
        super().__init__()
        if resize_block is not None and downsample_slice is not None:
            raise ValueError('Cannot specify resize_block and downsample_slice')
        self.layers = torch.nn.ModuleDict(layers)
        self.resize_block = resize_block
        self.downsample_slice = downsample_slice

    def forward(self, x):
        identity = x
        aux = []
        for layer in self.layers.values():
            x = layer(x)
            if isinstance(x, tuple):
                aux += list(x[1:])
                x = x[0]
        if self.resize_block is not None:
            identity = self.resize_block(identity)
        elif self.downsample_slice is not None:
            identity = identity[:, self.downsample_slice, :]
        x += identity
        if aux:
            return x, *aux
        return x

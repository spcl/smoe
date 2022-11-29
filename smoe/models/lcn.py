"""Locally-connected layer."""

from typing import Union
import math

import torch
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

from . import init


class LocallyConnected2d(torch.nn.Module):
    """
    2D locally-connected layer.

    This currently uses a simple unfold, element-wise product, and sum
    implementation.

    """

    def __init__(self, in_planes: int, out_planes: int,
                 kernel_size: Union[int, tuple[int, int]],
                 input_shape: Union[int, tuple[int, int]],
                 stride: int = 1, padding: Union[int, str] = 0, bias: bool = False) -> None:
        super().__init__()
        if bias:
            raise RuntimeError('LocallyConnected2d does not support bias')
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.input_shape = _pair(input_shape)
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        if isinstance(padding, str):
            if padding != 'same':
                raise RuntimeError(f'Unknown padding type {padding}')
            if self.stride != 1:
                raise RuntimeError('"same" padding requires stride = 1')
            self.padding = (self.kernel_size[0] - 1) // 2
        else:
            self.padding = padding
        # Compute output shape from input shape:
        self.output_shape = (
            math.floor((self.input_shape[0] + 2*self.padding - self.kernel_size[0]) / stride + 1),
            math.floor((self.input_shape[1] + 2*self.padding - self.kernel_size[0]) / stride + 1)
        )

        # Construct the weights so we don't need to make any changes to them.
        # Dimensions are (#filters, #channels * kernel size, #outputs).
        # This matches what we'll get when we unfold the input.
        self.weight = torch.nn.Parameter(torch.empty(  # pyright: ignore reportPrivateImportUsage
            out_planes, in_planes*self.kernel_size[0]*self.kernel_size[1],
            self.output_shape[0]*self.output_shape[1]))
        self.reset_parameters()

    def extra_repr(self) -> str:
        return (f'in_planes={self.in_planes} out_planes={self.out_planes}'
                f' input={self.input_shape} kernel={self.kernel_size}'
                f' stride={self.stride} padding={self.padding}'
                f' output={self.output_shape}')

    def reset_parameters(self) -> None:
        # Use leakyrelu to match what PyTorch does.
        init.kaiming_uniform_lcn(self.weight, nonlinearity='leaky_relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Unfold (im2col).
        x_unf = F.unfold(
            x, self.kernel_size, padding=self.padding, stride=self.stride)
        # Insert a dimension so we broadcast the filter dimension properly,
        # then do an element-wise multiplication (weights already laid out).
        # Finally, sum out the channel/kernel dimension.
        y_unf = (x_unf.unsqueeze(1) * self.weight).sum(2)
        # View as batch x filters x output shape.
        return y_unf.view(y_unf.size(0), y_unf.size(1), *self.output_shape)

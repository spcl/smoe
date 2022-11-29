"""Biases shared and applied over different dimensions."""

from typing import Optional

import torch



class Bias(torch.nn.Module):
    """Bias, optionally shared and applied over requested dimensions."""

    def __init__(self, in_planes: int, input_shape: tuple[int, int],
                 channel_bias: bool = True,
                 height_bias: bool = False,
                 width_bias: bool = False) -> None:
        """Set up and initialize biases.

        in_planes: Number of input channels.

        input_shape: Spatial dimensions of input.

        channel_bias: Whether to apply a bias channel-wise (this is the
        standard way to apply a bias).

        height_bias: Whether to apply a bias height-wise (column-wise).

        width_bias: Whether to apply a bias width-wise (row-wise).

        """
        super().__init__()

        self.channel_bias: Optional[torch.Tensor] = None
        self.height_bias: Optional[torch.Tensor] = None
        self.width_bias: Optional[torch.Tensor] = None

        if channel_bias:
            self.channel_bias = torch.nn.Parameter(torch.empty(  # pyright: ignore reportPrivateImportUsage
                1, in_planes, 1, 1))
        if height_bias:
            self.height_bias = torch.nn.Parameter(torch.empty(  # pyright: ignore reportPrivateImportusage
                1, 1, input_shape[0], 1))
        if width_bias:
            self.width_bias = torch.nn.Parameter(torch.empty(  # pyright: ignore reportprivateimportusage
                1, 1, 1, input_shape[1]))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.channel_bias is not None:
            torch.nn.init.zeros_(self.channel_bias)
        if self.height_bias is not None:
            torch.nn.init.zeros_(self.height_bias)
        if self.width_bias is not None:
            torch.nn.init.zeros_(self.width_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channel_bias is not None:
            x = x + self.channel_bias
        if self.height_bias is not None:
            x = x + self.height_bias
        if self.width_bias is not None:
            x = x + self.width_bias
        return x

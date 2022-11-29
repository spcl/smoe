"""Simple MLP networks."""

from typing import Any

import functools
import operator
import collections

import torch


class SimpleMLP(torch.nn.Module):
    """Implement a simple MLP.

    This always has a final layer projecting to the data shape.

    """

    def __init__(self, args: Any) -> None:
        super().__init__()
        if args.residuals:
            raise ValueError('Residual connections not supported')
        # Number of neurons needed to represent the data.
        data_neurons = functools.reduce(operator.mul, args.data_shape, 1)
        target_neurons = functools.reduce(operator.mul, args.data_target_shape, 1)
        layers = []
        layers.append(('flatten', torch.nn.Flatten()))
        in_neurons = data_neurons
        for i, num_neurons in enumerate(args.num_neurons):
            layers.append((f'linear_{i}', torch.nn.Linear(in_neurons, num_neurons,
                                          bias=args.with_bias)))
            layers.append((f'act_{i}', args.act()))
            in_neurons = num_neurons
        # Add final layer to project to the right size.
        layers.append(('linear_end', torch.nn.Linear(in_neurons, target_neurons,
                                      bias=args.with_bias)))
        layers.append(('unflatten', torch.nn.Unflatten(1, args.data_target_shape)))
        self.layers = torch.nn.Sequential(collections.OrderedDict(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

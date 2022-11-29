"""Simple multi-layer CNN model."""

from typing import Any

import collections
import functools
import operator

import torch

from . utils import get_layers, get_network_block


class SimpleCNN(torch.nn.Module):
    """Implement a simple multi-layer CNN."""

    def __init__(self, args: Any) -> None:
        super().__init__()
        # If the --layers argument is provided, this assumes that all
        # layer types are given. If it is not provided, all layer
        # types are assumed to be the same.
        # The exception is if --classification-layer is provided,
        # in which case the last layer will be a linear layer.
        layer_types = get_layers(args)
        layers = []
        in_planes = args.data_shape[0]
        if len(layer_types) - 1 != len(args.conv_filters):
            raise RuntimeError(f'Number of layer types {len(layer_types) - 1}'
                               ' does not match number of conv filters'
                               f' {len(args.conv_filters)}')
        # Last layer is handled specially.
        for i, (block, out_planes) in enumerate(zip(layer_types[:-1], args.conv_filters)):
            layers += get_network_block(
                f'{args.model}_{i}', block, args.act,
                in_planes, out_planes, args.data_shape[1:],
                args.with_bias, args.with_height_bias, args.with_width_bias,
                not args.no_batchnorm, args.residuals)
            in_planes = out_planes
        if layer_types[-1] is None:
            # Linear classification layer.
            if args.classification_gap:
                layers.append((
                    f'{args.model}_end_gap',
                    torch.nn.AdaptiveAvgPool2d(1)))
                in_neurons = in_planes
            else:
                in_neurons = functools.reduce(
                    operator.mul, (in_planes,) + args.data_shape[1:])
            layers.append((
                f'{args.model}_end_flatten',
                torch.nn.Flatten()))
            layers.append((
                f'{args.model}_end_linear',
                torch.nn.Linear(in_neurons, args.data_target_shape[0], bias=False)))
        else:
            out_planes = args.data_target_shape[0]
            layers += get_network_block(
                f'{args.model}_end', layer_types[-1], None,
                in_planes, out_planes, args.data_shape[1:],
                args.with_bias, args.with_height_bias, args.with_width_bias,
                False, args.residuals)
        self.layers = torch.nn.Sequential(collections.OrderedDict(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

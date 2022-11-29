"""Utilities for constructing networks."""

from typing import Any, Union

import functools
import collections

import torch

from . residual import ResidualBlock
from . lcn import LocallyConnected2d
from . condconv import CondConv2d
from . coordconv import CoordConv2d
from . convlcn import ConvLCN2d
from . bias import Bias


def get_layer_type_strs(args: Any) -> list[str]:
    """Return a list of layer type names.

    If the final layer is to be a classification layer, the list will
    include "clas" for that layer.

    """
    layer_type_names = args.layers
    if args.classification_layer:
        layer_type_names.append('clas')
    elif not args.layers:
        layer_type_names = [args.model]
    return layer_type_names

def get_layers(args: Any) -> list[Any]:
    """Return a list of layer classes.

    If the final layer is to be a classification layer, the list will
    include None for that layer.

    """
    layer_types = args.layers
    if not layer_types:
        layer_types = [args.model] * len(args.conv_filters)
    layers = [get_conv_block(layer, args) for layer in layer_types]
    if args.classification_layer:
        layers.append(None)
    elif not args.layers:
        layers.append(get_conv_block(args.model, args))
    return layers


def get_conv_block(name: str, args: Any) -> Any:
    """
    Return the layer class for a convolution with the given name.

    If needed, this will wrap the block to pass extra arguments.

    """
    if name == 'conv':
        return torch.nn.Conv2d
    if name == 'lcn':
        # Wrap this so we pass the input shape.
        # TODO: Generalize so we can handle stride/etc.
        return functools.partial(LocallyConnected2d,
                                 input_shape=args.data_shape[1:])
    if name == 'condconv':
        return CondConv2d
    if name == 'coordconv':
        return CoordConv2d
    if name == 'convlcn':
        return functools.partial(ConvLCN2d,
                                 input_shape=args.data_shape[1:])
    raise ValueError(f'Unknown convolution block {name}')


def get_network_block(base_name: str, block: Any, act: Any,
                      in_planes: int, out_planes: int,
                      data_shape: tuple[int, ...],
                      bias: bool, height_bias: bool, width_bias: bool,
                      batchnorm: bool, residual: bool) -> list[Any]:
    layers = []
    layers.append((
        base_name,
        block(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)))
    if bias or height_bias or width_bias:
        layers.append((
            base_name + '_bias',
            Bias(out_planes, data_shape, bias, height_bias, width_bias)))
    if batchnorm:
        layers.append((
            base_name + '_bn',
            torch.nn.BatchNorm2d(out_planes)))
    if residual:
        resize_block = None
        if in_planes != out_planes:
            resize_block = get_channel_resize_block(
                in_planes, out_planes, batchnorm, base_name)
        layers = [(
            base_name + '_block',
            ResidualBlock(layers, resize_block=resize_block))]
    if act is not None:
        layers.append((
            base_name + '_act',
            act()))
    return layers


def get_channel_resize_block(in_planes: int, out_planes: int,
                             batchnorm: bool, i: Union[int, str]):
    """Return a block that resizes in the channel dimension."""
    layers = []
    layers.append((f'conv_resize_{i}',
                   torch.nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                   stride=1, padding=0, bias=False)))
    if batchnorm:
        layers.append((f'resize_bn_{i}', torch.nn.BatchNorm2d(out_planes)))
    return torch.nn.Sequential(collections.OrderedDict(layers))

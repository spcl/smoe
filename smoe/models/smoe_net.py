"""Simple network with SMoE layers."""

from typing import Any

import functools
import dataclasses

import torch

from .. data.mask import load_region_mask
from . utils import get_conv_block, get_channel_resize_block
from . residual import ResidualBlock
from . smoe_gates import (SpatialConvGate2d, SpatialLinearGate2d,
                          SpatialLCNGate2d, SpatialFixedOneOutGate2d,
                          SpatialCoordConvGate2d, SpatialCoordConvMultiGate2d,
                          SpatialLatentTensorGate2d)
from . bias import Bias
from . smoe_config import SpatialMoEConfig
from . smoe import GatedSpatialMoE2d


class GatedSMoENet(torch.nn.Module):
    """Simple SMoE network."""

    def __init__(self, args: Any) -> None:
        super().__init__()
        if len(args.conv_filters) != len(args.num_experts):
            raise ValueError('Must give same number of filters and experts')
        # Construct config for all SMoE layers.
        base_smoe_config = SpatialMoEConfig(
            kernel_size=args.expert_kernel_size,
            padding='same',
            gate_block=self.get_gate_type(args),
            gate_act=args.gate_act,
            gate_kernel_size=args.gate_kernel_size,
            norm_weighted=args.norm_weighted,
            noise=args.noise,
            noise_std_scale=1.0,
            unweighted=args.unweighted_smoe,
            absval_routing=args.absval_routing,
            smooth_gate_error=args.smooth_gate_error,
            dampen_expert_error=args.dampen_expert_error,
            dampen_expert_error_factor=args.dampen_expert_error_factor,
            routing_error_quantile=args.routing_error_quantile,
            block_gate_grad=args.rc_loss,
            save_error_signal=args.rc_loss,
            importance_weight=args.importance_loss_weight,
            load_weight=args.load_loss_weight,
            spatial_agreement_weight=args.spatial_agreement_loss_weight,
            grad_scaler=args._grad_scaler)
        if args.gate_add_mask:
            if not args.mask:
                raise ValueError('Must give --mask with --gate-add-mask')
            base_smoe_config.gate_mask = load_region_mask(args.mask, expand=False)
        if args.smoe_expert_block:
            base_smoe_config.expert_block = get_conv_block(
                args.smoe_expert_block, args)
        layers = []
        needs_bias = (args.with_bias
                      or args.with_height_bias
                      or args.with_width_bias)
        in_planes = args.data_shape[0]
        for i, (num_filters, num_experts) in enumerate(zip(
                args.conv_filters, args.num_experts)):
            smoe_config = dataclasses.replace(
                base_smoe_config, in_planes=in_planes, out_planes=num_filters,
                num_experts=num_experts)
            layer_block = []
            layer_block.append((f'smoe_{i}', GatedSpatialMoE2d(smoe_config)))
            if needs_bias:
                layer_block.append((f'smoe_{i}_bias', Bias(
                    num_filters, args.data_shape[1:],
                    args.with_bias, args.with_height_bias,
                    args.with_width_bias)))
            if not args.no_batchnorm:
                layer_block.append((f'bn_{i}', torch.nn.BatchNorm2d(num_filters)))
            if args.residuals:
                resize_block = None
                if in_planes != num_filters:
                    resize_block = get_channel_resize_block(
                        in_planes, num_filters, not args.no_batchnorm, i)
                layers.append((f'smoe_block_{i}',
                               ResidualBlock(
                                   layer_block, resize_block=resize_block)))
            else:
                layers += layer_block
            layers.append((f'act_{i}', args.act()))
            in_planes = num_filters
        num_filters = args.data_target_shape[0]
        smoe_config = dataclasses.replace(
            base_smoe_config, in_planes=in_planes, out_planes=num_filters,
            num_experts=args.last_layer_experts)
        end_layer = []
        end_layer.append(('smoe_end', GatedSpatialMoE2d(smoe_config)))
        if needs_bias:
            end_layer.append(('smoe_end_bias', Bias(
                num_filters, args.data_shape[1:],
                args.with_bias, args.with_height_bias,
                args.with_width_bias)))
        if args.residuals:
            resize_block = None
            if in_planes != num_filters:
                resize_block = get_channel_resize_block(
                        in_planes, num_filters, not args.no_batchnorm, 'end')
            layers.append(('smoe_block_end',
                           ResidualBlock(
                               end_layer, resize_block=resize_block)))
        else:
            layers += end_layer
        self.layers_dict = torch.nn.ModuleDict(layers)  # pyright: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers_dict.values():
            x = layer(x)
        return x

    @staticmethod
    def get_gate_type(args: Any) -> Any:
        if args.gate_type == 'conv':
            return SpatialConvGate2d
        if args.gate_type == 'conv1':
            return functools.partial(SpatialConvGate2d,
                                     kernel_size=1, padding=0)
        if args.gate_type == 'coordconv':
            return SpatialCoordConvGate2d
        if args.gate_type == 'linear':
            return functools.partial(SpatialLinearGate2d,
                                     input_shape=args.data_shape[1:])
        if args.gate_type == 'lcn':
            return functools.partial(SpatialLCNGate2d,
                                     input_shape=args.data_shape[1:])
        if args.gate_type == 'lcn1':
            return functools.partial(SpatialLCNGate2d,
                                     kernel_size=1, padding=0,
                                     input_shape=args.data_shape[1:])
        if args.gate_type == 'fixed':
            return functools.partial(SpatialFixedOneOutGate2d,
                                     mask_file=args.mask)
        if args.gate_type == 'coordconv-multi':
            return SpatialCoordConvMultiGate2d
        if args.gate_type == 'latent':
            return functools.partial(SpatialLatentTensorGate2d,
                                     input_shape=args.data_shape[1:])
        raise ValueError(f'Unknown gate type {args.gate_type} for GatedSMoENet')

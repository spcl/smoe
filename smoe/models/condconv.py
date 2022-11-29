"""Conditional convolution layer."""

import torch
from torch.nn import functional as F
import timm


class CondConv2d(torch.nn.Module):
    """
    Conditional Convolution.

    Uses the TIMM implementation but wraps in the wrouting function.

    """

    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
                 padding=0, groups=1, dilation=1, bias=False, num_experts=3):
        super().__init__()
        self.cond_conv = timm.models.layers.CondConv2d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride,
            padding=padding, groups=groups, dilation=dilation,
            num_experts=num_experts, bias=bias)
        self.routing_fn = torch.nn.Linear(in_planes, num_experts)

    def forward(self, x):
        pooled_inputs = F.adaptive_avg_pool2d(x, 1).flatten(1)
        routing_weights = torch.sigmoid(self.routing_fn(pooled_inputs))
        x = self.cond_conv(x, routing_weights)
        return x

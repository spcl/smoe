"""Describe SMoE configurations."""

from dataclasses import dataclass
from typing import Any, Optional, Union

import torch


@dataclass
class SpatialMoEConfig:
    # in_planes, out_planes, and num_experts are defaulted so we can
    # easily construct a config.
    """Describes the complete configuration of an SMoE."""
    in_planes: int = -1
    """Number of input channels."""
    out_planes: int = -1
    """Number of output channels (i.e., selected experts)."""
    num_experts: int = -1
    """Total number of experts present."""
    # Expert block configuration. (TODO: Maybe wrap this in the block?)
    expert_block: Any = None
    """Callable or similar returning the SMoE expert block."""
    kernel_size: int = 3
    """Expert kernel size."""
    padding: Union[int, str] = 'same'
    """Expert padding size."""
    # Gate and routing configuration.
    gate_block: Any = None
    """Callable or similar returning the SMoE gate block."""
    gate_act: Any = None
    """Callable or similar returning an activation to be applied to routing weights."""
    gate_kernel_size: int = 3
    """Gate kernel size, if needed."""
    gate_mask: Optional[torch.Tensor] = None
    """Unexpanded region mask to concatenate to the gate input."""
    norm_weighted: bool = False
    """Whether to apply softmax normalization to routing weights."""
    noise: bool = False
    """Whether to add noise to the routing weights."""
    noise_std_scale: float = 1.0
    """Factor to scale the noise standard deviation by."""
    unweighted: bool = False
    """Whether to weight experts by corresponding routing weights."""
    absval_routing: bool = False
    """Whether to use absolute value routing."""
    smooth_gate_error: bool = False
    """Whether to smooth the error signal to the gate."""
    dampen_expert_error: bool = False
    """Whether to damp the error signal to experts erroneously routed to."""
    dampen_expert_error_factor: float = 0.0
    """Factor to dampen expert error by."""
    routing_error_quantile: float = 0.7
    """Quantile determining what qualifies as incorrect routing."""
    block_gate_grad: bool = False
    """Whether to disable the error signal to the gate."""
    save_error_signal: bool = False
    """Whether to save the error signal received by the SMoE."""
    importance_weight: float = 0.0
    """Weighting to give the importance auxiliary loss."""
    load_weight: float = 0.0
    """Weighting to give the load auxiliary loss."""
    spatial_agreement_weight: float = 0.0
    """Weighting to give the spatial agreement auxiliary loss."""
    grad_scaler: Optional[torch.cuda.amp.GradScaler] = None  # pyright: ignore reportPrivateimportusage
    """Gradient scaler that will be used."""

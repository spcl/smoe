"""Spatial mixture-of-experts routing."""

from typing import Any

import torch

from . smoe_config import SpatialMoEConfig
from .. utils.misc import get_incorrect_indices, unscale_quantity


class SMoERouting(torch.autograd.Function):
    """Implement SMoE routing.

    This uses the raw routing weights from an SMoE gate and the output
    from experts (in its full, non-sparse form) and picks the winning
    experts, optionally scaling them by the winning routing weights.

    The routing is done based on the top k values in the routing
    weights at each point.

    This is a bit hard to implement with normal differentiable PyTorch
    ops, so here we manually implement the forward and backward parts
    of this routing.

    This is essentially a straight-through estimator. Gradients are
    sent only to experts through the positions they are selected at.
    The gate, via the routing weights, also only receives gradients
    through such points.

    """

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)  # pyright: ignore reportPrivateImportusage
    def forward(ctx: Any,
                experts: torch.Tensor,
                routing_weights: torch.Tensor,
                smoe_config: SpatialMoEConfig,
                save_module: Any = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Implement SMoE routing.

        experts is the full, non-sparse output from the experts.

        routing_weights is the raw routing output from the SMoE gate.

        smoe_config describes the SMoE layer being implemented.

        save_module is some module in which to save the error signal,
        if any. The 'saved_error_signal' attribute will be set.

        Returns a tensor that has selected only the winning experts
        (i.e., it will have num_outplanes channel dimensions) at each
        point.

        Also returns the routing map and indices, because these may be
        used in analyses or other parts.

        """
        ctx.smoe_config = smoe_config
        ctx.save_module = save_module

        # Note: Sometimes the maximum value is 0, mostly because all
        # other values are negative.
        # Note sure whether this always makes sense.
        # Using softmax normalization should avoid this.
        vals, indices = routing_weights.topk(k=smoe_config.out_planes, dim=1)
        ctx.indices = indices
        ctx.save_for_backward(experts, routing_weights)

        routing_map = torch.zeros_like(routing_weights).scatter_(
            dim=1, index=indices, src=vals)
        ctx.mark_non_differentiable(routing_map)  # Will not need gradients.
        ctx.mark_non_differentiable(indices)
        if smoe_config.unweighted:
            selected = torch.gather(experts, dim=1, index=indices)
        else:
            scaled_experts = experts * routing_map
            selected = torch.gather(scaled_experts, dim=1, index=indices)
        return selected, routing_map, indices

    @staticmethod
    @torch.cuda.amp.custom_bwd  # pyright: ignore reportPrivateImportusage
    def backward(ctx: Any,
                 grad_selected: torch.Tensor,
                 grad_routing_map: torch.Tensor,
                 grad_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        # Note: grad_routing_map and grad_indices zeros, ignore.
        experts, routing_weights = ctx.saved_tensors
        grad_experts = grad_routing_weights = None
        smoe_config: SpatialMoEConfig = ctx.smoe_config

        if ctx.save_module and smoe_config.save_error_signal:
            ctx.save_module.saved_error_signal = grad_selected
            # If we have a weighted SMoE, we need to scale this by the
            # selected experts before they were scaled by the routing
            # weights.
            if not smoe_config.unweighted:
                #selected_routes = torch.gather(routing_weights, dim=1, index=ctx.indices)
                #ctx.save_module.saved_error_signal = grad_selected * selected_routes
                selected_experts = torch.gather(experts, dim=1, index=ctx.indices)
                ctx.save_module.saved_error_signal = grad_selected * selected_experts

        # scattered_grads (and the error signals) are the same size as
        # the raw expert output / routing map:
        # batch x #experts x height x width.
        # Note: grad_selected may not have the same dtype as experts
        # due to it being promoted to float32 in some cases.
        scattered_grads = torch.zeros(experts.size(), dtype=grad_selected.dtype,
                                      device=grad_selected.device).scatter_(
            dim=1, index=ctx.indices, src=grad_selected)

        if ctx.needs_input_grad[0]:
            if smoe_config.unweighted:
                grad_experts = scattered_grads
            else:
                grad_experts = scattered_grads * routing_weights
        if ctx.needs_input_grad[1] and not smoe_config.block_gate_grad:
            if smoe_config.unweighted:
                grad_routing_weights = scattered_grads.clone()
            else:
                grad_routing_weights = scattered_grads * experts

        num_experts = scattered_grads.size(1)
        # Handle gate error smoothing and expert damping.
        if ((smoe_config.smooth_gate_error or smoe_config.dampen_expert_error)
            and num_experts > smoe_config.out_planes  # Avoid dividing by 0.
            and grad_experts is not None):
            if smoe_config.smooth_gate_error:
                raise RuntimeError('Not currently supporting smoothed gate'
                                   ' errors, reimplement if needed')
            if smoe_config.dampen_expert_error:
                # TODO: If we block the gate gradient, we can apply
                # this to the input gradients directly before scatter.
                scale = unscale_quantity(grad_selected.numel() / 2,
                                         smoe_config.grad_scaler)
                incorrect = get_incorrect_indices(
                    grad_selected, smoe_config.routing_error_quantile,
                    scale=scale)
                # Build a tensor to scale the expert gradient by.
                # We will scatter just as above, but with a tensor
                # filled with appropriate scaling factors based on
                # incorrect indices.
                scaling_factors = torch.ones_like(grad_selected)
                scaling_factors[incorrect] = smoe_config.dampen_expert_error_factor
                damping = torch.ones_like(scattered_grads).scatter_(
                    dim=1, index=ctx.indices, src=scaling_factors)
                grad_experts *= damping


        return (grad_experts, grad_routing_weights,
                # None for all the other arguments in forward.
                None, None)

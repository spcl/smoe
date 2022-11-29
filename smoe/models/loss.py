"""Loss functions for training."""

from typing import Sequence, Callable, Any

import torch
import torch.nn.functional as F
from .. data.mask import load_region_mask
from .. utils.distributed import get_cuda_device
from .. utils.misc import get_correct_indices, get_correct_indices_by_tol, unscale_quantity


class MaskScaledLoss(torch.nn.Module):
    """A wrapper that scales a loss function based on a region mask."""

    def __init__(self, mask_path: str,
                 region_scales: Sequence[float],
                 loss_func: Callable) -> None:
        super().__init__()
        mask_unexp = load_region_mask(mask_path, expand=False)
        num_regions = torch.unique(mask_unexp).numel()
        if len(region_scales) != num_regions:
            raise ValueError(f"Must specify scales for {num_regions} regions")
        self.mask = torch.zeros(mask_unexp.size())
        for i in range(num_regions):
            indices = mask_unexp == i
            self.mask[indices] = region_scales[i]
        self.mask = self.mask.to(get_cuda_device())
        self.loss_func = loss_func

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = self.loss_func(x, y, reduction='none')
        scaled_loss = loss * self.mask
        return scaled_loss.mean()


@torch.no_grad()
def construct_routing_labels(correct: torch.Tensor,
                             routing_indices: torch.Tensor,
                             num_experts: int) -> torch.Tensor:
    """Construct labels for gate routing classification.

    This uses a provided correctness tensor to determine whether each
    expert at each point was correct or not. If the expert was
    incorrect, we say that the gate should not have selected it.

    Essentially, this turns into a (multi-label) classification
    problem to decide which experts are the right ones at each point.
    The labels are constructed based on whether the experts were
    correct or not, essentially forming a self-supervised training
    process.

    When an expert is correct, the corresponding label is 1. When one
    is incorrect, the label is 0, and also all unselected experts get
    a label value of 1 / (#unselected experts).

    """
    out_planes = correct.size(1)
    # Number of experts not selected at each point.
    unselected = num_experts - out_planes
    if unselected == 0:
        raise RuntimeError('0 unselected experts not supported')
    # Get the number of wrong selections at each point.
    num_wrong = out_planes - correct.sum(dim=1, keepdim=True)
    # Compute the label weight to give unselected experts.
    unselected_weight = num_wrong / unselected
    # Build the initial labels tensor.
    labels = unselected_weight.repeat(1, num_experts, *([1]*(correct.ndim - 2)))
    # Scatter the correct tensor into the labels tensor.
    # This will set the entries that are correct / incorrect to be
    # 0 / 1, respectively.
    labels.scatter_(dim=1, index=routing_indices, src=correct)
    return labels


def routing_classification_loss(net: Any,
                                output: torch.Tensor,
                                target: torch.Tensor,
                                abstol: float = 1e-5,
                                reltol: float = 0.) -> list[torch.Tensor]:
    """Compute the routing classification losses for each gate layer.

    This relies on the network predictions and true values, so only
    works for the final layer.

    """
    losses = []
    for module in net.modules():
        # Assume this implies it's a gate.
        if hasattr(module, 'routing_weights'):
            correct = get_correct_indices_by_tol(output, target, abstol, reltol)
            labels = construct_routing_labels(
                correct, module.routing_indices,
                module.smoe_config.num_experts)
            loss = F.binary_cross_entropy_with_logits(
                module.routing_weights, labels)
            losses.append(loss)
    return losses


def routing_classification_loss_by_error(net: Any,
                                         scaler: torch.cuda.amp.GradScaler,  # pyright: ignore reportprivateimportusage
                                         quantile: float) -> list[torch.Tensor]:
    """Compute routing classification losses for each gate layer.

    This uses the saved error signal of each layer to determine whether
    routing is correct and then returns the loss for each such layer.

    """
    losses = []
    for module in net.modules():
        # Assume this implies it's a gate.
        if hasattr(module, 'saved_error_signal'):
            correct = get_correct_indices(
                module.saved_error_signal, quantile,
                unscale_quantity(module.saved_error_signal.numel() / 2, scaler)).to(
                    dtype=module.routing_weights.dtype)
            labels = construct_routing_labels(
                correct, module.routing_indices,
                module.smoe_config.num_experts)
            loss = F.binary_cross_entropy_with_logits(
                module.routing_weights, labels)
            losses.append(loss)
    return losses

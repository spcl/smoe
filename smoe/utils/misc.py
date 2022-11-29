"""Miscellaneous utilities that don't nicely fit elsewhere."""

from typing import Union, Optional, overload

import torch


class ReduceLROnPlateauPatch(torch.optim.lr_scheduler.ReduceLROnPlateau):
    """Patched ReduceLROnPlateau scheduler to add get_last_lr."""

    def get_last_lr(self):
        # It thinks there is no self.optimizer.
        return [group['lr'] for group in self.optimizer.param_groups]  # pyright: ignore

    def step(self, *args, **kwargs):
        # This is a patch to prevent erronous warnings.
        super().step(*args, **kwargs)


def count_parameters(net):
    """Return the total number of parameters in net."""
    count = 0
    for params in net.parameters():
        count += params.numel()
    return count


@torch.no_grad()
def get_correct_indices(tensor: torch.Tensor,
                        quantile: float = 0.7,
                        scale: Union[int, float] = 1,
                        eps: float = 1e-5,
                        incorrect: bool = False) -> torch.Tensor:
    """Return a boolean index tensor of where tensor is correct.

    Given an error tensor, this identifies a threshold based on the
    specified quantile and any (absolute) value larger than that, plus
    a small tolerance eps, is incorrect. Everything else is correct.

    tensor should have a leading batch dimension, and mistakes will be
    computed per sample.

    scale is a scaling factor to apply to tensor, for undoing things
    like means or gradient scaling.

    incorrect is whether to actually return the indices of the
    incorrect values (i.e., the opposite of usual).

    """
    # We want to look at the largest magnitude errors.
    tensor = tensor.abs() * scale
    # Compute quantiles. Reshape so we can compute for each batch dim,
    # as quantiles doesn't support multiple dimensions.
    # Force conversion to FP32 because quantile does not support FP16.
    q = tensor.reshape(tensor.size(0), -1).to(dtype=torch.float32).quantile(quantile, dim=1)
    # Reshape so broadcasting works correctly.
    q = q.reshape(-1, *([1]*(tensor.ndim - 1)))
    if incorrect:
        result = tensor > (q + eps)
    else:
        result = tensor <= (q + eps)
    return result


def get_incorrect_indices(tensor: torch.Tensor,
                          quantile: float = 0.7,
                          scale: Union[int, float] = 1,
                          eps: float = 1e-3) -> torch.Tensor:
    """Convenience wrapper for get_correct_indices(..., incorrect=True)."""
    return get_correct_indices(tensor, quantile=quantile, scale=scale, eps=eps,
                               incorrect=True)


@torch.no_grad()
def get_correct_indices_by_tol(output: torch.Tensor,
                               target: torch.Tensor,
                               abstol: float = 1e-5,
                               reltol: float = 1e-3) -> torch.Tensor:
    """Return a boolean index tensor of where output is correct.

    This computes correctness using a tolerance compared to a ground
    truth tensor, unlike get_correct_indices, which uses relative
    error quantiles.

    """
    if reltol == 0:
        correct = (output - target).abs() <= abstol
    else:
        correct = (output - target).abs() <= abstol + reltol*target.abs()
    return correct.to(dtype=target.dtype)


@overload
def unscale_quantity(val: int,
                     scaler: Optional[torch.cuda.amp.GradScaler]  # pyright: ignore reportPrivateimportusage
                     ) -> int: ...
@overload
def unscale_quantity(val: float,
                     scaler: Optional[torch.cuda.amp.GradScaler]  # pyright: ignore reportPrivateimportusage
                     ) -> float: ...
@overload
def unscale_quantity(val: torch.Tensor,
                     scaler: Optional[torch.cuda.amp.GradScaler]  # pyright: ignore reportPrivateimportusage
                     ) -> torch.Tensor: ...
@torch.no_grad()
def unscale_quantity(val: Union[int, float, torch.Tensor],
                     scaler: Optional[torch.cuda.amp.GradScaler]  # pyright: ignore reportPrivateimportusage
                     ) -> Union[int, float, torch.Tensor]:
    """Unscale val based on the scaling factor in scaler, if enabled."""
    if scaler is not None and scaler.is_enabled():
        scale = scaler._get_scale_async()
        if scale is not None:
            return val / scale
    return val

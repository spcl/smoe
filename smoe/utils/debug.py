"""Utilities for debugging."""

import functools
import torch


def debug_tensor_hook(tensor, **kwargs):
    """Add a hook to a tensor for debugging."""
    def _hook(grad, kwdict=None):
        if not torch.all(torch.isfinite(grad)):
            if kwdict and 'name' in kwdict:
                print('Bad gradient in', kwdict['name'])
            else:
                print('Bad gradient')
            nan_locs = torch.isnan(grad).nonzero()
            inf_locs = torch.isinf(grad).nonzero()
            print('NaN locs:')
            print(nan_locs)
            print('Inf locs:')
            print(inf_locs)
            breakpoint()

    if tensor.requires_grad:
        tensor.register_hook(functools.partial(_hook, kwdict=kwargs))

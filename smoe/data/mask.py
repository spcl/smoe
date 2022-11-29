"""Load masks for regions."""

import torch
import numpy as np


def load_region_mask(filename, expand=True):
    """Load a Numpy mask describing data regions.

    The mask should have only spatial dimensions and each value is an
    index giving the region number (starting from 0).

    This supports only a single region per point.

    If expand is True (default), this expands the mask such that there
    is one channel per region, and the mask is a boolean tensor that
    is True at the points where the region is.

    """
    np_mask = torch.from_numpy(np.load(filename))
    if not expand:
        return np_mask
    num_experts = int(np_mask.max().item()) + 1
    mask = torch.zeros((num_experts,) + np_mask.size(), dtype=torch.bool)
    for i in range(num_experts):
        mask[i, np_mask == i] = 1
    return mask

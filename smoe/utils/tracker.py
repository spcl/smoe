"""Utilities to track averages."""

import statistics
import torch
from . distributed import allreduce_tensor


class AverageTracker:
    """Keeps track of the average of a value."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Clear the tracker."""
        self.vals = []

    def update(self, val, n=1):
        """Add n copies of val to the tracker."""
        if n == 1:
            self.vals.append(val)
        else:
            self.vals.extend([val]*n)

    def mean(self):
        """Return the mean."""
        if not self.vals:
            return float('nan')
        return statistics.mean(self.vals)

    def latest(self):
        """Return the latest value."""
        if not self.vals:
            return float('nan')
        return self.vals[-1]

    def save(self, filename):
        """Save data to a file."""
        with open(filename, 'a') as fp:
            fp.write(','.join([str(v) for v in self.vals]) + '\n')


@torch.jit.script  # pyright: ignore reportPrivateImportUsage
def _mean_impl(data, counts):
    """Internal scripted mean implementation."""
    return data.sum(dim=0) / counts.sum()


class AverageTrackerDevice:
    """Keep track of the average of a value or tensor.

    This is optimized for storing the results on device.

    """

    def __init__(self, n, device, allreduce=True, shape=None):
        """Track n total values on device.

        allreduce: Perform an allreduce over scaled values before
        computing mean.

        shape: Track a tensor of this shape rather than a scalar.

        """
        self.n = n
        self.shape = (n,) if shape is None else (n,) + tuple(shape)
        self.device = device
        self.allreduce = allreduce
        self.last_allreduce_count = None
        self.saved_mean = None
        self.reset()

    def reset(self):
        """Clear the tracker."""
        self.data = torch.empty(self.shape, device=self.device)
        self.counts = torch.empty((self.n,) + (1,)*(len(self.shape) - 1),
                                  device='cpu', pin_memory=True)
        self.cur_count = 0
        # For caching results.
        self.last_allreduce_count = None
        self.saved_mean = None

    @torch.no_grad()
    def update(self, val, count=1.0):
        """Add val and associated count to tracker."""
        if self.cur_count == self.n:
            raise RuntimeError('Updating average tracker past end')
        self.data[self.cur_count] = val
        self.counts[self.cur_count] = count
        self.cur_count += 1

    @torch.no_grad()
    def mean(self):
        """Return the mean.

        This will be a device tensor.

        """
        if self.cur_count == 0:
            return float('nan')
        if self.cur_count == self.last_allreduce_count:
            return self.saved_mean
        valid_data = self.data.narrow(0, 0, self.cur_count)
        valid_counts = self.counts.narrow(0, 0, self.cur_count).to(self.device)
        scaled_vals = valid_data * valid_counts
        if self.allreduce:
            scaled_vals = allreduce_tensor(scaled_vals)
        mean = _mean_impl(scaled_vals, valid_counts).tolist()
        self.last_allreduce_count = self.cur_count
        self.saved_mean = mean
        return mean

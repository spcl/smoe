"""Metrics to measure progress, and infrastructure for logging them."""

import functools
import torch
from . distributed import get_cuda_device
from . tracker import AverageTrackerDevice
from ..data.mask import load_region_mask


def dim_tuple(tensor):
    """Return a tuple containing the indices of each dimension of tensor.

    For an n-dimensional tensor, this is (0, 1, ..., n - 1).

    """
    return tuple(range(tensor.ndim))


@torch.no_grad()
def mse_metric(x, y, mask=None, mask_nnz=None):
    """Compute the mean square error between x and y.

    If mask is not None, each mask region is computed separately.

    mask_nnz is the number of non-zeros in each mask channel. This
    should be precomputed to avoid overhead.

    """
    if mask is None:
        return ((x - y)**2).mean()
    if x.size(1) != 1:
        raise RuntimeError('Mask not supported for multiple channels')
    z = ((x - y)**2) * mask
    return z.sum(dim=(0,) + dim_tuple(x)[2:]) / (mask_nnz * x.size(0))


@torch.no_grad()
def mae_metric(x, y, mask=None, mask_nnz=None):
    """Compute the mean absolute error between x and y.

    If mask is not None, each mask region is computed separately.

    mask_nnz is the number of non-zeros in each mask channel. This
    should be precomputed to avoid overhead.

    """
    if mask is None:
        return (x - y).abs().mean()
    if x.size(1) != 1:
        raise RuntimeError('Mask not supported for multiple channels')
    z = (x - y) * mask
    z = z.abs()
    return z.sum(dim=(0,) + dim_tuple(x)[2:]) / (mask_nnz * x.size(0))


@torch.no_grad()
def rmse_metric(x, y, mask=None, mask_nnz=None):
    """Compute the root mean squared error between x and y.

    If mask is not None, each mask region is computed separately.

    mask_nnz is the number of non-zeros in each mask channel. This
    should be precomputed to avoid overhead.

    """
    # Note: We want to return the mean of the RMSE over the samples,
    # which means we need to take that mean after the square root.
    if mask is None:
        return ((x - y)**2).mean(dim=dim_tuple(x)[1:]).sqrt().mean()
    if x.size(1) != 1:
        raise RuntimeError('Mask not supported for multiple channels')
    z = ((x - y)**2) * mask
    z = z.sum(dim=(dim_tuple(x)[2:])) / mask_nnz
    z.sqrt_()
    return z.mean(dim=0)


@torch.no_grad()
def nrmse_metric(x, y, mask=None, mask_nnz=None):
    """Compute the normalized root mean squared error between x and y.

    If mask is not None, each mask region is computed separately.

    mask_nnz is the number of non-zeros in each mask channel. This
    should be precomputed to avoid overhead.

    This will normalize using the range of each sample in y.

    """
    if mask is None:
        dim = dim_tuple(x)[1:]
        rmse = ((x - y)**2).mean(dim=dim).sqrt()
        nrmse = rmse / (y.amax(dim=dim) - y.amin(dim=dim))
        return nrmse.mean()
    if x.size(1) != 1:
        raise RuntimeError('Mask not supported for multiple channels')
    dim = dim_tuple(x)[2:]
    z = ((x - y)**2) * mask
    rmse = (z.sum(dim=dim) / mask_nnz).sqrt()
    nrmse = rmse / (y.amax(dim=dim) - y.amin(dim=dim))
    return nrmse.mean(dim=0)


@torch.no_grad()
def prcntclose_metric(x, y, mask=None, mask_nnz=None, atol=1e-8, rtol=0.01,
                      mask_tol=None):
    """Compute the percent of values that are 'close'.

    This essentially is an element-wise allclose.

    If mask is not None, each mask region is computed separately.

    mask_nnz is the number of non-zeros in each mask channel. This
    should be precomputed to avoid overhead.

    If mask_tol is not None, it will be used to scale rtol at each
    point.

    """
    if mask_tol is not None:
        close = (x - y).abs() <= atol + rtol*mask_tol*y.abs()
    else:
        close = (x - y).abs() <= atol + rtol*y.abs()
    if mask is None:
        return close.sum() / close.numel() * 100
    if x.size(1) != 1:
        raise RuntimeError('Mask not supported for multiple channels')
    close = close * mask
    return close.sum(dim=(0,) + dim_tuple(close)[2:]) / (mask_nnz * x.size(0)) * 100


@torch.no_grad()
def topk_accuracy_metric(x: torch.Tensor, y: torch.Tensor, k: int) -> torch.Tensor:
    """Compute the top-k accuracy of x using y."""
    _, pred = x.topk(k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y.view(1, -1).expand_as(pred))
    correct_k = correct[:k].float().sum()
    correct_k.mul_(100.0 / x.size(0))
    return correct_k


class MetricManager:
    """Manages applying and tracking metrics."""

    def __init__(self, metric_names, criterion, n, prcntclose=None,
                 prcntclose_tol_scale=None, allreduce=True, mask=None):
        """Initialize the metric manager with a list of metrics and mask."""
        metrics_map = {
            'mse': mse_metric,
            'mae': mae_metric,
            'rmse': rmse_metric,
            'nrmse': nrmse_metric,
            'top1': functools.partial(topk_accuracy_metric, k=1),
            'top5': functools.partial(topk_accuracy_metric, k=5)
        }
        if prcntclose_tol_scale and not mask:
            raise ValueError('Must give a mask to use --prcntclose-tol-scale')
        if mask:
            self.mask = load_region_mask(mask).to(get_cuda_device())
            self.mask_nnz = self.mask.sum(dim=dim_tuple(self.mask)[1:])
        else:
            self.mask = None
            self.mask_nnz = None
        if prcntclose_tol_scale:
            # Load mask again, unexpanded.
            mask_unexp = load_region_mask(mask, expand=False)
            num_regions = torch.unique(mask_unexp).numel()
            # Pad scaling if there's not enough elements.
            if len(prcntclose_tol_scale) != num_regions:
                prcntclose_tol_scale += [1.0] * (num_regions - len(prcntclose_tol_scale))
            self.mask_tol = torch.zeros(mask_unexp.size())
            for i in range(num_regions):
                indices = mask_unexp == i
                self.mask_tol[indices] = prcntclose_tol_scale[i]
            self.mask_tol = self.mask_tol.to(get_cuda_device())
        else:
            self.mask_tol = None
        mask_metrics_map = {
            'mask-mse': functools.partial(mse_metric, mask=self.mask,
                                          mask_nnz=self.mask_nnz),
            'mask-mae': functools.partial(mae_metric, mask=self.mask,
                                          mask_nnz=self.mask_nnz),
            'mask-rmse': functools.partial(rmse_metric, mask=self.mask,
                                           mask_nnz=self.mask_nnz),
            'mask-nrmse': functools.partial(nrmse_metric, mask=self.mask,
                                            mask_nnz=self.mask_nnz),
        }
        if prcntclose is None:
            prcntclose = []
        for prcnt in prcntclose:
            metrics_map[f'prcntclose{prcnt*100}'] = functools.partial(
                prcntclose_metric, rtol=prcnt)
            mask_metrics_map[f'mask-prcntclose{prcnt*100}'] = functools.partial(
                prcntclose_metric, mask=self.mask, mask_nnz=self.mask_nnz,
                rtol=prcnt)
            if self.mask_tol is not None:
                metrics_map[f'prcntclose{prcnt*100}s'] = functools.partial(
                    prcntclose_metric, rtol=prcnt, mask_tol=self.mask_tol)
                mask_metrics_map[f'mask-prcntclose{prcnt*100}s'] = functools.partial(
                    prcntclose_metric, mask=self.mask, mask_nnz=self.mask_nnz,
                    rtol=prcnt, mask_tol=self.mask_tol)
        if 'all' in metric_names:
            metric_names = list(metrics_map.keys())
            if self.mask is not None:
                metric_names += list(mask_metrics_map.keys())
        # Add the right prcntclose entries.
        if 'prcntclose' in metric_names and prcntclose:
            metric_names.remove('prcntclose')
            metric_names += [f'prcntclose{prcnt*100}' for prcnt in prcntclose]
            if self.mask_tol is not None:
                metric_names += [f'prcntclose{prcnt*100}s' for prcnt in prcntclose]
        if 'mask-prcntclose' in metric_names and prcntclose:
            metric_names.remove('mask-prcntclose')
            metric_names += [f'mask-prcntclose{prcnt*100}' for prcnt in prcntclose]
            if self.mask_tol is not None:
                metric_names += [f'mask-prcntclose{prcnt*100}s' for prcnt in prcntclose]
        self.metrics = {}
        self.metric_trackers = {}
        for metric in metric_names:
            if metric in metrics_map:
                self.metrics[metric] = metrics_map[metric]
                self.metric_trackers[metric] = AverageTrackerDevice(
                    n, get_cuda_device(), allreduce=allreduce)
            elif metric in mask_metrics_map:
                if self.mask is None:
                    raise ValueError('Must provide mask for masked metrics')
                self.metrics[metric] = mask_metrics_map[metric]
                self.metric_trackers[metric] = AverageTrackerDevice(
                    n, get_cuda_device(), allreduce=allreduce,
                    shape=(self.mask.size(0),))
            else:
                raise ValueError('Unknown metric ' + metric)
        self.metrics['loss'] = criterion
        self.metric_trackers['loss'] = AverageTrackerDevice(
            n, get_cuda_device(), allreduce=allreduce)
        self.metric_vals = {}
        self.reset()

    def reset(self):
        """Clear tracking information for metrics."""
        self.metric_vals = {}
        for tracker in self.metric_trackers.values():
            tracker.reset()

    def compute_metrics(self, output, targets):
        """Compute and save all metrics."""
        self.metric_vals = {
            metric: self.metrics[metric](output, targets)
            for metric in self.metrics
        }

    def update_trackers(self, count=1.0):
        """Update metric trackers."""
        for metric, val in self.metric_vals.items():
            self.metric_trackers[metric].update(val, count)

    def get_metric_means(self):
        """Return a dict with the mean values of each metric."""
        return {
            metric: self.metric_trackers[metric].mean()
            for metric in self.metrics
        }

    def log_metrics(self, log, indent=0, prefix='', metrics=None):
        """Log the mean values of each metric.

        If provided, uses metrics for the values rather than computing
        them.

        """
        start = ' ' * indent + prefix
        metric_vals = metrics
        if not metric_vals:
            metric_vals = {
                metric: tracker.mean()
                for metric, tracker in self.metric_trackers.items()
            }
        for metric, mean in metric_vals.items():
            if isinstance(mean, list):
                mean_fmt = ', '.join(f'{m:.5f}' for m in mean)
                log.log(start + ' ' + metric + ': ' + mean_fmt)
            else:
                log.log(start + f' {metric}: {mean:.5f}')

"""Manage noise tensors."""

import torch


class NormalNoise:
    """Manage a normal noise tensor for application."""

    def __init__(self, mean, std):
        """Initialize with mean and std."""
        self.mean = mean
        self.std = std
        self.noise_tensor = None

    def get_noise(self, x):
        """Get a fresh noise tensor that is the same shape as x."""
        if self.noise_tensor is None or self.noise_tensor.size() != x.size:
            self.noise_tensor = torch.empty_like(x)
        self.noise_tensor.normal_(mean=self.mean, std=self.std)
        return self.noise_tensor

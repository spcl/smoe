"""Heat diffusion dataset."""

from typing import Any

import os.path

import numpy as np
import torch
import torch.utils.data

from . import mask


class HeatDiffusionDataset(torch.utils.data.Dataset):
    """Load heat diffusion dataset."""

    def __init__(self, path: str, args: Any, train: bool = True) -> None:
        # Training/validation determined based on path.
        super().__init__()
        if args.download:
            raise ValueError('HeadDiffusion dataset does not support download')
        if not os.path.exists(path):
            raise ValueError(f'{path} does not exist')
        # Data is num runs x num steps x height x width.
        # Add a dummy "channel" dimension.
        self.data = torch.from_numpy(np.load(path))
        # Normalize data.
        if not args.data_no_norm:
            if args.data_maxmin_norm:
                data_min = self.data.min()
                self.data = (self.data - data_min) / (self.data.max() - data_min)
            else:
                if args.data_mean is None or args.data_std is None:
                    # Compute these and save. Assumes the train dataset
                    # is constructed first.
                    args.data_mean = self.data.mean()
                    args.data_std = self.data.std()
                self.data = (self.data - args.data_mean) / args.data_std
        self.num_context = args.data_context
        if self.num_context == 1:
            self.num_context = 0  # These are the same case.
        if self.num_context == 0:
            self.data.unsqueeze_(2)
        self.num_runs = self.data.size(0)
        self.num_steps = self.data.size(1)
        # Number of available samples, accounting for needing enough
        # context for the first samples and the target.
        self.samples_per_run = self.num_steps - self.num_context - 1
        self.num_samples = self.samples_per_run * self.num_runs

        # Load mask if needed.
        if args.data_mask:
            if not args.mask:
                raise ValueError('Must provide --mask to use --data-mask')
            self.mask = mask.load_region_mask(args.mask, expand=False).unsqueeze(0)
        else:
            self.mask = None

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Get indices, accounting for context/etc.
        run_idx = index // self.samples_per_run
        idx_in_run = index % self.samples_per_run + self.num_context
        if self.num_context:
            sample = self.data[run_idx, idx_in_run - self.num_context:idx_in_run]
            target = self.data[run_idx, idx_in_run + 1].unsqueeze(0)
        else:
            sample = self.data[run_idx, idx_in_run]
            target = self.data[run_idx, idx_in_run + 1]
        if self.mask is not None:
            sample = torch.cat((sample, self.mask), dim=0)
        return sample, target

    def __str__(self) -> str:
        return (f'SynthConv(runs={self.num_runs}, steps={self.num_steps}, '
                f'shape={self.get_shape()} target_shape={self.get_target_shape()})')

    def get_shape(self) -> tuple[int, ...]:
        """Return the shape of the data."""
        if self.num_context:
            channels = self.num_context
        else:
            channels = 1
        if self.mask is not None:
            channels += 1
        return (channels,) + tuple(self.data.size())[-2:]

    def get_target_shape(self) -> tuple[int, ...]:
        """Return the shape of the data target."""
        # Always have one output target channel.
        return (1,) + tuple(self.data.size())[-2:]

    @staticmethod
    def get_train_path(basepath: str) -> str:
        return os.path.join(basepath, 'train.npy')

    @staticmethod
    def get_validation_path(basepath: str) -> str:
        return os.path.join(basepath, 'val.npy')

    @staticmethod
    def get_test_path(basepath: str) -> str:
        return os.path.join(basepath, 'test.npy')

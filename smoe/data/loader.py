"""Construct data loaders for datasets."""

import torch
import torch.utils.data
import torch.utils.data.distributed

from . import heat

def get_dataloaders(args):
    """Return train and validation data loaders."""
    if args.dataset == 'heat':
        dataset_class = heat.HeatDiffusionDataset
    else:
        raise ValueError(f'Unknown dataset {args.dataset}')
    train_dataset = dataset_class(
        dataset_class.get_train_path(args.data_path), args, train=True)
    if args.dist:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.workers,
        sampler=train_sampler, pin_memory=True, drop_last=args.drop_last)
    if not args.no_eval:
        validation_dataset = dataset_class(
            dataset_class.get_validation_path(args.data_path), args, train=False)
        if args.dist:
            validation_sampler = torch.utils.data.distributed.DistributedSampler(
                validation_dataset, shuffle=False)
        else:
            validation_sampler = torch.utils.data.SequentialSampler(
                validation_dataset)
        validation_loader = torch.utils.data.DataLoader(
            validation_dataset, batch_size=args.batch_size,
            num_workers=args.workers, sampler=validation_sampler,
            pin_memory=True, drop_last=args.drop_last)

        test_dataset = dataset_class(
            dataset_class.get_test_path(args.data_path), args, train=False)
        if args.dist:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset, shuffle=False)
        else:
            test_sampler = torch.utils.data.SequentialSampler(
                test_dataset)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size,
            num_workers=args.workers, sampler=test_sampler,
            pin_memory=True, drop_last=args.drop_last)
    else:
        validation_loader = None
        test_loader = None

    # Update the data shape if needed.
    if args.data_shape is None:
        args.data_shape = train_dataset.get_shape()
    if args.data_target_shape is None:
        args.data_target_shape = train_dataset.get_target_shape()

    return train_loader, validation_loader, test_loader

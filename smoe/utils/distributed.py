"""Utilities to support distributed training."""

import os
import os.path
import socket
import time
import torch
import torch.distributed


def get_num_gpus():
    """Number of GPUs on this node."""
    return torch.cuda.device_count()


def get_local_rank(required=False):
    """Get local rank from environment."""
    if 'MV2_COMM_WORLD_LOCAL_RANK' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_LOCAL_RANK'])
    if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    if 'SLURM_LOCALID' in os.environ:
        return int(os.environ['SLURM_LOCALID'])
    if required:
        raise RuntimeError('Could not get local rank')
    return 0


def get_local_size(required=False):
    """Get local size from environment."""
    if 'MV2_COMM_WORLD_LOCAL_SIZE' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_LOCAL_SIZE'])
    if 'OMPI_COMM_WORLD_LOCAL_SIZE' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
    if 'SLURM_NTASKS_PER_NODE' in os.environ:
        return int(os.environ['SLURM_NTASKS_PER_NODE'])
    if required:
        raise RuntimeError('Could not get local size')
    return 1


def get_world_rank(required=False):
    """Get rank in world from environment."""
    if 'MV2_COMM_WORLD_RANK' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_RANK'])
    if 'OMPI_COMM_WORLD_RANK' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_RANK'])
    if 'SLURM_PROCID' in os.environ:
        return int(os.environ['SLURM_PROCID'])
    if required:
        raise RuntimeError('Could not get world rank')
    return 0


def get_world_size(required=False):
    """Get world size from environment."""
    if 'MV2_COMM_WORLD_SIZE' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_SIZE'])
    if 'OMPI_COMM_WORLD_SIZE' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_SIZE'])
    if 'SLURM_NTASKS' in os.environ:
        return int(os.environ['SLURM_NTASKS'])
    if required:
        raise RuntimeError('Could not get world size')
    return 1


def get_cuda_device():
    """Get this rank's CUDA device."""
    return torch.device(f'cuda:{get_local_rank()}')


def get_job_id():
    """Return the resource manager job ID, if any"""
    if 'SLURM_JOBID' in os.environ:
        return os.environ['SLURM_JOBID']
    if 'LSB_JOBID' in os.environ:
        return os.environ['LSB_JOBID']
    return None


def initialize_dist(init_file, rendezvous='file'):
    """Initialize the PyTorch distributed backend.

    This always uses NCCL.

    """
    # For safety, must make sure CUDA is initialized first.
    torch.cuda.init()
    torch.cuda.set_device(get_local_rank())

    init_file = os.path.abspath(init_file)

    if rendezvous == 'tcp':
        init_method = None
        if get_world_rank() == 0:
            # Get an IP and port to use.
            ip = socket.gethostbyname(socket.gethostname())
            s = socket.socket()
            s.bind(('', 0))  # Get a free port provided by the host.
            port = s.getsockname()[1]
            init_method = f'tcp://{ip}:{port}'
            with open(init_file, 'w') as f:
                f.write(init_method)
        else:
            while not os.path.exists(init_file):
                time.sleep(1)
            with open(init_file, 'r') as f:
                init_method = f.read()
    elif rendezvous == 'file':
        init_method = f'file://{init_file}'
    else:
        raise ValueError(f'Unrecognized scheme "{rendezvous}"')
    torch.distributed.init_process_group(
        backend='nccl', init_method=init_method,
        rank=get_world_rank(), world_size=get_world_size())

    torch.distributed.barrier()
    # Attempt to ensure the init file is removed.
    if get_world_rank() == 0 and os.path.exists(init_file):
        os.unlink(init_file)


def allreduce_tensor(t):
    """Allreduce and average tensor t."""
    rt = t.clone().detach()
    torch.distributed.all_reduce(rt)
    rt /= get_world_size()
    return rt

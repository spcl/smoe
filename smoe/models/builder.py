"""Build models based on arguments."""

import functools

import torch

from . import cnn
from . import mlp
from . import smoe_net


MODEL_BUILDERS = {
    'cnn': cnn.SimpleCNN,
    'conv': cnn.SimpleCNN,
    'lcn': cnn.SimpleCNN,
    'condconv': cnn.SimpleCNN,
    'coordconv': cnn.SimpleCNN,
    'convlcn': cnn.SimpleCNN,
    'mlp': mlp.SimpleMLP,
    'smoe': smoe_net.GatedSMoENet
}

ACTIVATIONS = {
    'relu': functools.partial(torch.nn.ReLU, inplace=True),
    'tanh': torch.nn.Tanh,
    'sigmoid': torch.nn.Sigmoid,
    'id': torch.nn.Identity,
}


def get_activation(act_str):
    """Return the activation function corresponding to act_str."""
    act_str = act_str.lower()
    if act_str not in ACTIVATIONS:
        raise ValueError('Unknown activation function ' + act_str)
    return ACTIVATIONS[act_str]


def construct_model(model_name, args):
    """Return the requested model."""
    args.act = get_activation(args.act)
    args.gate_act = get_activation(args.gate_act)

    if model_name not in MODEL_BUILDERS:
        raise ValueError(f'Unknown model name {model_name}')
    return MODEL_BUILDERS[model_name](args)

"""Tools to initialize models."""

from typing import Optional
import math
import torch


def kaiming_uniform_lcn(params: torch.Tensor,
                        nonlinearity: str = 'leaky_relu') -> None:
    """Do a Kaiming uniform initialization for a locally-connected layer."""
    # The built-in torch initializers won't calculate the scaling
    # properly because of the unusual weight shape.
    fan_in = params.size()[1]
    gain = torch.nn.init.calculate_gain(nonlinearity, math.sqrt(5))
    std = gain / math.sqrt(fan_in)
    bound = math.sqrt(3.0) * std
    with torch.no_grad():
        params.uniform_(-bound, bound)


def init_params(init_type: str, net: torch.nn.Module,
                exclude: Optional[list[str]] = None,
                nonlinearity: str = 'relu',
                gate_orthogonal: bool = False) -> None:
    """
    Initialize parameters in network.

    Depends on init_type:
    - pytorch: Does nothing (let's PyTorch's default init stand).
    - default: PyTorch init but with gain fixed.

    exclude: List of names in parameters to exclude.

    nonlinearity: Gain for the parameter initialization.

    Any layer with '_end' in its name will use a linear initialization.

    gate_orthogonal: Apply an orthogonal initialization to gate weights.

    """
    if init_type == 'pytorch':
        return
    if exclude is None:
        exclude = []
    # Pyright seems to think exclude can be None.
    params = filter(lambda x: not any(name in x[0] for name in exclude),  # pyright: ignore reportOptionalIterable
                    net.named_parameters())
    if init_type == 'default':
        for name, param in params:
            if 'bias' in name:
                torch.nn.init.zeros_(param)
            elif 'weight' in name:
                param_nonlin = 'linear' if '_end' in name else nonlinearity
                if gate_orthogonal and 'gate' in name:
                    torch.nn.init.orthogonal_(param)
                elif '.lcn' in name:
                    kaiming_uniform_lcn(param, nonlinearity=param_nonlin)
                else:
                    #print('CURRENTLY FREEZING WEIGHTS')
                    torch.nn.init.kaiming_uniform_(
                        param, nonlinearity=param_nonlin)
                    #t = torch.tensor(
                    #    [[[[0., 0.25, 0.],
                    #       [0.25, 0., 0.25],
                    #       [0., 0.25, 0.]]],
                    #     [[[0., 0.025, 0.],
                    #       [0.025, 0.9, 0.025],
                    #       [0., 0.25, 0.]]],
                    #     [[[0., 0.0025, 0.],
                    #       [0.0025, 0.99, 0.0025],
                    #       [0., 0.0025, 0.]]]]
                    #)
                    #param.data = t.to(device=param.device)
                    #param.requires_grad = False

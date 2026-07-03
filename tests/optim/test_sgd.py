from typing_extensions import *

import itertools

import pytest
import torch

from utils import run_optim
from roundpipe.optim import SGD


@pytest.mark.parametrize(
    "lr, momentum, dampening, weight_decay, nesterov, maximize",
    [
        combo
        for combo in itertools.product(
            [0.01, torch.tensor(0.001)],
            [0.0, 0.9],
            [0.0, 0.5],
            [0.0, 0.01],
            [False, True],
            [False, True],
        )
        # Nesterov momentum requires a positive momentum and zero dampening.
        if not (combo[4] and (combo[1] <= 0 or combo[2] != 0))
    ],
)
def test_SGD(
    lr: Union[float, torch.Tensor],
    momentum: float,
    dampening: float,
    weight_decay: float,
    nesterov: bool,
    maximize: bool,
) -> None:
    run_optim(
        SGD,
        torch.optim.SGD,
        lr=lr,
        momentum=momentum,
        dampening=dampening,
        weight_decay=weight_decay,
        nesterov=nesterov,
        maximize=maximize,
    )

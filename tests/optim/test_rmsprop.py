from typing_extensions import *

import itertools

import pytest
import torch

from utils import run_optim
from roundpipe.optim import RMSprop


@pytest.mark.parametrize(
    "lr, weight_decay, momentum, centered, maximize",
    list(
        itertools.product(
            [0.01, torch.tensor(0.005)],
            [0.0, 0.01],
            [0.0, 0.9],
            [True, False],
            [True, False],
        )
    ),
)
def test_RMSprop(
    lr: Union[float, torch.Tensor],
    weight_decay: float,
    momentum: float,
    centered: bool,
    maximize: bool,
) -> None:
    run_optim(
        RMSprop,
        torch.optim.RMSprop,
        lr=lr,
        alpha=0.99,
        eps=1e-4,
        weight_decay=weight_decay,
        momentum=momentum,
        centered=centered,
        maximize=maximize,
    )

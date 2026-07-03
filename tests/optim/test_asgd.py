from typing_extensions import *

import itertools

import pytest
import torch

from utils import run_optim
from roundpipe.optim import ASGD


@pytest.mark.parametrize(
    "lr, weight_decay, maximize, t0",
    list(
        itertools.product(
            [0.01, torch.tensor(0.005)],
            [0.0, 0.01],
            [True, False],
            [1e6, 5.0],
        )
    ),
)
def test_ASGD(
    lr: Union[float, torch.Tensor],
    weight_decay: float,
    maximize: bool,
    t0: float,
) -> None:
    run_optim(
        ASGD,
        torch.optim.ASGD,
        lr=lr,
        lambd=1e-4,
        alpha=0.75,
        t0=t0,
        weight_decay=weight_decay,
        maximize=maximize,
    )

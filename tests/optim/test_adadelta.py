from typing_extensions import *

import itertools

import pytest
import torch

from utils import run_optim
from roundpipe.optim import Adadelta


@pytest.mark.parametrize(
    "lr, rho, weight_decay, maximize",
    list(
        itertools.product(
            [0.5, torch.tensor(0.3)],
            [0.9],
            [0.0, 0.01],
            [True, False],
        )
    ),
)
def test_Adadelta(
    lr: Union[float, torch.Tensor],
    rho: float,
    weight_decay: float,
    maximize: bool,
) -> None:
    run_optim(
        Adadelta,
        torch.optim.Adadelta,
        lr=lr,
        rho=rho,
        eps=1e-4,
        weight_decay=weight_decay,
        maximize=maximize,
    )

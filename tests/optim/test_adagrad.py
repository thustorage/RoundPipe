from typing_extensions import *

import itertools

import pytest
import torch

from utils import run_optim
from roundpipe.optim import Adagrad


@pytest.mark.parametrize(
    "lr, lr_decay, weight_decay, initial_accumulator_value, maximize",
    list(
        itertools.product(
            [0.1, torch.tensor(0.05)],
            [0.0, 0.05],
            [0.0, 0.01],
            [0.0, 0.1],
            [False, True],
        )
    ),
)
def test_Adagrad(
    lr: Union[float, torch.Tensor],
    lr_decay: float,
    weight_decay: float,
    initial_accumulator_value: float,
    maximize: bool,
) -> None:
    run_optim(
        Adagrad,
        torch.optim.Adagrad,
        lr=lr,
        lr_decay=lr_decay,
        weight_decay=weight_decay,
        initial_accumulator_value=initial_accumulator_value,
        eps=1e-4,
        maximize=maximize,
    )

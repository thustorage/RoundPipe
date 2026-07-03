from typing_extensions import *

import itertools

import pytest
import torch

from utils import run_optim
from roundpipe.optim import Adamax


@pytest.mark.parametrize(
    "lr, weight_decay, maximize",
    list(
        itertools.product(
            [0.002, torch.tensor(0.001)],
            [0.0, 0.01],
            [True, False],
        )
    ),
)
def test_Adamax(
    lr: Union[float, torch.Tensor],
    weight_decay: float,
    maximize: bool,
) -> None:
    run_optim(
        Adamax,
        torch.optim.Adamax,
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-4,
        weight_decay=weight_decay,
        maximize=maximize,
    )

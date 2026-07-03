from typing_extensions import *

import itertools

import pytest
import torch

from utils import run_optim
from roundpipe.optim import RAdam


@pytest.mark.parametrize(
    "lr, weight_decay, decoupled_weight_decay, maximize",
    list(
        itertools.product(
            [0.001, torch.tensor(0.0005)],
            [0.0, 0.01],
            [True, False],
            [True, False],
        )
    ),
)
def test_RAdam(
    lr: Union[float, torch.Tensor],
    weight_decay: float,
    decoupled_weight_decay: bool,
    maximize: bool,
) -> None:
    run_optim(
        RAdam,
        torch.optim.RAdam,
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-4,
        weight_decay=weight_decay,
        decoupled_weight_decay=decoupled_weight_decay,
        maximize=maximize,
    )

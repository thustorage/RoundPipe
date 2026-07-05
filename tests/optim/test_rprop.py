from typing_extensions import *

import itertools

import pytest
import torch

from utils import run_optim
from roundpipe.optim import Rprop


@pytest.mark.parametrize(
    "lr, maximize",
    list(
        itertools.product(
            [0.01, torch.tensor(0.005)],
            [True, False],
        )
    ),
)
def test_Rprop(
    lr: Union[float, torch.Tensor],
    maximize: bool,
) -> None:
    run_optim(
        Rprop,
        torch.optim.Rprop,
        lr=lr,
        etas=(0.5, 1.2),
        step_sizes=(1e-6, 50.0),
        maximize=maximize,
    )

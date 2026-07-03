from typing_extensions import *

import itertools

import pytest
import torch

from utils import run_optim
from roundpipe.optim import AdamW


@pytest.mark.parametrize(
    "lr_betas, weight_decay, amsgrad, maximize",
    list(
        itertools.product(
            [
                (0.01, 0.9, 0.999),
                (torch.tensor(0.001), torch.tensor(0.8), torch.tensor(0.95)),
            ],
            [0.0, 0.01],
            [True, False],
            [True, False],
        )
    ),
)
def test_AdamW(
    lr_betas: Union[
        Tuple[float, float, float], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ],
    weight_decay: float,
    amsgrad: bool,
    maximize: bool,
) -> None:
    lr, *betas = lr_betas
    run_optim(
        AdamW,
        torch.optim.AdamW,
        lr=lr,
        betas=tuple(betas),
        eps=1e-4,
        weight_decay=weight_decay,
        amsgrad=amsgrad,
        maximize=maximize,
    )

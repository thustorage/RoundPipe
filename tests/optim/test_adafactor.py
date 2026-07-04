from typing_extensions import *

import itertools
import random

import pytest
import torch

from roundpipe.optim import Adafactor


def run_optim_2d(
    cls: type[torch.optim.Optimizer],
    ref_cls: type[torch.optim.Optimizer],
    *optim_args: Any,
    atol: float = 1e-6,
    steps: int = 10,
    **optim_kwargs: Any,
) -> None:
    """Compare a RoundPipe optimizer against its PyTorch reference.

    Builds a motley collection of float32 and complex64 tensors, runs both
    optimizers for several steps under identical gradients, and asserts the
    parameters stay close.

    Args:
        cls: The RoundPipe optimizer class under test.
        ref_cls: The PyTorch reference optimizer class.
        *optim_args: Positional arguments forwarded to both optimizers.
        atol: Absolute tolerance for `torch.allclose`.
        steps: Number of optimization steps to run.
        **optim_kwargs: Keyword arguments forwarded to both optimizers.
    """
    params: List[torch.Tensor] = []
    for n in range(1, 11):
        params.append(
            torch.randn(n, n + 1, n + 2, dtype=torch.float32, requires_grad=True)
        )
    for n in range(1, 1025):
        params.append(torch.randn(n, dtype=torch.float32, requires_grad=True))
    for i in range(1, 34):
        for j in range(1, 34):
            params.append(torch.randn(i, j, dtype=torch.float32, requires_grad=True))
    random.shuffle(params)
    ref_params = [p.clone().detach().requires_grad_(True) for p in params]

    optimizer = cls(params, *optim_args, **optim_kwargs)
    ref_optimizer = ref_cls(ref_params, *optim_args, **optim_kwargs)
    for _ in range(steps):
        for i, (p, ref_p) in enumerate(zip(params, ref_params)):
            p.grad = torch.randn_like(p)
            if i % 100 == 0:
                p.grad *= 1e-20  # to test small gradients
            ref_p.grad = p.grad.clone()
        optimizer.step()
        ref_optimizer.step()
        for p, ref_p in zip(params, ref_params):
            assert torch.allclose(p, ref_p, atol=atol)


@pytest.mark.parametrize(
    "lr, beta2_decay, eps, d, weight_decay, maximize",
    list(
        itertools.product(
            [1e-2, torch.tensor(1.0)],
            [-0.8, -0.5],
            [(None, 1e-3), (1e-8, 1e-3)],
            [1.0, 2.0],
            [0.0, 0.01],
            [False, True],
        )
    ),
)
def test_Adafactor(
    lr: Union[float, torch.Tensor],
    beta2_decay: float,
    eps: Tuple[Optional[float], float],
    d: float,
    weight_decay: float,
    maximize: bool,
) -> None:
    run_optim_2d(
        Adafactor,
        torch.optim.Adafactor,
        lr=lr,
        beta2_decay=beta2_decay,
        eps=eps,
        d=d,
        weight_decay=weight_decay,
        maximize=maximize,
        atol=1e-4,
    )

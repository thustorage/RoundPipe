from typing_extensions import *

import random

import torch


def run_optim(
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
    for n in range(1025):
        params.append(torch.randn(n, dtype=torch.float32, requires_grad=True))
    for n in range(1025):
        params.append(torch.randn(n, dtype=torch.complex64, requires_grad=True))
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

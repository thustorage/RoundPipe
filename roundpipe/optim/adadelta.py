from typing_extensions import *

import warnings

import torch
from torch.optim.optimizer import Optimizer, ParamsT, _get_scalar_dtype

from .optim_builder import get_optim_function, load_optim_function


class Adadelta(Optimizer):
    """Implements Adadelta algorithm with fp32 stepping on CPU."""

    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, torch.Tensor] = 1.0,
        rho: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        foreach: Optional[bool] = None,
        *,
        capturable: bool = False,
        maximize: bool = False,
        differentiable: bool = False,
    ):
        """
        Args:
            params: iterable of parameters or named_parameters to optimize or
                iterable of dicts defining parameter groups. When using named_parameters,
                all parameters in all groups should be named
            lr: coefficient that scales delta before it is applied to the parameters
            rho: coefficient used for computing a running average of squared gradients
            eps: term added to the denominator to improve numerical stability
            weight_decay: weight decay (L2 penalty)
            maximize: maximize the objective with respect to the params, instead of minimizing
            foreach: Compatible placeholder for PyTorch's Adadelta optimizer.
            capturable: Compatible placeholder for PyTorch's Adadelta optimizer.
            differentiable: Compatible placeholder for PyTorch's Adadelta optimizer.
        """
        load_optim_function("adadelta")
        assert capturable is False, "capturable=True is not supported."
        assert differentiable is False, "differentiable=True is not supported."
        if foreach is not None:
            warnings.warn(
                "The foreach option is not supported and will be ignored.", UserWarning
            )

        if isinstance(lr, torch.Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")

        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= rho <= 1.0:
            raise ValueError(f"Invalid rho value: {rho}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            rho=rho,
            eps=eps,
            weight_decay=weight_decay,
            maximize=maximize,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state: Dict[str, Any]):
        """Sets the state of the optimizer.
        This method is used when loading a saved optimizer state.
        It ensures that all necessary keys are present in the state dictionary

        Args:
            state: The state dictionary to set.
        """
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                    step_val = float(p_state["step"])
                    p_state["step"] = torch.tensor(step_val, dtype=_get_scalar_dtype())

    def _init_group(
        self,
        group: Dict[str, Any],
        params_with_grad: List[torch.Tensor],
        grads: List[torch.Tensor],
        square_avgs: List[torch.Tensor],
        acc_deltas: List[torch.Tensor],
        state_steps: List[torch.Tensor],
    ):
        """Initializes the state for each parameter group.
        Results are stored in the provided lists inplace.

        Args:
            group: The parameter group to initialize.
            params_with_grad: List to store parameters with gradients.
            grads: List to store gradients of the parameters.
            square_avgs: List to store running averages of squared gradients.
            acc_deltas: List to store running averages of squared parameter updates.
            state_steps: List to store the step count for each parameter.
        """
        for p in group["params"]:
            if p.grad is not None:
                if p.grad.is_sparse:
                    raise RuntimeError("Adadelta does not support sparse gradients")
                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype())
                    # Running average of squared gradient values
                    state["square_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Running average of squared parameter update values
                    state["acc_delta"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                square_avgs.append(state["square_avg"])
                acc_deltas.append(state["acc_delta"])
                state_steps.append(state["step"])

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], Any]] = None) -> Any:
        """Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            The loss value if a closure is provided, otherwise None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: List[torch.Tensor] = []
            grads: List[torch.Tensor] = []
            square_avgs: List[torch.Tensor] = []
            acc_deltas: List[torch.Tensor] = []
            state_steps: List[torch.Tensor] = []

            self._init_group(
                group, params_with_grad, grads, square_avgs, acc_deltas, state_steps
            )

            adadelta(
                params_with_grad,
                grads,
                square_avgs,
                acc_deltas,
                state_steps,
                lr=group["lr"],
                rho=group["rho"],
                eps=group["eps"],
                weight_decay=group["weight_decay"],
                maximize=group["maximize"],
            )

        return loss


def adadelta(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    square_avgs: List[torch.Tensor],
    acc_deltas: List[torch.Tensor],
    state_steps: List[torch.Tensor],
    capturable: bool = False,
    foreach: Optional[bool] = None,
    differentiable: bool = False,
    has_complex: bool = False,
    *,
    lr: Union[float, torch.Tensor],
    rho: float,
    eps: float,
    weight_decay: float,
    maximize: bool,
):
    """Functional API that performs Adadelta algorithm computation.

    See `roundpipe.optim.Adadelta` for details.
    """
    assert not capturable, "capturable=True is not supported."
    assert not differentiable, "differentiable=True is not supported."
    if foreach is not None:
        warnings.warn(
            "The foreach option is not supported and will be ignored.", UserWarning
        )

    lr, rho, eps, weight_decay = (
        float(lr),
        float(rho),
        float(eps),
        float(weight_decay),
    )
    for tensor_list in (params, grads, square_avgs, acc_deltas):
        for i, t in enumerate(tensor_list):
            if torch.is_complex(t):
                tensor_list[i] = t = torch.view_as_real(t)
            assert t.is_cpu, "RoundPipe Adadelta only supports CPU tensors."
            assert (
                t.dtype is torch.float32
            ), "RoundPipe Adadelta only supports float32 tensors."
            assert t.is_contiguous(), "All tensors must be contiguous."

    adadelta_kernel = get_optim_function("adadelta")
    adadelta_kernel(
        params,
        grads,
        square_avgs,
        acc_deltas,
        state_steps,
        lr,
        rho,
        eps,
        weight_decay,
        maximize,
    )

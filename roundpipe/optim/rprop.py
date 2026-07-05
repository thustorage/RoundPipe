from typing_extensions import *

import warnings

import torch
from torch.optim.optimizer import (
    Optimizer,
    ParamsT,
    _get_scalar_dtype,
)

from .optim_builder import get_optim_function, load_optim_function


class Rprop(Optimizer):
    """Implements the resilient backpropagation algorithm with fp32 stepping on CPU."""

    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, torch.Tensor] = 1e-2,
        etas: Tuple[float, float] = (0.5, 1.2),
        step_sizes: Tuple[float, float] = (1e-6, 50),
        *,
        capturable: bool = False,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        differentiable: bool = False,
    ):
        """
        Args:
            params: iterable of parameters or named_parameters to optimize or
                iterable of dicts defining parameter groups. When using named_parameters,
                all parameters in all groups should be named
            lr: learning rate.
            etas: pair of (etaminus, etaplus), that are multiplicative increase and
                decrease factors
            step_sizes: a pair of minimal and maximal allowed step sizes
            maximize: maximize the objective with respect to the params, instead of minimizing
            capturable: Compatible placeholder for PyTorch's Rprop optimizer.
            foreach: Compatible placeholder for PyTorch's Rprop optimizer.
            differentiable: Compatible placeholder for PyTorch's Rprop optimizer.
        """
        load_optim_function("rprop")
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
        if not 0.0 < etas[0] < 1.0 < etas[1]:
            raise ValueError(f"Invalid eta values: {etas[0]}, {etas[1]}")

        defaults = dict(
            lr=lr,
            etas=etas,
            step_sizes=step_sizes,
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
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        prevs: List[torch.Tensor],
        step_sizes: List[torch.Tensor],
        state_steps: List[torch.Tensor],
    ):
        """Initializes the state for each parameter group.
        Results are stored in the provided lists inplace.

        Args:
            group: The parameter group to initialize.
            params: List to store parameters with gradients.
            grads: List to store gradients of the parameters.
            prevs: List to store the previous gradients of the parameters.
            step_sizes: List to store the per-element step sizes.
            state_steps: List to store the step count for each parameter.
        """
        for p in group["params"]:
            if p.grad is not None:
                if p.grad.is_sparse:
                    raise RuntimeError("Rprop does not support sparse gradients")
                params.append(p)
                grad = p.grad
                grads.append(grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype())
                    state["prev"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    if p.dtype.is_complex:
                        # Complex numbers are treated as two independent reals, so the
                        # step size must be non-zero for the imaginary part too.
                        state["step_size"] = torch.full_like(
                            grad, complex(group["lr"], group["lr"])
                        )
                    else:
                        state["step_size"] = torch.full_like(grad, float(group["lr"]))

                prevs.append(state["prev"])
                step_sizes.append(state["step_size"])
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
            params: List[torch.Tensor] = []
            grads: List[torch.Tensor] = []
            prevs: List[torch.Tensor] = []
            step_sizes: List[torch.Tensor] = []
            state_steps: List[torch.Tensor] = []

            etaminus, etaplus = group["etas"]
            step_size_min, step_size_max = group["step_sizes"]

            self._init_group(group, params, grads, prevs, step_sizes, state_steps)

            rprop(
                params,
                grads,
                prevs,
                step_sizes,
                state_steps,
                step_size_min=step_size_min,
                step_size_max=step_size_max,
                etaminus=etaminus,
                etaplus=etaplus,
                maximize=group["maximize"],
            )

        return loss


def rprop(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    prevs: List[torch.Tensor],
    step_sizes: List[torch.Tensor],
    state_steps: List[torch.Tensor],
    foreach: Optional[bool] = None,
    capturable: bool = False,
    differentiable: bool = False,
    has_complex: bool = False,
    *,
    step_size_min: float,
    step_size_max: float,
    etaminus: float,
    etaplus: float,
    maximize: bool,
):
    """Functional API that performs Rprop algorithm computation.

    See `roundpipe.optim.Rprop` for details.
    """
    assert not capturable, "capturable=True is not supported."
    assert not differentiable, "differentiable=True is not supported."
    if foreach is not None:
        warnings.warn(
            "The foreach option is not supported and will be ignored.", UserWarning
        )

    step_size_min = float(step_size_min)
    step_size_max = float(step_size_max)
    etaminus = float(etaminus)
    etaplus = float(etaplus)

    # View complex tensors as real and validate. Buffers stored in state stay complex;
    # the real view shares storage, so in-place kernel updates propagate back.
    for tensor_list in (params, grads, prevs, step_sizes):
        for i, t in enumerate(tensor_list):
            if torch.is_complex(t):
                tensor_list[i] = t = torch.view_as_real(t)
            assert t.is_cpu, "RoundPipe Rprop only supports CPU tensors."
            assert (
                t.dtype is torch.float32
            ), "RoundPipe Rprop only supports float32 tensors."
            assert t.is_contiguous(), "All tensors must be contiguous."

    rprop_kernel = get_optim_function("rprop")
    rprop_kernel(
        params,
        grads,
        prevs,
        step_sizes,
        state_steps,
        etaminus,
        etaplus,
        step_size_min,
        step_size_max,
        maximize,
    )

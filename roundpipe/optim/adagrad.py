from typing_extensions import *

import warnings

import torch
from torch.optim.optimizer import Optimizer, ParamsT, _get_scalar_dtype

from .optim_builder import get_optim_function, load_optim_function


class Adagrad(Optimizer):
    """Implements Adagrad algorithm with fp32 stepping on CPU."""

    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, torch.Tensor] = 1e-2,
        lr_decay: float = 0.0,
        weight_decay: float = 0.0,
        initial_accumulator_value: float = 0.0,
        eps: float = 1e-10,
        foreach: Optional[bool] = None,
        *,
        maximize: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):
        """
        Args:
            params: iterable of parameters or named_parameters to optimize or
                iterable of dicts defining parameter groups. When using named_parameters,
                all parameters in all groups should be named
            lr: learning rate.
            lr_decay: learning rate decay.
            weight_decay: weight decay (L2 penalty).
            initial_accumulator_value: initial value of the sum of squares of gradients.
            eps: term added to the denominator to improve numerical stability.
            maximize: maximize the objective with respect to the params, instead of minimizing
            foreach: Compatible placeholder for PyTorch's Adagrad optimizer.
            differentiable: Compatible placeholder for PyTorch's Adagrad optimizer.
            fused: Compatible placeholder for PyTorch's Adagrad optimizer.
        """
        load_optim_function("adagrad")
        assert differentiable is False, "differentiable=True is not supported."
        if fused is not None:
            warnings.warn(
                "The fused option is not supported and will be ignored.", UserWarning
            )
        if foreach is not None:
            warnings.warn(
                "The foreach option is not supported and will be ignored.", UserWarning
            )

        if isinstance(lr, torch.Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")

        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= lr_decay:
            raise ValueError(f"Invalid lr_decay value: {lr_decay}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= initial_accumulator_value:
            raise ValueError(
                f"Invalid initial_accumulator_value value: {initial_accumulator_value}"
            )
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")

        defaults = dict(
            lr=lr,
            lr_decay=lr_decay,
            eps=eps,
            weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value,
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

    def share_memory(self) -> None:
        """Calls tensor.share_memory_() on the state sum tensors."""
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["sum"].share_memory_()

    def _init_group(
        self,
        group: Dict[str, Any],
        params_with_grad: List[torch.Tensor],
        grads: List[torch.Tensor],
        state_sums: List[torch.Tensor],
        state_steps: List[torch.Tensor],
    ) -> bool:
        """Initializes the state for each parameter group.
        Results are stored in the provided lists inplace.

        Args:
            group: The parameter group to initialize.
            params_with_grad: List to store parameters with gradients.
            grads: List to store gradients of the parameters.
            state_sums: List to store the running sum of squared gradients.
            state_steps: List to store the step count for each parameter.

        Returns:
            Whether any of the parameters are complex.
        """
        has_complex = False
        for p in group["params"]:
            if p.grad is not None:
                if p.grad.is_sparse:
                    raise RuntimeError("Adagrad does not support sparse gradients")
                has_complex |= torch.is_complex(p)
                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype())
                    initial_accumulator_value = group["initial_accumulator_value"]
                    init_value = (
                        complex(initial_accumulator_value, initial_accumulator_value)
                        if torch.is_complex(p)
                        else initial_accumulator_value
                    )
                    # The running sum starts at initial_accumulator_value, not zeros.
                    state["sum"] = torch.full_like(
                        p, init_value, memory_format=torch.preserve_format
                    )

                state_sums.append(state["sum"])
                state_steps.append(state["step"])

        return has_complex

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
            state_sums: List[torch.Tensor] = []
            state_steps: List[torch.Tensor] = []

            has_complex = self._init_group(
                group, params_with_grad, grads, state_sums, state_steps
            )

            adagrad(
                params_with_grad,
                grads,
                state_sums,
                state_steps,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                lr_decay=group["lr_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                has_complex=has_complex,
            )

        return loss


def adagrad(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    state_sums: List[torch.Tensor],
    state_steps: List[torch.Tensor],
    fused: Optional[bool] = None,
    grad_scale: Optional[torch.Tensor] = None,
    found_inf: Optional[torch.Tensor] = None,
    has_sparse_grad: bool = False,
    foreach: Optional[bool] = None,
    differentiable: bool = False,
    has_complex: bool = False,
    *,
    lr: Union[float, torch.Tensor],
    weight_decay: float,
    lr_decay: float,
    eps: float,
    maximize: bool,
):
    """Functional API that performs Adagrad algorithm computation.

    See `roundpipe.optim.Adagrad` for details.
    """
    assert not differentiable, "differentiable=True is not supported."
    if fused is not None:
        warnings.warn(
            "The fused option is not supported and will be ignored.", UserWarning
        )
    if foreach is not None:
        warnings.warn(
            "The foreach option is not supported and will be ignored.", UserWarning
        )
    assert (
        grad_scale is None and found_inf is None
    ), "integrated grad scaling is not supported."
    assert not has_sparse_grad, "sparse gradients are not supported."

    lr = float(lr)
    weight_decay = float(weight_decay)
    lr_decay = float(lr_decay)
    eps = float(eps)

    # View complex tensors as real and validate. Copy the lists so we do not replace
    # the entries the caller reads back into self.state (the real views share storage
    # with the complex state tensors, so kernel writes still land in state).
    kernel_params = list(params)
    kernel_grads = list(grads)
    kernel_state_sums = list(state_sums)
    for tensor_list in (kernel_params, kernel_grads, kernel_state_sums):
        for i, t in enumerate(tensor_list):
            if torch.is_complex(t):
                tensor_list[i] = t = torch.view_as_real(t)
            assert t.is_cpu, "RoundPipe Adagrad only supports CPU tensors."
            assert (
                t.dtype is torch.float32
            ), "RoundPipe Adagrad only supports float32 tensors."
            assert t.is_contiguous(), "All tensors must be contiguous."

    adagrad_kernel = get_optim_function("adagrad")
    adagrad_kernel(
        kernel_params,
        kernel_grads,
        kernel_state_sums,
        state_steps,
        lr,
        lr_decay,
        eps,
        weight_decay,
        maximize,
    )

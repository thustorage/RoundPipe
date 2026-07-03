from typing_extensions import *

import warnings

import torch
from torch.optim.optimizer import Optimizer, ParamsT, _get_scalar_dtype

from .optim_builder import get_optim_function, load_optim_function


class Adamax(Optimizer):
    """Implements Adamax algorithm (a variant of Adam based on infinity norm) with
    fp32 stepping on CPU."""

    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, torch.Tensor] = 2e-3,
        betas: Tuple[Union[float, torch.Tensor], Union[float, torch.Tensor]] = (
            0.9,
            0.999,
        ),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        foreach: Optional[bool] = None,
        *,
        maximize: bool = False,
        differentiable: bool = False,
        capturable: bool = False,
    ):
        """
        Args:
            params: iterable of parameters or named_parameters to optimize or
                iterable of dicts defining parameter groups. When using named_parameters,
                all parameters in all groups should be named
            lr: learning rate.
            betas: coefficients used for computing running averages of gradient and its
                infinity norm
            eps: term added to the denominator to improve numerical stability
            weight_decay: weight decay (L2 penalty)
            maximize: maximize the objective with respect to the params, instead of minimizing
            foreach: Compatible placeholder for PyTorch's Adamax optimizer.
            differentiable: Compatible placeholder for PyTorch's Adamax optimizer.
            capturable: Compatible placeholder for PyTorch's Adamax optimizer.
        """
        load_optim_function("adamax")
        assert capturable is False, "capturable=True is not supported."
        assert differentiable is False, "differentiable=True is not supported."
        if foreach is not None:
            warnings.warn(
                "The foreach option is not supported and will be ignored.", UserWarning
            )

        if isinstance(lr, torch.Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if isinstance(betas[0], torch.Tensor) and betas[0].numel() != 1:
            raise ValueError("Tensor betas[0] must be 1-element")
        if isinstance(betas[1], torch.Tensor) and betas[1].numel() != 1:
            raise ValueError("Tensor betas[1] must be 1-element")

        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
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
        exp_avgs: List[torch.Tensor],
        exp_infs: List[torch.Tensor],
        state_steps: List[torch.Tensor],
    ):
        """Initializes the state for each parameter group.
        Results are stored in the provided lists inplace.

        Args:
            group: The parameter group to initialize.
            params_with_grad: List to store parameters with gradients.
            grads: List to store gradients of the parameters.
            exp_avgs: List to store exponential moving averages of gradients.
            exp_infs: List to store exponentially weighted infinity norms.
            state_steps: List to store the step count for each parameter.
        """
        for p in group["params"]:
            if p.grad is not None:
                if p.grad.is_sparse:
                    raise RuntimeError("Adamax does not support sparse gradients")
                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype())
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponentially weighted infinity norm
                    state["exp_inf"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avgs.append(state["exp_avg"])
                exp_infs.append(state["exp_inf"])
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
            exp_avgs: List[torch.Tensor] = []
            exp_infs: List[torch.Tensor] = []
            state_steps: List[torch.Tensor] = []
            beta1, beta2 = group["betas"]

            self._init_group(
                group, params_with_grad, grads, exp_avgs, exp_infs, state_steps
            )

            adamax(
                params_with_grad,
                grads,
                exp_avgs,
                exp_infs,
                state_steps,
                eps=group["eps"],
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                maximize=group["maximize"],
            )

        return loss


def adamax(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    exp_avgs: List[torch.Tensor],
    exp_infs: List[torch.Tensor],
    state_steps: List[torch.Tensor],
    foreach: Optional[bool] = None,
    capturable: bool = False,
    differentiable: bool = False,
    has_complex: bool = False,
    *,
    eps: float,
    beta1: Union[torch.Tensor, float],
    beta2: Union[torch.Tensor, float],
    lr: Union[float, torch.Tensor],
    weight_decay: float,
    maximize: bool,
):
    """Functional API that performs Adamax algorithm computation.

    See `roundpipe.optim.Adamax` for details.
    """
    assert not capturable, "capturable=True is not supported."
    assert not differentiable, "differentiable=True is not supported."
    if foreach is not None:
        warnings.warn(
            "The foreach option is not supported and will be ignored.", UserWarning
        )

    lr, beta1, beta2, eps, weight_decay = (
        float(lr),
        float(beta1),
        float(beta2),
        float(eps),
        float(weight_decay),
    )
    for tensor_list in (params, grads, exp_avgs, exp_infs):
        for i, t in enumerate(tensor_list):
            if torch.is_complex(t):
                tensor_list[i] = t = torch.view_as_real(t)
            assert t.is_cpu, "RoundPipe Adamax only supports CPU tensors."
            assert (
                t.dtype is torch.float32
            ), "RoundPipe Adamax only supports float32 tensors."
            assert t.is_contiguous(), "All tensors must be contiguous."

    adamax_kernel = get_optim_function("adamax")
    adamax_kernel(
        params,
        grads,
        exp_avgs,
        exp_infs,
        state_steps,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        maximize,
    )

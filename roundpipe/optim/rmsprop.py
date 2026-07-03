from typing_extensions import *

import warnings

import torch
from torch.optim.optimizer import Optimizer, ParamsT, _get_scalar_dtype

from .optim_builder import get_optim_function, load_optim_function


class RMSprop(Optimizer):
    """Implements RMSprop algorithm with fp32 stepping on CPU."""

    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, torch.Tensor] = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        centered: bool = False,
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
            alpha: smoothing constant.
            eps: term added to the denominator to improve numerical stability
            weight_decay: weight decay (L2 penalty)
            momentum: momentum factor
            centered: if True, compute the centered RMSprop, the gradient is
                normalized by an estimation of its variance
            maximize: maximize the objective with respect to the params, instead of minimizing
            capturable: Compatible placeholder for PyTorch's RMSprop optimizer.
            foreach: Compatible placeholder for PyTorch's RMSprop optimizer.
            differentiable: Compatible placeholder for PyTorch's RMSprop optimizer.
        """
        load_optim_function("rmsprop")
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
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= momentum:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= alpha:
            raise ValueError(f"Invalid alpha value: {alpha}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            alpha=alpha,
            eps=eps,
            centered=centered,
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
            group.setdefault("momentum", 0)
            group.setdefault("centered", False)
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
        grad_avgs: List[torch.Tensor],
        momentum_buffer_list: List[torch.Tensor],
        state_steps: List[torch.Tensor],
    ):
        """Initializes the state for each parameter group.
        Results are stored in the provided lists inplace.

        Args:
            group: The parameter group to initialize.
            params_with_grad: List to store parameters with gradients.
            grads: List to store gradients of the parameters.
            square_avgs: List to store running averages of squared gradients.
            grad_avgs: List to store running averages of gradients (centered only).
            momentum_buffer_list: List to store momentum buffers (momentum only).
            state_steps: List to store the step count for each parameter.
        """
        for p in group["params"]:
            if p.grad is not None:
                if p.grad.is_sparse:
                    raise RuntimeError("RMSprop does not support sparse gradients")
                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype())
                    state["square_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    if group["momentum"] > 0:
                        state["momentum_buffer"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                    if group["centered"]:
                        state["grad_avg"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                square_avgs.append(state["square_avg"])
                state_steps.append(state["step"])
                if group["momentum"] > 0:
                    momentum_buffer_list.append(state["momentum_buffer"])
                if group["centered"]:
                    grad_avgs.append(state["grad_avg"])

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
            grad_avgs: List[torch.Tensor] = []
            momentum_buffer_list: List[torch.Tensor] = []
            state_steps: List[torch.Tensor] = []

            self._init_group(
                group,
                params_with_grad,
                grads,
                square_avgs,
                grad_avgs,
                momentum_buffer_list,
                state_steps,
            )

            rmsprop(
                params_with_grad,
                grads,
                square_avgs,
                grad_avgs,
                momentum_buffer_list,
                state_steps,
                lr=group["lr"],
                alpha=group["alpha"],
                eps=group["eps"],
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                centered=group["centered"],
                maximize=group["maximize"],
            )

        return loss


def rmsprop(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    square_avgs: List[torch.Tensor],
    grad_avgs: List[torch.Tensor],
    momentum_buffer_list: List[torch.Tensor],
    state_steps: List[torch.Tensor],
    foreach: Optional[bool] = None,
    capturable: bool = False,
    differentiable: bool = False,
    has_complex: bool = False,
    *,
    lr: Union[float, torch.Tensor],
    alpha: float,
    eps: float,
    weight_decay: float,
    momentum: float,
    centered: bool,
    maximize: bool,
):
    """Functional API that performs RMSprop algorithm computation.

    See `roundpipe.optim.RMSprop` for details.
    """
    assert not capturable, "capturable=True is not supported."
    assert not differentiable, "differentiable=True is not supported."
    if foreach is not None:
        warnings.warn(
            "The foreach option is not supported and will be ignored.", UserWarning
        )

    lr = float(lr)
    alpha = float(alpha)
    eps = float(eps)
    weight_decay = float(weight_decay)
    momentum = float(momentum)

    # View complex tensors as real and validate. Buffers stored in state stay complex;
    # the real view shares storage, so in-place kernel updates propagate back.
    for tensor_list in (
        params,
        grads,
        square_avgs,
        grad_avgs,
        momentum_buffer_list,
    ):
        for i, t in enumerate(tensor_list):
            if torch.is_complex(t):
                tensor_list[i] = t = torch.view_as_real(t)
            assert t.is_cpu, "RoundPipe RMSprop only supports CPU tensors."
            assert (
                t.dtype is torch.float32
            ), "RoundPipe RMSprop only supports float32 tensors."
            assert t.is_contiguous(), "All tensors must be contiguous."

    rmsprop_kernel = get_optim_function("rmsprop")
    rmsprop_kernel(
        params,
        grads,
        square_avgs,
        grad_avgs,
        momentum_buffer_list,
        state_steps,
        lr,
        alpha,
        eps,
        weight_decay,
        momentum,
        centered,
        maximize,
    )

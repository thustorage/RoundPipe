from typing_extensions import *

import warnings

import torch
from torch.optim.optimizer import Optimizer, ParamsT

from .optim_builder import get_optim_function, load_optim_function


class SGD(Optimizer):
    """Implements stochastic gradient descent (optionally with momentum) with fp32 stepping on CPU."""

    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, torch.Tensor] = 1e-3,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):
        """
        Args:
            params: iterable of parameters or named_parameters to optimize or
                iterable of dicts defining parameter groups. When using named_parameters,
                all parameters in all groups should be named
            lr: learning rate.
            momentum: momentum factor
            dampening: dampening for momentum
            weight_decay: weight decay (L2 penalty)
            nesterov: enables Nesterov momentum. Only applicable when momentum is non-zero.
            maximize: maximize the objective with respect to the params, instead of minimizing
            foreach: Compatible placeholder for PyTorch's SGD optimizer.
            differentiable: Compatible placeholder for PyTorch's SGD optimizer.
            fused: Compatible placeholder for PyTorch's SGD optimizer.
        """
        load_optim_function("sgd")
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
        if not 0.0 <= momentum:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
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
            group.setdefault("nesterov", False)
            group.setdefault("maximize", False)

    def _init_group(
        self,
        group: Dict[str, Any],
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        momentum_buffer_list: List[Optional[torch.Tensor]],
    ):
        """Initializes the state for each parameter group.
        Results are stored in the provided lists inplace.

        Args:
            group: The parameter group to initialize.
            params: List to store parameters with gradients.
            grads: List to store gradients of the parameters.
            momentum_buffer_list: List to store momentum buffers of the parameters.
                Entries are None until the first step populates them (only when
                momentum is non-zero).
        """
        for p in group["params"]:
            if p.grad is not None:
                if p.grad.is_sparse:
                    raise RuntimeError("SGD does not support sparse gradients")
                params.append(p)
                grads.append(p.grad)

                if group["momentum"] != 0:
                    state = self.state[p]
                    momentum_buffer_list.append(state.get("momentum_buffer"))

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
            momentum_buffer_list: List[Optional[torch.Tensor]] = []

            self._init_group(group, params, grads, momentum_buffer_list)

            sgd(
                params,
                grads,
                momentum_buffer_list,
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                lr=group["lr"],
                dampening=group["dampening"],
                nesterov=group["nesterov"],
                maximize=group["maximize"],
            )

            if group["momentum"] != 0:
                # update momentum_buffers in state
                for p, momentum_buffer in zip(params, momentum_buffer_list):
                    state = self.state[p]
                    state["momentum_buffer"] = momentum_buffer

        return loss


def sgd(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    momentum_buffer_list: List[Optional[torch.Tensor]],
    has_sparse_grad: bool = False,
    foreach: Optional[bool] = None,
    fused: Optional[bool] = None,
    grad_scale: Optional[torch.Tensor] = None,
    found_inf: Optional[torch.Tensor] = None,
    *,
    weight_decay: float,
    momentum: float,
    lr: Union[float, torch.Tensor],
    dampening: float,
    nesterov: bool,
    maximize: bool,
):
    """Functional API that performs SGD algorithm computation.

    See `roundpipe.optim.SGD` for details.
    """
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
    momentum = float(momentum)
    dampening = float(dampening)
    weight_decay = float(weight_decay)

    # Materialize momentum buffers. PyTorch keeps them as None until the first step,
    # then initializes them to the (weight-decayed) gradient without dampening. We
    # allocate zeros and flag the first step so the kernel reproduces buf = grad.
    is_first_step: List[int] = [0] * len(params)
    momentum_buffers: List[torch.Tensor] = []
    if momentum != 0:
        for i, p in enumerate(params):
            buf = momentum_buffer_list[i]
            if buf is None:
                buf = torch.zeros_like(p, memory_format=torch.preserve_format)
                momentum_buffer_list[i] = buf
                is_first_step[i] = 1
            momentum_buffers.append(buf)

    # View complex tensors as real and validate. Copy the params/grads lists so we do
    # not replace the entries the caller uses to write momentum buffers back to state.
    kernel_params = list(params)
    kernel_grads = list(grads)
    for tensor_list in (kernel_params, kernel_grads, momentum_buffers):
        for i, t in enumerate(tensor_list):
            if torch.is_complex(t):
                tensor_list[i] = t = torch.view_as_real(t)
            assert t.is_cpu, "RoundPipe SGD only supports CPU tensors."
            assert (
                t.dtype is torch.float32
            ), "RoundPipe SGD only supports float32 tensors."
            assert t.is_contiguous(), "All tensors must be contiguous."

    sgd_kernel = get_optim_function("sgd")
    sgd_kernel(
        kernel_params,
        kernel_grads,
        momentum_buffers,
        is_first_step,
        lr,
        momentum,
        dampening,
        weight_decay,
        nesterov,
        maximize,
    )

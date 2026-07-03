from typing_extensions import *

import warnings

import torch
from torch.optim.optimizer import Optimizer, ParamsT, _get_scalar_dtype

from .optim_builder import get_optim_function, load_optim_function


class ASGD(Optimizer):
    """Implements Averaged Stochastic Gradient Descent with fp32 stepping on CPU."""

    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, torch.Tensor] = 1e-2,
        lambd: float = 1e-4,
        alpha: float = 0.75,
        t0: float = 1e6,
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
            lambd: decay term
            alpha: power for eta update
            t0: point at which to start averaging
            weight_decay: weight decay (L2 penalty)
            maximize: maximize the objective with respect to the params, instead of minimizing
            foreach: Compatible placeholder for PyTorch's ASGD optimizer.
            differentiable: Compatible placeholder for PyTorch's ASGD optimizer.
            capturable: Compatible placeholder for PyTorch's ASGD optimizer.
        """
        load_optim_function("asgd")
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
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            lambd=lambd,
            alpha=alpha,
            t0=t0,
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
                if len(p_state) != 0:
                    if not torch.is_tensor(p_state["step"]):
                        step_val = float(p_state["step"])
                        p_state["step"] = torch.tensor(
                            step_val, dtype=_get_scalar_dtype()
                        )
                    if not torch.is_tensor(p_state["eta"]):
                        p_state["eta"] = torch.tensor(
                            p_state["eta"], dtype=_get_scalar_dtype()
                        )
                    if not torch.is_tensor(p_state["mu"]):
                        p_state["mu"] = torch.tensor(
                            p_state["mu"], dtype=_get_scalar_dtype()
                        )

    def _init_group(
        self,
        group: Dict[str, Any],
        params_with_grad: List[torch.Tensor],
        grads: List[torch.Tensor],
        mus: List[torch.Tensor],
        axs: List[torch.Tensor],
        etas: List[torch.Tensor],
        state_steps: List[torch.Tensor],
    ):
        """Initializes the state for each parameter group.
        Results are stored in the provided lists inplace.

        Args:
            group: The parameter group to initialize.
            params_with_grad: List to store parameters with gradients.
            grads: List to store gradients of the parameters.
            mus: List to store the per-param averaging factor mu.
            axs: List to store the averaged parameter buffers.
            etas: List to store the per-param step size eta.
            state_steps: List to store the step count for each parameter.
        """
        for p in group["params"]:
            if p.grad is not None:
                if p.grad.is_sparse:
                    raise RuntimeError("ASGD does not support sparse gradients")
                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype())
                    # eta and mu are per-param scalar state (eta starts at lr, mu at 1).
                    state["eta"] = torch.tensor(
                        float(group["lr"]), dtype=_get_scalar_dtype()
                    )
                    state["mu"] = torch.tensor(1.0, dtype=_get_scalar_dtype())
                    # Averaged parameter buffer.
                    state["ax"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                mus.append(state["mu"])
                axs.append(state["ax"])
                etas.append(state["eta"])
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
            mus: List[torch.Tensor] = []
            axs: List[torch.Tensor] = []
            etas: List[torch.Tensor] = []
            state_steps: List[torch.Tensor] = []

            self._init_group(
                group, params_with_grad, grads, mus, axs, etas, state_steps
            )

            asgd(
                params_with_grad,
                grads,
                axs,
                mus,
                etas,
                state_steps,
                lambd=group["lambd"],
                lr=group["lr"],
                t0=group["t0"],
                alpha=group["alpha"],
                weight_decay=group["weight_decay"],
                maximize=group["maximize"],
            )

        return loss


def asgd(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    axs: List[torch.Tensor],
    mus: List[torch.Tensor],
    etas: List[torch.Tensor],
    state_steps: List[torch.Tensor],
    foreach: Optional[bool] = None,
    maximize: bool = False,
    differentiable: bool = False,
    capturable: bool = False,
    has_complex: bool = False,
    *,
    lambd: float,
    lr: Union[float, torch.Tensor],
    t0: float,
    alpha: float,
    weight_decay: float,
):
    """Functional API that performs ASGD algorithm computation.

    See `roundpipe.optim.ASGD` for details.
    """
    assert not capturable, "capturable=True is not supported."
    assert not differentiable, "differentiable=True is not supported."
    if foreach is not None:
        warnings.warn(
            "The foreach option is not supported and will be ignored.", UserWarning
        )

    lr = float(lr)
    lambd = float(lambd)
    alpha = float(alpha)
    t0 = float(t0)
    weight_decay = float(weight_decay)

    # View complex tensors as real and validate. Copy the lists so we do not replace
    # entries the caller reads back into state (buffers stay complex; the kernel gets
    # the real view, sharing storage).
    kernel_params = list(params)
    kernel_grads = list(grads)
    kernel_axs = list(axs)
    for tensor_list in (kernel_params, kernel_grads, kernel_axs):
        for i, t in enumerate(tensor_list):
            if torch.is_complex(t):
                tensor_list[i] = t = torch.view_as_real(t)
            assert t.is_cpu, "RoundPipe ASGD only supports CPU tensors."
            assert (
                t.dtype is torch.float32
            ), "RoundPipe ASGD only supports float32 tensors."
            assert t.is_contiguous(), "All tensors must be contiguous."

    asgd_kernel = get_optim_function("asgd")
    asgd_kernel(
        kernel_params,
        kernel_grads,
        kernel_axs,
        mus,
        etas,
        state_steps,
        lambd,
        lr,
        t0,
        alpha,
        weight_decay,
        maximize,
    )

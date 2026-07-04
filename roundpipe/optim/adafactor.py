from typing_extensions import *

import warnings

import torch
from torch.optim.optimizer import Optimizer, ParamsT, _get_scalar_dtype

from .optim_builder import get_optim_function, load_optim_function


class Adafactor(Optimizer):
    """Implements Adafactor algorithm with fp32 stepping on CPU."""

    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, torch.Tensor] = 1e-2,
        beta2_decay: float = -0.8,
        eps: Tuple[Optional[float], float] = (None, 1e-3),
        d: float = 1.0,
        weight_decay: float = 0.0,
        *,
        foreach: Optional[bool] = None,
        maximize: bool = False,
    ):
        """
        Args:
            params: iterable of parameters or named_parameters to optimize or
                iterable of dicts defining parameter groups. When using named_parameters,
                all parameters in all groups should be named
            lr: learning rate. Unlike other optimizers, Adafactor uses it only to apply
                weight decay and as the maximum value for the relative step size rho_t.
            beta2_decay: the decay rate of beta2.
            eps: (epsilon1, epsilon2). epsilon1 stabilizes the update when the squared
                gradient becomes small; when None it defaults to the float32 machine
                epsilon. epsilon2 avoids too small a weight update under parameter scaling.
            d: the clipping threshold, used to avoid larger-than-desired updates.
            weight_decay: weight decay coefficient
            maximize: maximize the objective with respect to the params, instead of minimizing
            foreach: Compatible placeholder for PyTorch's Adafactor optimizer.
        """
        load_optim_function("adafactor")
        if foreach is not None:
            warnings.warn(
                "The foreach option is not supported and will be ignored.", UserWarning
            )

        if isinstance(lr, torch.Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Learning rate should be >= 0 but is: {lr}")
        if not 0.0 >= beta2_decay:
            raise ValueError(f"beta2_decay should be <= 0 but is: {beta2_decay}")
        if eps[0] is not None and not 0.0 <= eps[0]:
            raise ValueError(f"epsilon1 should be >= 0 but is: {eps[0]}")
        if not 0.0 <= eps[1]:
            raise ValueError(f"epsilon2 should be >= 0 but is: {eps[1]}")
        if not 1.0 <= d:
            raise ValueError(f"Clipping threshold d should be >= 1 but is: {d}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"weight_decay should be >= 0 but is: {weight_decay}")

        defaults = dict(
            lr=lr,
            beta2_decay=beta2_decay,
            eps=eps,
            d=d,
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
        row_vars: List[Optional[torch.Tensor]],
        col_vars: List[Optional[torch.Tensor]],
        variances: List[Optional[torch.Tensor]],
        state_steps: List[torch.Tensor],
    ):
        """Initializes the state for each parameter group.
        Results are stored in the provided lists inplace.

        Args:
            group: The parameter group to initialize.
            params_with_grad: List to store parameters with gradients.
            grads: List to store gradients of the parameters.
            row_vars: List to store the row factor of the second moment (ndim > 1).
            col_vars: List to store the column factor of the second moment (ndim > 1).
            variances: List to store the full second moment (ndim == 1).
            state_steps: List to store the step count for each parameter.
        """
        for p in group["params"]:
            if p.grad is None:
                continue
            if torch.is_complex(p):
                raise RuntimeError("Adafactor does not support complex parameters")
            if p.grad.is_sparse:
                raise RuntimeError("Adafactor does not support sparse gradients")

            params_with_grad.append(p)
            grads.append(p.grad)

            state = self.state[p]
            # Lazy state initialization
            if len(state) == 0:
                state["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype())

                if p.grad.dim() > 1:
                    row_shape = list(p.grad.shape)
                    row_shape[-1] = 1
                    state["row_var"] = p.grad.new_zeros(row_shape)
                    col_shape = list(p.grad.shape)
                    col_shape[-2] = 1
                    state["col_var"] = p.grad.new_zeros(col_shape)
                else:
                    state["variance"] = torch.zeros_like(
                        p.grad, memory_format=torch.preserve_format
                    )

            row_vars.append(state.get("row_var", None))
            col_vars.append(state.get("col_var", None))
            variances.append(state.get("variance", None))
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
            row_vars: List[Optional[torch.Tensor]] = []
            col_vars: List[Optional[torch.Tensor]] = []
            variances: List[Optional[torch.Tensor]] = []
            state_steps: List[torch.Tensor] = []
            eps1, eps2 = group["eps"]

            self._init_group(
                group,
                params_with_grad,
                grads,
                row_vars,
                col_vars,
                variances,
                state_steps,
            )

            adafactor(
                params_with_grad,
                grads,
                row_vars,
                col_vars,
                variances,
                state_steps,
                d=group["d"],
                lr=group["lr"],
                beta2_decay=group["beta2_decay"],
                weight_decay=group["weight_decay"],
                eps1=eps1,
                eps2=eps2,
                maximize=group["maximize"],
            )

        return loss


def adafactor(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    row_vars: List[Optional[torch.Tensor]],
    col_vars: List[Optional[torch.Tensor]],
    variances: List[Optional[torch.Tensor]],
    state_steps: List[torch.Tensor],
    foreach: Optional[bool] = None,
    grad_scale: Optional[torch.Tensor] = None,
    found_inf: Optional[torch.Tensor] = None,
    has_complex: bool = False,
    *,
    d: float,
    lr: Union[float, torch.Tensor],
    beta2_decay: float,
    weight_decay: float,
    eps1: Optional[float],
    eps2: float,
    maximize: bool,
):
    """Functional API that performs Adafactor algorithm computation.

    See `roundpipe.optim.Adafactor` for details.
    """
    if foreach is not None:
        warnings.warn(
            "The foreach option is not supported and will be ignored.", UserWarning
        )
    assert (
        grad_scale is None and found_inf is None
    ), "integrated grad scaling is not supported."

    lr = float(lr)
    if eps1 is None:
        eps1 = torch.finfo(torch.float32).eps

    # The kernel takes one flat parameter list (like the other optimizers) and
    # splits the factored (dim > 1) and full-variance (dim == 1) cases by shape.
    # row_vars/col_vars hold the factors for matrices and variances the full moment
    # for vectors; the unused slots get a shared empty placeholder (pybind cannot
    # forward Python None) that the kernel never dereferences.
    placeholder = torch.empty(0)
    row_factors: List[torch.Tensor] = []
    col_factors: List[torch.Tensor] = []
    full_variances: List[torch.Tensor] = []
    for i, p in enumerate(params):
        if p.dim() > 1:
            row_var, col_var = row_vars[i], col_vars[i]
            assert (
                row_var is not None and col_var is not None
            ), "Row and column factors should be defined for factored parameters."
            row_factors.append(row_var)
            col_factors.append(col_var)
            full_variances.append(placeholder)
        else:
            variance = variances[i]
            assert (
                variance is not None
            ), "Variance should be defined for non-factored parameters."
            row_factors.append(placeholder)
            col_factors.append(placeholder)
            full_variances.append(variance)

    for tensor_list in (params, grads, row_factors, col_factors, full_variances):
        for t in tensor_list:
            assert t.is_cpu, "RoundPipe Adafactor only supports CPU tensors."
            assert (
                t.dtype is torch.float32
            ), "RoundPipe Adafactor only supports float32 tensors."
            assert t.is_contiguous(), "All tensors must be contiguous."

    adafactor_kernel = get_optim_function("adafactor")
    adafactor_kernel(
        params,
        grads,
        row_factors,
        col_factors,
        full_variances,
        state_steps,
        lr,
        beta2_decay,
        eps1,
        eps2,
        d,
        weight_decay,
        maximize,
    )

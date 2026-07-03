from typing_extensions import *

import warnings

import torch
from torch.optim.optimizer import Optimizer, ParamsT, _get_scalar_dtype

from .optim_builder import get_optim_function, load_optim_function


class NAdam(Optimizer):
    """Implements NAdam algorithm with fp32 stepping on CPU."""

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
        momentum_decay: float = 4e-3,
        decoupled_weight_decay: bool = False,
        *,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
    ):
        """
        Args:
            params: iterable of parameters or named_parameters to optimize or
                iterable of dicts defining parameter groups. When using named_parameters,
                all parameters in all groups should be named
            lr: learning rate.
            betas: coefficients used for computing running averages of gradient and its square
            eps: term added to the denominator to improve numerical stability
            weight_decay: weight decay (L2 penalty)
            momentum_decay: momentum decay
            decoupled_weight_decay: whether to decouple the weight decay as in AdamW to obtain
                NAdamW. If True, the algorithm does not accumulate weight decay in the momentum
                nor variance.
            maximize: maximize the objective with respect to the params, instead of minimizing
            foreach: Compatible placeholder for PyTorch's NAdam optimizer.
            capturable: Compatible placeholder for PyTorch's NAdam optimizer.
            differentiable: Compatible placeholder for PyTorch's NAdam optimizer.
        """
        load_optim_function("nadam")
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
        if not 0.0 <= momentum_decay:
            raise ValueError(f"Invalid momentum_decay value: {momentum_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            momentum_decay=momentum_decay,
            decoupled_weight_decay=decoupled_weight_decay,
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
            group.setdefault("decoupled_weight_decay", False)
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0:
                    if not torch.is_tensor(p_state["step"]):
                        step_val = float(p_state["step"])
                        p_state["step"] = torch.tensor(
                            step_val, dtype=_get_scalar_dtype()
                        )
                    if not torch.is_tensor(p_state["mu_product"]):
                        mu_prod_val = float(p_state["mu_product"])
                        p_state["mu_product"] = torch.tensor(
                            mu_prod_val, dtype=_get_scalar_dtype()
                        )

    def _init_group(
        self,
        group: Dict[str, Any],
        params_with_grad: List[torch.Tensor],
        grads: List[torch.Tensor],
        exp_avgs: List[torch.Tensor],
        exp_avg_sqs: List[torch.Tensor],
        mu_products: List[torch.Tensor],
        state_steps: List[torch.Tensor],
    ):
        """Initializes the state for each parameter group.
        Results are stored in the provided lists inplace.

        Args:
            group: The parameter group to initialize.
            params_with_grad: List to store parameters with gradients.
            grads: List to store gradients of the parameters.
            exp_avgs: List to store exponential moving averages of gradients.
            exp_avg_sqs: List to store exponential moving averages of squared gradients.
            mu_products: List to store the running products of the momentum coefficients.
            state_steps: List to store the step count for each parameter.
        """
        for p in group["params"]:
            if p.grad is not None:
                if p.grad.is_sparse:
                    raise RuntimeError("NAdam does not support sparse gradients")
                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype())
                    state["mu_product"] = torch.tensor(1.0, dtype=_get_scalar_dtype())
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                mu_products.append(state["mu_product"])
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
            exp_avg_sqs: List[torch.Tensor] = []
            mu_products: List[torch.Tensor] = []
            state_steps: List[torch.Tensor] = []
            beta1, beta2 = group["betas"]

            self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                mu_products,
                state_steps,
            )

            nadam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                mu_products,
                state_steps,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                momentum_decay=group["momentum_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                decoupled_weight_decay=group["decoupled_weight_decay"],
            )

        return loss


def nadam(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    exp_avgs: List[torch.Tensor],
    exp_avg_sqs: List[torch.Tensor],
    mu_products: List[torch.Tensor],
    state_steps: List[torch.Tensor],
    decoupled_weight_decay: bool = False,
    foreach: Optional[bool] = None,
    capturable: bool = False,
    differentiable: bool = False,
    has_complex: bool = False,
    maximize: bool = False,
    *,
    beta1: Union[torch.Tensor, float],
    beta2: Union[torch.Tensor, float],
    lr: Union[float, torch.Tensor],
    weight_decay: float,
    momentum_decay: float,
    eps: float,
):
    """Functional API that performs NAdam algorithm computation.

    See `roundpipe.optim.NAdam` for details.
    """
    assert not capturable, "capturable=True is not supported."
    assert not differentiable, "differentiable=True is not supported."
    if foreach is not None:
        warnings.warn(
            "The foreach option is not supported and will be ignored.", UserWarning
        )

    lr, beta1, beta2 = float(lr), float(beta1), float(beta2)
    weight_decay, momentum_decay, eps = (
        float(weight_decay),
        float(momentum_decay),
        float(eps),
    )

    for tensor_list in (params, grads, exp_avgs, exp_avg_sqs):
        for i, t in enumerate(tensor_list):
            if torch.is_complex(t):
                tensor_list[i] = t = torch.view_as_real(t)
            assert t.is_cpu, "RoundPipe NAdam only supports CPU tensors."
            assert (
                t.dtype is torch.float32
            ), "RoundPipe NAdam only supports float32 tensors."
            assert t.is_contiguous(), "All tensors must be contiguous."

    nadam_kernel = get_optim_function("nadam")
    nadam_kernel(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        mu_products,
        state_steps,
        beta1,
        beta2,
        lr,
        weight_decay,
        momentum_decay,
        eps,
        maximize,
        decoupled_weight_decay,
    )

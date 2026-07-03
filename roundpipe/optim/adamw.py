from typing_extensions import *

import torch
from torch.optim.optimizer import ParamsT

from .adam import Adam, adam


class AdamW(Adam):
    """Implements AdamW algorithm with fp32 stepping on CPU.

    AdamW is Adam with decoupled weight decay: the weight decay does not
    accumulate in the momentum nor variance. It is equivalent to
    `roundpipe.optim.Adam` with `decoupled_weight_decay=True`.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, torch.Tensor] = 1e-3,
        betas: Tuple[Union[float, torch.Tensor], Union[float, torch.Tensor]] = (
            0.9,
            0.999,
        ),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):
        """
        Args:
            params: iterable of parameters or named_parameters to optimize or
                iterable of dicts defining parameter groups. When using named_parameters,
                all parameters in all groups should be named
            lr: learning rate.
            betas: coefficients used for computing running averages of gradient and its square
            eps: term added to the denominator to improve numerical stability
            weight_decay: weight decay coefficient
            amsgrad: whether to use the AMSGrad variant of this algorithm from the paper
                `On the Convergence of Adam and Beyond`
            maximize: maximize the objective with respect to the params, instead of minimizing
            foreach: Compatible placeholder for PyTorch's AdamW optimizer.
            capturable: Compatible placeholder for PyTorch's AdamW optimizer.
            differentiable: Compatible placeholder for PyTorch's AdamW optimizer.
            fused: Compatible placeholder for PyTorch's AdamW optimizer.
        """
        super().__init__(
            params,
            lr,
            betas,
            eps,
            weight_decay,
            amsgrad,
            foreach=foreach,
            maximize=maximize,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
            decoupled_weight_decay=True,
        )

    def __setstate__(self, state: Dict[str, Any]):
        """Sets the state of the optimizer.

        Forces `decoupled_weight_decay` to True for every parameter group so that
        loading any state into AdamW keeps the decoupled weight decay behavior.

        Args:
            state: The state dictionary to set.
        """
        super().__setstate__(state)
        for group in self.param_groups:
            group["decoupled_weight_decay"] = True


def adamw(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    exp_avgs: List[torch.Tensor],
    exp_avg_sqs: List[torch.Tensor],
    max_exp_avg_sqs: List[torch.Tensor],
    state_steps: List[torch.Tensor],
    foreach: Optional[bool] = None,
    capturable: bool = False,
    differentiable: bool = False,
    fused: Optional[bool] = None,
    grad_scale: Optional[torch.Tensor] = None,
    found_inf: Optional[torch.Tensor] = None,
    has_complex: bool = False,
    *,
    amsgrad: bool,
    beta1: Union[torch.Tensor, float],
    beta2: Union[torch.Tensor, float],
    lr: Union[float, torch.Tensor],
    weight_decay: float,
    eps: float,
    maximize: bool,
):
    """Functional API that performs AdamW algorithm computation.

    See `roundpipe.optim.AdamW` for details.
    """
    adam(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        foreach=foreach,
        capturable=capturable,
        differentiable=differentiable,
        fused=fused,
        grad_scale=grad_scale,
        found_inf=found_inf,
        has_complex=has_complex,
        amsgrad=amsgrad,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
        decoupled_weight_decay=True,
    )

# Optimizer

RoundPipe supports any optimizer from PyTorch and correctly synchronizes optimizer state in distributed training. For PyTorch optimizers that support a `fused` implementation, we recommend passing `fused=True` when creating the optimizer for significantly better performance. However, PyTorch's built-in optimizers have poor CPU performance, so we maintain CPU-optimized implementations of selected optimizers, compiled specifically for the host CPU to accelerate optimizer updates.

The following are the optimizer implementations maintained in RoundPipe, listed in alphabetical order. Each is API-compatible with its `torch.optim` counterpart and can be used as a drop-in replacement, performing parameter updates on CPU in fp32 precision. They share the same limitations:

- Only supports `float32` tensors on CPU.
- Sparse gradients are not supported.
- All tensors must be contiguous.

PyTorch options that are unrelated to the algorithm are accepted as **compatibility placeholders** so existing scripts keep working: `foreach` and `fused` are ignored with a warning, while `capturable` and `differentiable` cannot be set to `True`.

## roundpipe.optim.Adadelta

```python
class roundpipe.optim.Adadelta(
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
)
```

Implements the Adadelta optimization algorithm; the interface is designed to be API-compatible with `torch.optim.Adadelta` and can be used as a drop-in replacement.

**Parameters:**

- `params`: An iterable of parameters or dicts defining parameter groups.
- `lr`: Coefficient that scales the delta before it is applied to the parameters. Defaults to `1.0`.
- `rho`: Coefficient used for computing a running average of squared gradients. Defaults to `0.9`.
- `eps`: Term added to the denominator to improve numerical stability. Defaults to `1e-6`.
- `weight_decay`: Weight decay (L2 penalty) coefficient. Defaults to `0.0`.
- `maximize`: Whether to maximize the objective function (instead of minimizing). Defaults to `False`.
- `foreach`: Compatibility placeholder parameter; ignored with a warning if provided.
- `capturable`: Compatibility placeholder parameter; setting to `True` is not supported.
- `differentiable`: Compatibility placeholder parameter; setting to `True` is not supported.

## roundpipe.optim.Adagrad

```python
class roundpipe.optim.Adagrad(
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
)
```

Implements the Adagrad optimization algorithm; the interface is designed to be API-compatible with `torch.optim.Adagrad` and can be used as a drop-in replacement.

**Parameters:**

- `params`: An iterable of parameters or dicts defining parameter groups.
- `lr`: Learning rate. Defaults to `1e-2`.
- `lr_decay`: Learning rate decay. Defaults to `0.0`.
- `weight_decay`: Weight decay (L2 penalty) coefficient. Defaults to `0.0`.
- `initial_accumulator_value`: Initial value for the accumulated sum of squared gradients. Defaults to `0.0`.
- `eps`: Term added to the denominator to improve numerical stability. Defaults to `1e-10`.
- `maximize`: Whether to maximize the objective function (instead of minimizing). Defaults to `False`.
- `foreach`: Compatibility placeholder parameter; ignored with a warning if provided.
- `differentiable`: Compatibility placeholder parameter; setting to `True` is not supported.
- `fused`: Compatibility placeholder parameter; ignored with a warning if provided.

## roundpipe.optim.Adam

```python
class roundpipe.optim.Adam(
    params: ParamsT,
    lr: Union[float, torch.Tensor] = 1e-3,
    betas: Tuple[Union[float, torch.Tensor], Union[float, torch.Tensor]] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    amsgrad: bool = False,
    *,
    foreach: Optional[bool] = None,
    maximize: bool = False,
    capturable: bool = False,
    differentiable: bool = False,
    fused: Optional[bool] = None,
    decoupled_weight_decay: bool = False,
)
```

Implements the Adam optimization algorithm; the interface is designed to be API-compatible with `torch.optim.Adam` and can be used as a drop-in replacement.

**Parameters:**

- `params`: An iterable of parameters or dicts defining parameter groups.
- `lr`: Learning rate. Defaults to `1e-3`.
- `betas`: Coefficients used for computing running averages of gradient and its square. Defaults to `(0.9, 0.999)`.
- `eps`: Term added to the denominator to improve numerical stability. Defaults to `1e-8`.
- `weight_decay`: Weight decay (L2 penalty) coefficient. Defaults to `0.0`.
- `amsgrad`: Whether to use the AMSGrad variant from the paper *On the Convergence of Adam and Beyond*. Defaults to `False`.
- `maximize`: Whether to maximize the objective function (instead of minimizing). Defaults to `False`.
- `decoupled_weight_decay`: If `True`, equivalent to the AdamW algorithm where weight decay does not accumulate in the momentum and variance terms. Defaults to `False`.
- `foreach`: Compatibility placeholder parameter; ignored with a warning if provided.
- `capturable`: Compatibility placeholder parameter; setting to `True` is not supported.
- `differentiable`: Compatibility placeholder parameter; setting to `True` is not supported.
- `fused`: Compatibility placeholder parameter; ignored with a warning if provided.

## roundpipe.optim.Adamax

```python
class roundpipe.optim.Adamax(
    params: ParamsT,
    lr: Union[float, torch.Tensor] = 2e-3,
    betas: Tuple[Union[float, torch.Tensor], Union[float, torch.Tensor]] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    foreach: Optional[bool] = None,
    *,
    maximize: bool = False,
    differentiable: bool = False,
    capturable: bool = False,
)
```

Implements the Adamax optimization algorithm (a variant of Adam based on the infinity norm); the interface is designed to be API-compatible with `torch.optim.Adamax` and can be used as a drop-in replacement.

**Parameters:**

- `params`: An iterable of parameters or dicts defining parameter groups.
- `lr`: Learning rate. Defaults to `2e-3`.
- `betas`: Coefficients used for computing running averages of gradient and its square. Defaults to `(0.9, 0.999)`.
- `eps`: Term added to the denominator to improve numerical stability. Defaults to `1e-8`.
- `weight_decay`: Weight decay (L2 penalty) coefficient. Defaults to `0.0`.
- `maximize`: Whether to maximize the objective function (instead of minimizing). Defaults to `False`.
- `foreach`: Compatibility placeholder parameter; ignored with a warning if provided.
- `capturable`: Compatibility placeholder parameter; setting to `True` is not supported.
- `differentiable`: Compatibility placeholder parameter; setting to `True` is not supported.

## roundpipe.optim.AdamW

```python
class roundpipe.optim.AdamW(
    params: ParamsT,
    lr: Union[float, torch.Tensor] = 1e-3,
    betas: Tuple[Union[float, torch.Tensor], Union[float, torch.Tensor]] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 1e-2,
    amsgrad: bool = False,
    *,
    maximize: bool = False,
    foreach: Optional[bool] = None,
    capturable: bool = False,
    differentiable: bool = False,
    fused: Optional[bool] = None,
)
```

Implements the AdamW optimization algorithm (Adam with decoupled weight decay); the interface is designed to be API-compatible with `torch.optim.AdamW` and can be used as a drop-in replacement.

**Parameters:**

- `params`: An iterable of parameters or dicts defining parameter groups.
- `lr`: Learning rate. Defaults to `1e-3`.
- `betas`: Coefficients used for computing running averages of gradient and its square. Defaults to `(0.9, 0.999)`.
- `eps`: Term added to the denominator to improve numerical stability. Defaults to `1e-8`.
- `weight_decay`: Decoupled weight decay coefficient (as in the AdamW algorithm). Defaults to `1e-2`.
- `amsgrad`: Whether to use the AMSGrad variant from the paper *On the Convergence of Adam and Beyond*. Defaults to `False`.
- `maximize`: Whether to maximize the objective function (instead of minimizing). Defaults to `False`.
- `foreach`: Compatibility placeholder parameter; ignored with a warning if provided.
- `capturable`: Compatibility placeholder parameter; setting to `True` is not supported.
- `differentiable`: Compatibility placeholder parameter; setting to `True` is not supported.
- `fused`: Compatibility placeholder parameter; ignored with a warning if provided.

## roundpipe.optim.ASGD

```python
class roundpipe.optim.ASGD(
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
)
```

Implements the ASGD (Averaged Stochastic Gradient Descent) optimization algorithm; the interface is designed to be API-compatible with `torch.optim.ASGD` and can be used as a drop-in replacement.

**Parameters:**

- `params`: An iterable of parameters or dicts defining parameter groups.
- `lr`: Learning rate. Defaults to `1e-2`.
- `lambd`: Decay term. Defaults to `1e-4`.
- `alpha`: Power used in the learning-rate (eta) update. Defaults to `0.75`.
- `t0`: Point at which to start averaging. Defaults to `1e6`.
- `weight_decay`: Weight decay (L2 penalty) coefficient. Defaults to `0.0`.
- `maximize`: Whether to maximize the objective function (instead of minimizing). Defaults to `False`.
- `foreach`: Compatibility placeholder parameter; ignored with a warning if provided.
- `capturable`: Compatibility placeholder parameter; setting to `True` is not supported.
- `differentiable`: Compatibility placeholder parameter; setting to `True` is not supported.

## roundpipe.optim.NAdam

```python
class roundpipe.optim.NAdam(
    params: ParamsT,
    lr: Union[float, torch.Tensor] = 2e-3,
    betas: Tuple[Union[float, torch.Tensor], Union[float, torch.Tensor]] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    momentum_decay: float = 4e-3,
    decoupled_weight_decay: bool = False,
    *,
    foreach: Optional[bool] = None,
    maximize: bool = False,
    capturable: bool = False,
    differentiable: bool = False,
)
```

Implements the NAdam optimization algorithm (Adam with Nesterov momentum); the interface is designed to be API-compatible with `torch.optim.NAdam` and can be used as a drop-in replacement.

**Parameters:**

- `params`: An iterable of parameters or dicts defining parameter groups.
- `lr`: Learning rate. Defaults to `2e-3`.
- `betas`: Coefficients used for computing running averages of gradient and its square. Defaults to `(0.9, 0.999)`.
- `eps`: Term added to the denominator to improve numerical stability. Defaults to `1e-8`.
- `weight_decay`: Weight decay (L2 penalty) coefficient. Defaults to `0.0`.
- `momentum_decay`: Momentum decay coefficient. Defaults to `4e-3`.
- `decoupled_weight_decay`: If `True`, uses decoupled weight decay as in AdamW, so weight decay does not accumulate in the momentum and variance terms. Defaults to `False`.
- `maximize`: Whether to maximize the objective function (instead of minimizing). Defaults to `False`.
- `foreach`: Compatibility placeholder parameter; ignored with a warning if provided.
- `capturable`: Compatibility placeholder parameter; setting to `True` is not supported.
- `differentiable`: Compatibility placeholder parameter; setting to `True` is not supported.

## roundpipe.optim.RAdam

```python
class roundpipe.optim.RAdam(
    params: ParamsT,
    lr: Union[float, torch.Tensor] = 1e-3,
    betas: Tuple[Union[float, torch.Tensor], Union[float, torch.Tensor]] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    decoupled_weight_decay: bool = False,
    *,
    foreach: Optional[bool] = None,
    maximize: bool = False,
    capturable: bool = False,
    differentiable: bool = False,
)
```

Implements the RAdam (Rectified Adam) optimization algorithm; the interface is designed to be API-compatible with `torch.optim.RAdam` and can be used as a drop-in replacement.

**Parameters:**

- `params`: An iterable of parameters or dicts defining parameter groups.
- `lr`: Learning rate. Defaults to `1e-3`.
- `betas`: Coefficients used for computing running averages of gradient and its square. Defaults to `(0.9, 0.999)`.
- `eps`: Term added to the denominator to improve numerical stability. Defaults to `1e-8`.
- `weight_decay`: Weight decay (L2 penalty) coefficient. Defaults to `0.0`.
- `decoupled_weight_decay`: If `True`, decouples the weight decay as in AdamW to obtain RAdamW, so weight decay does not accumulate in the momentum and variance terms. Defaults to `False`.
- `maximize`: Whether to maximize the objective function (instead of minimizing). Defaults to `False`.
- `foreach`: Compatibility placeholder parameter; ignored with a warning if provided.
- `capturable`: Compatibility placeholder parameter; setting to `True` is not supported.
- `differentiable`: Compatibility placeholder parameter; setting to `True` is not supported.

## roundpipe.optim.RMSprop

```python
class roundpipe.optim.RMSprop(
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
)
```

Implements the RMSprop optimization algorithm; the interface is designed to be API-compatible with `torch.optim.RMSprop` and can be used as a drop-in replacement.

**Parameters:**

- `params`: An iterable of parameters or dicts defining parameter groups.
- `lr`: Learning rate. Defaults to `1e-2`.
- `alpha`: Smoothing constant. Defaults to `0.99`.
- `eps`: Term added to the denominator to improve numerical stability. Defaults to `1e-8`.
- `weight_decay`: Weight decay (L2 penalty) coefficient. Defaults to `0.0`.
- `momentum`: Momentum factor. Defaults to `0.0`.
- `centered`: If `True`, compute the centered RMSprop, where the gradient is normalized by an estimate of its variance. Defaults to `False`.
- `maximize`: Whether to maximize the objective function (instead of minimizing). Defaults to `False`.
- `foreach`: Compatibility placeholder parameter; ignored with a warning if provided.
- `capturable`: Compatibility placeholder parameter; setting to `True` is not supported.
- `differentiable`: Compatibility placeholder parameter; setting to `True` is not supported.

## roundpipe.optim.Rprop

```python
class roundpipe.optim.Rprop(
    params: ParamsT,
    lr: Union[float, torch.Tensor] = 1e-2,
    etas: Tuple[float, float] = (0.5, 1.2),
    step_sizes: Tuple[float, float] = (1e-6, 50),
    *,
    capturable: bool = False,
    foreach: Optional[bool] = None,
    maximize: bool = False,
    differentiable: bool = False,
)
```

Implements the Rprop (resilient backpropagation) optimization algorithm; the interface is designed to be API-compatible with `torch.optim.Rprop` and can be used as a drop-in replacement.

**Parameters:**

- `params`: An iterable of parameters or dicts defining parameter groups.
- `lr`: Learning rate. Defaults to `1e-2`.
- `etas`: Pair of multiplicative decrease and increase factors `(etaminus, etaplus)` applied to the step size. Defaults to `(0.5, 1.2)`.
- `step_sizes`: Pair of minimal and maximal allowed step sizes. Defaults to `(1e-6, 50)`.
- `maximize`: Whether to maximize the objective function (instead of minimizing). Defaults to `False`.
- `foreach`: Compatibility placeholder parameter; ignored with a warning if provided.
- `capturable`: Compatibility placeholder parameter; setting to `True` is not supported.
- `differentiable`: Compatibility placeholder parameter; setting to `True` is not supported.

## roundpipe.optim.SGD

```python
class roundpipe.optim.SGD(
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
)
```

Implements the SGD (stochastic gradient descent, optionally with momentum) optimization algorithm; the interface is designed to be API-compatible with `torch.optim.SGD` and can be used as a drop-in replacement.

**Parameters:**

- `params`: An iterable of parameters or dicts defining parameter groups.
- `lr`: Learning rate. Defaults to `1e-3`.
- `momentum`: Momentum factor. Defaults to `0.0`.
- `dampening`: Dampening for momentum. Defaults to `0.0`.
- `weight_decay`: Weight decay (L2 penalty) coefficient. Defaults to `0.0`.
- `nesterov`: Enables Nesterov momentum. Defaults to `False`.
- `maximize`: Whether to maximize the objective function (instead of minimizing). Defaults to `False`.
- `foreach`: Compatibility placeholder parameter; ignored with a warning if provided.
- `differentiable`: Compatibility placeholder parameter; setting to `True` is not supported.
- `fused`: Compatibility placeholder parameter; ignored with a warning if provided.

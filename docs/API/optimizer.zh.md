# 优化器

RoundPipe 支持 PyTorch 中的任何优化器，并且在分布式训练中能够正确地同步优化器状态。对于支持`fused`实现的 PyTorch 优化器，我们建议您在创建优化器时传入`fused=True`，可以显著提升性能。不过 PyTorch 中的优化器在 CPU 上的性能较差，因此我们维护了部分优化器的 CPU 实现，通过针对运行设备的 CPU 进行编译优化，加快优化器的更新速度。

以下是 RoundPipe 中维护的优化器实现，按照字母顺序排列。每个实现都与其对应的 `torch.optim` 实现 API 兼容，可直接替换使用，并使用 fp32 精度在 CPU 上执行参数更新。它们具有相同的限制：

- 仅支持 CPU 上的 `float32` 张量。
- 不支持稀疏梯度。
- 所有张量必须是连续的（contiguous）。

和算法无关的 PyTorch 选项会作为**兼容占位参数**接受，以保证现有脚本继续工作：`foreach` 和 `fused` 传入时会被忽略并产生警告，而 `capturable` 和 `differentiable` 不支持设为 `True`。

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

实现 Adadelta 优化算法，接口设计与 `torch.optim.Adadelta` 兼容，可直接替换使用。

**参数：**

- `params`：参数迭代器或定义参数组的字典迭代器。
- `lr`：在将 delta 应用到参数之前对其进行缩放的系数。默认为 `1.0`。
- `rho`：用于计算梯度平方的运行均值的系数。默认为 `0.9`。
- `eps`：添加到分母以提高数值稳定性的项。默认为 `1e-6`。
- `weight_decay`：权重衰减（L2 penalty）系数。默认为 `0.0`。
- `maximize`：是否最大化目标函数（而非最小化）。默认为 `False`。
- `foreach`：兼容占位参数，传入时会被忽略并产生警告。
- `capturable`：兼容占位参数，不支持设为 `True`。
- `differentiable`：兼容占位参数，不支持设为 `True`。

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

实现 Adagrad 优化算法，接口设计与 `torch.optim.Adagrad` 兼容，可直接替换使用。

**参数：**

- `params`：参数迭代器或定义参数组的字典迭代器。
- `lr`：学习率。默认为 `1e-2`。
- `lr_decay`：学习率衰减。默认为 `0.0`。
- `weight_decay`：权重衰减（L2 penalty）系数。默认为 `0.0`。
- `initial_accumulator_value`：梯度平方累积和的初始值。默认为 `0.0`。
- `eps`：添加到分母以提高数值稳定性的项。默认为 `1e-10`。
- `maximize`：是否最大化目标函数（而非最小化）。默认为 `False`。
- `foreach`：兼容占位参数，传入时会被忽略并产生警告。
- `differentiable`：兼容占位参数，不支持设为 `True`。
- `fused`：兼容占位参数，传入时会被忽略并产生警告。

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

实现 Adam 优化算法，接口设计与 `torch.optim.Adam` 兼容，可直接替换使用。

**参数：**

- `params`：参数迭代器或定义参数组的字典迭代器。
- `lr`：学习率。默认为 `1e-3`。
- `betas`：用于计算梯度及其平方的运行均值的系数。默认为 `(0.9, 0.999)`。
- `eps`：添加到分母以提高数值稳定性的项。默认为 `1e-8`。
- `weight_decay`：权重衰减（L2 penalty）系数。默认为 `0.0`。
- `amsgrad`：是否使用论文 *On the Convergence of Adam and Beyond* 中的 AMSGrad 变体。默认为 `False`。
- `maximize`：是否最大化目标函数（而非最小化）。默认为 `False`。
- `decoupled_weight_decay`：如果为 `True`，等效于 AdamW 算法，权重衰减不会累积在动量和方差中。默认为 `False`。
- `foreach`：兼容占位参数，传入时会被忽略并产生警告。
- `capturable`：兼容占位参数，不支持设为 `True`。
- `differentiable`：兼容占位参数，不支持设为 `True`。
- `fused`：兼容占位参数，传入时会被忽略并产生警告。

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

实现 Adamax 优化算法（基于无穷范数的 Adam 变体），接口设计与 `torch.optim.Adamax` 兼容，可直接替换使用。

**参数：**

- `params`：参数迭代器或定义参数组的字典迭代器。
- `lr`：学习率。默认为 `2e-3`。
- `betas`：用于计算梯度及其平方的运行均值的系数。默认为 `(0.9, 0.999)`。
- `eps`：添加到分母以提高数值稳定性的项。默认为 `1e-8`。
- `weight_decay`：权重衰减（L2 penalty）系数。默认为 `0.0`。
- `maximize`：是否最大化目标函数（而非最小化）。默认为 `False`。
- `foreach`：兼容占位参数，传入时会被忽略并产生警告。
- `capturable`：兼容占位参数，不支持设为 `True`。
- `differentiable`：兼容占位参数，不支持设为 `True`。

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

实现 AdamW 优化算法（带解耦权重衰减的 Adam），接口设计与 `torch.optim.AdamW` 兼容，可直接替换使用。

**参数：**

- `params`：参数迭代器或定义参数组的字典迭代器。
- `lr`：学习率。默认为 `1e-3`。
- `betas`：用于计算梯度及其平方的运行均值的系数。默认为 `(0.9, 0.999)`。
- `eps`：添加到分母以提高数值稳定性的项。默认为 `1e-8`。
- `weight_decay`：解耦的权重衰减系数（如 AdamW 算法）。默认为 `1e-2`。
- `amsgrad`：是否使用论文 *On the Convergence of Adam and Beyond* 中的 AMSGrad 变体。默认为 `False`。
- `maximize`：是否最大化目标函数（而非最小化）。默认为 `False`。
- `foreach`：兼容占位参数，传入时会被忽略并产生警告。
- `capturable`：兼容占位参数，不支持设为 `True`。
- `differentiable`：兼容占位参数，不支持设为 `True`。
- `fused`：兼容占位参数，传入时会被忽略并产生警告。

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

实现 ASGD（平均随机梯度下降）优化算法，接口设计与 `torch.optim.ASGD` 兼容，可直接替换使用。

**参数：**

- `params`：参数迭代器或定义参数组的字典迭代器。
- `lr`：学习率。默认为 `1e-2`。
- `lambd`：衰减项。默认为 `1e-4`。
- `alpha`：学习率（eta）更新中使用的幂。默认为 `0.75`。
- `t0`：开始平均的时间点。默认为 `1e6`。
- `weight_decay`：权重衰减（L2 penalty）系数。默认为 `0.0`。
- `maximize`：是否最大化目标函数（而非最小化）。默认为 `False`。
- `foreach`：兼容占位参数，传入时会被忽略并产生警告。
- `capturable`：兼容占位参数，不支持设为 `True`。
- `differentiable`：兼容占位参数，不支持设为 `True`。

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

实现 NAdam 优化算法（带 Nesterov 动量的 Adam），接口设计与 `torch.optim.NAdam` 兼容，可直接替换使用。

**参数：**

- `params`：参数迭代器或定义参数组的字典迭代器。
- `lr`：学习率。默认为 `2e-3`。
- `betas`：用于计算梯度及其平方的运行均值的系数。默认为 `(0.9, 0.999)`。
- `eps`：添加到分母以提高数值稳定性的项。默认为 `1e-8`。
- `weight_decay`：权重衰减（L2 penalty）系数。默认为 `0.0`。
- `momentum_decay`：动量衰减系数。默认为 `4e-3`。
- `decoupled_weight_decay`：如果为 `True`，使用如 AdamW 的解耦权重衰减，权重衰减不会累积在动量和方差中。默认为 `False`。
- `maximize`：是否最大化目标函数（而非最小化）。默认为 `False`。
- `foreach`：兼容占位参数，传入时会被忽略并产生警告。
- `capturable`：兼容占位参数，不支持设为 `True`。
- `differentiable`：兼容占位参数，不支持设为 `True`。

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

实现 RAdam（修正版 Adam）优化算法，接口设计与 `torch.optim.RAdam` 兼容，可直接替换使用。

**参数：**

- `params`：参数迭代器或定义参数组的字典迭代器。
- `lr`：学习率。默认为 `1e-3`。
- `betas`：用于计算梯度及其平方的运行均值的系数。默认为 `(0.9, 0.999)`。
- `eps`：添加到分母以提高数值稳定性的项。默认为 `1e-8`。
- `weight_decay`：权重衰减（L2 penalty）系数。默认为 `0.0`。
- `decoupled_weight_decay`：如果为 `True`，如 AdamW 那样解耦权重衰减以得到 RAdamW，权重衰减不会累积在动量和方差中。默认为 `False`。
- `maximize`：是否最大化目标函数（而非最小化）。默认为 `False`。
- `foreach`：兼容占位参数，传入时会被忽略并产生警告。
- `capturable`：兼容占位参数，不支持设为 `True`。
- `differentiable`：兼容占位参数，不支持设为 `True`。

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

实现 RMSprop 优化算法，接口设计与 `torch.optim.RMSprop` 兼容，可直接替换使用。

**参数：**

- `params`：参数迭代器或定义参数组的字典迭代器。
- `lr`：学习率。默认为 `1e-2`。
- `alpha`：平滑常数。默认为 `0.99`。
- `eps`：添加到分母以提高数值稳定性的项。默认为 `1e-8`。
- `weight_decay`：权重衰减（L2 penalty）系数。默认为 `0.0`。
- `momentum`：动量因子。默认为 `0.0`。
- `centered`：如果为 `True`，计算中心化的 RMSprop，梯度会被其方差的估计值归一化。默认为 `False`。
- `maximize`：是否最大化目标函数（而非最小化）。默认为 `False`。
- `foreach`：兼容占位参数，传入时会被忽略并产生警告。
- `capturable`：兼容占位参数，不支持设为 `True`。
- `differentiable`：兼容占位参数，不支持设为 `True`。

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

实现 Rprop（弹性反向传播）优化算法，接口设计与 `torch.optim.Rprop` 兼容，可直接替换使用。

**参数：**

- `params`：参数迭代器或定义参数组的字典迭代器。
- `lr`：学习率。默认为 `1e-2`。
- `etas`：作用于步长的一对乘性减小与增大因子 `(etaminus, etaplus)`。默认为 `(0.5, 1.2)`。
- `step_sizes`：允许的最小与最大步长构成的一对值。默认为 `(1e-6, 50)`。
- `maximize`：是否最大化目标函数（而非最小化）。默认为 `False`。
- `foreach`：兼容占位参数，传入时会被忽略并产生警告。
- `capturable`：兼容占位参数，不支持设为 `True`。
- `differentiable`：兼容占位参数，不支持设为 `True`。

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

实现 SGD（随机梯度下降，可选带动量）优化算法，接口设计与 `torch.optim.SGD` 兼容，可直接替换使用。

**参数：**

- `params`：参数迭代器或定义参数组的字典迭代器。
- `lr`：学习率。默认为 `1e-3`。
- `momentum`：动量因子。默认为 `0.0`。
- `dampening`：动量的阻尼系数。默认为 `0.0`。
- `weight_decay`：权重衰减（L2 penalty）系数。默认为 `0.0`。
- `nesterov`：启用 Nesterov 动量。默认为 `False`。
- `maximize`：是否最大化目标函数（而非最小化）。默认为 `False`。
- `foreach`：兼容占位参数，传入时会被忽略并产生警告。
- `differentiable`：兼容占位参数，不支持设为 `True`。
- `fused`：兼容占位参数，传入时会被忽略并产生警告。

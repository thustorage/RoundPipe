# 贡献一个 CPU 优化器

本教程将带你了解 RoundPipe 如何实现高性能的 CPU 端优化器，并手把手教你如何添加自己的优化器。

读完之后，你将理解构成每个 RoundPipe 优化器的三个层次、每一处性能优化背后的原因，以及如何确认你的 kernel 确实被编译成了 SIMD 指令。

我们以 `roundpipe.optim.Adam` 作为示例，其代码位于 `roundpipe/optim/` 目录下。

## RoundPipe 的 CPU 优化器有什么不同

RoundPipe **优化器更新在 CPU 上执行**，但 PyTorch 内置的优化器在 CPU 上性能较差。对于大模型，优化器可能成为拖慢训练的瓶颈。因此 RoundPipe 提供了优化过的优化器的 CPU 实现，它们具有以下优势：

- **针对运行主机的 CPU 编译** —— 使用 `-march=native`，从而利用每一种可用的 SIMD 指令集（AVX2、AVX-512 等）。
- **热点循环中无分支** —— 运行期的开关标志通过模板特化被提升到编译期，使内层循环能够干净地向量化。
- **不持有 GIL** —— 绑定在执行步骤期间释放 GIL，因此一次耗时较长的更新不会阻塞其他 Python 线程。
- **可直接替换** —— Python API 与 `torch.optim` 完全兼容，且计算结果与 PyTorch 数值一致。

## RoundPipe 优化器的三个层次

每个优化器从 PyTorch 到具体实现分为三层：

```
roundpipe/optim/
├── adam.py            # 第 1 层：Python 类 + 函数式 API（与 PyTorch 兼容）
├── optim_builder.py   # 第 2 层：为当前主机 JIT 编译并缓存 C++ kernel
└── csrc/
    └── adam.cpp       # 第 3 层：OpenMP/SIMD 的 C++ kernel（真正的计算）
```

1. **Python 层**（`adam.py`）是一个普通的 `torch.optim.Optimizer` 子类。它像 PyTorch
   一样管理参数组和每个参数的状态，校验张量，然后把扁平的指针交给编译好的 kernel。
2. **构建层**（`optim_builder.py`）使用激进的、针对主机的编译选项即时（JIT）编译 C++
   源码，并缓存结果，因此每台机器只需付出一次编译开销。
3. **kernel 层**（`csrc/adam.cpp`）是真正进行计算的地方，也是需要优化的部分。

下面我们以 Adam 为例逐层剖析。

## 第 3 层：C++ kernel

`roundpipe/optim/csrc/adam.cpp`由三部分组成：模板化的内层 kernel、一个把运行期布尔值转换为模板参数的辅助函数，以及暴露给 Python 的分发函数。

### 内层循环

```cpp
template <bool amsgrad, bool maximize, bool zero_weight_decay,
          bool decoupled_weight_decay>
void adam_kernel(float *__restrict params, const float *__restrict grads,
                 float *__restrict exp_avg, float *__restrict exp_avg_sq,
                 float *__restrict max_exp_avg_sq, float lr, float beta1, float beta2,
                 float eps, float weight_decay, int64_t param_size, int64_t step) {
    float one_beta1 = 1.0f - beta1;
    float one_beta2 = 1.0f - beta2;
    float bias_correction1 = 1.0f - powf(beta1, step);
    float bias_correction2 = 1.0f - powf(beta2, step);
    float one_lr_weight_decay = 1.0f - lr * weight_decay;
    float step_size = lr / bias_correction1;
    float div_bias_correction2 = 1.0f / bias_correction2;

    for (int64_t i = 0; i < param_size; ++i) {
        float grad = !maximize ? grads[i] : -grads[i];
        if (!zero_weight_decay) {
            if (decoupled_weight_decay) {
                params[i] *= one_lr_weight_decay;
            } else {
                grad += weight_decay * params[i];
            }
        }
        exp_avg[i] = beta1 * exp_avg[i] + one_beta1 * grad;
        exp_avg_sq[i] = beta2 * exp_avg_sq[i] + one_beta2 * grad * grad;
        float denom;
        if (amsgrad) {
            max_exp_avg_sq[i] = fmaxf(max_exp_avg_sq[i], exp_avg_sq[i]);
            denom = sqrtf(max_exp_avg_sq[i] * div_bias_correction2) + eps;
        } else {
            denom = sqrtf(exp_avg_sq[i] * div_bias_correction2) + eps;
        }
        params[i] -= step_size * exp_avg[i] / denom;
    }
}
```

为了获得最佳性能，内层循环做了以下优化：

- **所有循环不变量都被提到循环外。** 偏差修正、`step_size` 以及各类倒数都只计算一次。
  注意到 `div_bias_correction2` 是一个倒数，因此循环内使用乘法而非除法。
- **所有分支放入模板参数。** 由于 `amsgrad`、`maximize`、`zero_weight_decay`
  和 `decoupled_weight_decay` 都是编译期常量，上面的每个 `if` 都在编译期就被求值，自动清除无用的
  分支。剩下的只有算术运算，编译器可以将其自动向量化为 SIMD。
- **`__restrict` 指针**向编译器保证这些数组不会相互别名，因此编译器可以安全地
  重排读写并进行向量化。

### 把运行期开关转为编译期开关

我们无法直接用运行期的布尔值去调用模板。下面这个辅助函数每次剥离一个布尔值，并把它重新作为模板参数导出：

```cpp
template <bool... FixedBools, typename... Args>
void adam_kernel(bool current_bool, Args... args) {
    if (current_bool) {
        adam_kernel<FixedBools..., true>(args...);
    } else {
        adam_kernel<FixedBools..., false>(args...);
    }
}
```

调用 `adam_kernel(amsgrad, maximize, zero_weight_decay, decoupled_weight_decay, ...)`
会在编译期递归地生成全部 `2^4 = 16` 个特化版本，并在运行期用分发到正确的版本。

### 跨核心并行

分发函数 `adam` 接收参数张量列表，取出裸指针，并运行一个 OpenMP 并行区域：

```cpp
#pragma omp parallel
{
    int rank = omp_get_thread_num();
    int nthreads = omp_get_num_threads();
    for (size_t i = 0; i < params.size(); ++i) {
        int64_t block_size = numel[i] / nthreads + (rank < (numel[i] % nthreads));
        int64_t offset =
            (numel[i] / nthreads) * rank + min<int64_t>(rank, numel[i] % nthreads);
        adam_kernel(amsgrad, maximize, weight_decay == 0.0f, decoupled_weight_decay,
                    params_ptr[i] + offset, grads_ptr[i] + offset,
                    exp_avg_ptr[i] + offset, exp_avg_sq_ptr[i] + offset,
                    max_exp_avg_sq_ptr[i] + offset, lr, beta1, beta2, eps,
                    weight_decay, block_size, step_int[i]);
    }
}
```

关键细节在于：RoundPipe **不是**按参数**列表**来并行（即一个线程处理一个张量）。参数张量
的大小差异极大——例如嵌入矩阵与偏置向量——那样会让大多数核心处于空闲。相反，**每个线程都
从每个张量中取走一段连续的切片**。`nthreads` 个线程中的每一个大约处理每个张量
`numel / nthreads` 个元素；`rank < numel % nthreads` 这一项把余下的元素每个分给编号最小
的那些线程。这样无论大小如何分布都能获得近乎完美的负载均衡，而且每个线程访问的是连续内存，
对缓存和 SIMD 都很友好。

### 将 kernel 暴露给 Python

文件末尾用 pybind11 绑定该函数，并释放 GIL：

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adam", &adam, py::call_guard<py::gil_scoped_release>(),
          "Adam optimizer step implementation in C++");
}
```

`py::gil_scoped_release` 在 kernel 执行期间释放 GIL，因此更新不会在
运行时阻塞其他 Python 线程。

## 第 2 层：构建与缓存 kernel

`roundpipe/optim/optim_builder.py` 在首次使用时用 `torch.utils.cpp_extension.load` 编译
kernel，并把可调用对象缓存在一个模块级字典中。

Python 层只需要这两个导出的辅助函数：

- `load_optim_function(name)` —— （如有必要）编译并缓存 kernel。
- `get_optim_function(name)` —— 返回缓存的可调用对象，必要时按需编译。

一切都以 **`name`** 为键，它必须与以下两者匹配：

- C++ 源文件 `csrc/<name>.cpp`，以及
- 其中的 pybind11 函数名（`m.def("<name>", ...)`）。

编译出的模块名里折入了两个哈希——`get_cpu_flags_hash()`（主机的 CPU 特性标志）和
`get_cpp_flags_hash()`（编译选项）。如果你把缓存的构建产物挪到另一台 CPU 上，或修改了
`CPP_FLAGS`，模块名就会改变，kernel 会被透明地重新编译，而不会悄悄运行一个为另一台机器
调优的二进制。添加优化器时，你通常无需改动这个文件。

## 验证内层循环已向量化 {#verify-vectorization}

为了确认核心循环已被向量化，RoundPipe **在构建每个 kernel
时都始终开启向量化报告**，每次 kernel 编译都会留下一个 `vec.log` 。

### 找到 `vec.log`

编译发生在 Torch 扩展的构建目录中，默认是
`~/.cache/torch_extensions/…/roundpipe_optim_adam_<hashes>/`（可通过环境变量
`TORCH_EXTENSIONS_DIR` 覆盖）。由于 kernel 会被缓存，该文件反映的是
最近一次编译的结果。如果 kernel 已经构建好而你想要一份新的报告，删除该模块的构建目录（或
改动源码、编译选项、主机 CPU）即可触发重新编译，例如：

```bash
python -c "import torch; from roundpipe.optim import Adam; \
p = torch.zeros(1024, requires_grad=True); p.grad = torch.randn(1024); \
Adam([p]).step()"
```

### 阅读日志

查找指向 `adam.cpp` 中内层 `for` 循环的一行：

```
adam.cpp:20:29: optimized: loop vectorized using 32 byte vectors
```

`32 byte vectors` 表示使用了 256 位的 AVX；在支持 AVX-512 的主机上你会看到 `64 byte vectors`。

向量化失败的原因通常有：

- **循环里有分支** → 把运行期开关提升为 `bool` 模板参数（如 `amsgrad` / `maximize`），
  使分支在编译期被消除。
- **指针可能别名** → 给数组指针加上 `__restrict`。
- **循环里有函数调用** → 将一切循环不变量移出循环，确保调用的函数可以被内联。

## 第 1 层：Python 优化器类

`roundpipe/optim/adam.py` 是一个普通的 `torch.optim.Optimizer` 子类。它要在接口和行为
上都与 `torch.optim.Adam` 完全一致，然后把计算转发给 kernel。它有三项职责。

**1. 在构造优化器时编译 kernel。** `__init__` 的第一行就预热缓存，使第一次 `step()` 不会因
编译而变慢：

```python
def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), ...):
    load_optim_function("adam")
    ...
```

**2. 像 PyTorch 一样管理状态。** `_init_group` 惰性地分配 `step`、`exp_avg`、`exp_avg_sq`
（以及用于 AMSGrad 的 `max_exp_avg_sq`），并把每个参数的张量收集进扁平列表。`step()` 遍历
各参数组并调用函数式的 `adam(...)`，保持与 PyTorch 完全一致。

**3. 先校验，再调用 kernel。** 函数式的 `adam(...)` 调用 C++ 实现前先检查输入是否满足要求，
通常包括：

- kernel 兼容的数据类型和布局。
- PyTorch 支持而本 kernel 不支持的选项（`fused`、`foreach`、`capturable`、`differentiable`）
  被当作**兼容占位参数**接受。视情况忽略、给出警告或抛出错误，确保现有的训练脚本
  无需改动即可继续工作。

最后，`__init__.py` 导出该类：

```python
from .adam import Adam

__all__ = ["Adam"]
```

## 正确性：与 PyTorch 对拍

每个优化器都要与其 PyTorch 对应实现进行对拍验证。
`tests/test_optim.py` 构造了一批五花八门的张量，在相同的梯度下让两个优化器各跑若干步，并检查结果。

所有优化器都需要编写对应的测试，枚举不同的优化器参数并调用`run_optim`进行测试。


## 检查清单

- `csrc/<name>.cpp`：模板化的无分支内层循环、`__restrict` 指针、循环不变量外提、跨线程的
  OpenMP 按块切分。
- pybind 绑定名恰好为 `<name>`，并带 `py::gil_scoped_release`。
- 通过查看 `vec.log` 确认内层循环已向量化（见[验证内层循环已向量化](#verify-vectorization)）。
- `<name>.py`：与 PyTorch 兼容的 API、`__init__` 中的 `load_optim_function`、CPU/fp32/连续
  的断言、通过 `view_as_real` 处理复数。
- 不支持的 PyTorch 选项以兼容占位参数处理。
- 从 `optim/__init__.py` 导出。
- 与 PyTorch 参考实现对拍，在所有开关组合下通过测试。
- 在 `docs/API/optimizer.*.md` 添加对应的文档。

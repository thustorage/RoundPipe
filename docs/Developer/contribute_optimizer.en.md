# Contribute a CPU Optimizer

This tutorial walks you through how RoundPipe implements fast, CPU-side
optimizers, and shows you — step by step — how to add your own.

By the end you will understand the three layers that make up every RoundPipe
optimizer, the reason behind each performance optimization, and how to confirm
that your kernel actually compiles down to SIMD.

We use `roundpipe.optim.Adam` as the example. Its code lives under
`roundpipe/optim/`.

## What's Different About RoundPipe's CPU Optimizers

In RoundPipe, **optimizer updates run on the CPU**, but PyTorch's built-in
optimizers have poor CPU performance. For a large model the optimizer can become
a bottleneck that slows training. RoundPipe therefore provides optimized CPU
implementations of these optimizers, with the following advantages:

- **Compiled specifically for the host CPU** — using `-march=native` so every
  available SIMD instruction set (AVX2, AVX-512, …) is exploited.
- **Branch-free in the hot loop** — runtime flags are lifted to compile time via
  template specialization so the inner loop vectorizes cleanly.
- **GIL-free** — the binding releases the GIL for the duration of the step, so a
  long update never blocks other Python threads.
- **A drop-in replacement** — the Python API is byte-for-byte compatible with
  `torch.optim`, and results match PyTorch numerically.

## The Three Layers of a RoundPipe Optimizer

Every optimizer is split into three layers, from the PyTorch API down to the
implementation:

```
roundpipe/optim/
├── adam.py            # Layer 1: Python class + functional API (PyTorch-compatible)
├── optim_builder.py   # Layer 2: JIT compile & cache the C++ kernel for this host
└── csrc/
    └── adam.cpp       # Layer 3: the OpenMP/SIMD C++ kernel (the actual math)
```

1. **The Python layer** (`adam.py`) is a normal `torch.optim.Optimizer`
   subclass. It manages parameter groups and per-parameter state exactly like
   PyTorch, validates the tensors, and then hands the flat pointers to the
   compiled kernel.
2. **The build layer** (`optim_builder.py`) just-in-time compiles the C++
   source with aggressive, host-specific flags, and caches the result so you
   only pay the compile cost once per machine.
3. **The kernel layer** (`csrc/adam.cpp`) is where the computation actually
   happens — the part you optimize.

Let's look at each layer through the lens of Adam.

## Layer 3: The C++ Kernel

`roundpipe/optim/csrc/adam.cpp` has three parts: the templated inner kernel, a
helper that turns runtime booleans into template arguments, and the dispatch
function exposed to Python.

### The inner loop

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

For the best performance, the inner loop applies these optimizations:

- **All loop-invariant work is hoisted out of the loop.** Bias corrections,
  `step_size`, and reciprocals are computed once. Note `div_bias_correction2`
  is a reciprocal so the loop uses a multiply instead of a divide.
- **All branches are lifted into template parameters.** Because `amsgrad`,
  `maximize`, `zero_weight_decay`, and `decoupled_weight_decay` are compile-time
  constants, every `if` above is evaluated at compile time, automatically
  removing the dead branches. What remains is pure arithmetic that the compiler
  can auto-vectorize into SIMD.
- **`__restrict` pointers** promise the compiler that the arrays don't alias, so
  it can safely reorder loads/stores and vectorize.

### Turning runtime flags into compile-time flags

We can't call a template directly with runtime booleans. This helper peels the
booleans off one at a time and re-emits them as template arguments:

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

Calling `adam_kernel(amsgrad, maximize, zero_weight_decay, decoupled_weight_decay, ...)`
recursively generates all `2^4 = 16` specializations at compile time and
dispatches to the right one at runtime.

### Parallelizing across cores

The dispatch function `adam` receives the list of parameter tensors, extracts
raw pointers, and runs one OpenMP parallel region:

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

The important detail: RoundPipe does **not** parallelize over the *list* of
parameters (one tensor per thread). Parameter tensors vary enormously in size —
an embedding matrix versus a bias vector — so that would leave most cores idle.
Instead, **every thread takes a contiguous slice of every tensor**. Each of the
`nthreads` threads handles roughly `numel / nthreads` elements of each tensor;
the `rank < numel % nthreads` term hands the leftover elements to the lowest
ranks, one each. This gives near-perfect load balance no matter how sizes are
distributed, and each thread walks contiguous memory, which is cache- and
SIMD-friendly.

### Exposing the kernel to Python

The bottom of the file binds the function with pybind11 and releases the GIL:

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adam", &adam, py::call_guard<py::gil_scoped_release>(),
          "Adam optimizer step implementation in C++");
}
```

`py::gil_scoped_release` drops the GIL for the duration of the kernel, so the
update doesn't block other Python threads while it runs.

## Layer 2: Building and Caching the Kernel

`roundpipe/optim/optim_builder.py` compiles the kernel on first use with
`torch.utils.cpp_extension.load` and caches the callable in a module-level dict.

The two exported helpers are all the Python layer needs:

- `load_optim_function(name)` — compile (if necessary) and cache the kernel.
- `get_optim_function(name)` — return the cached callable, compiling on demand.

Everything is keyed by **`name`**, which must match:

- the C++ source file `csrc/<name>.cpp`, and
- the pybind11 function name inside it (`m.def("<name>", ...)`).

The compiled module name folds in two hashes — `get_cpu_flags_hash()` (the
host's CPU feature flags) and `get_cpp_flags_hash()` (the compile flags). If you
move the cached build to a different CPU, or change `CPP_FLAGS`, the name
changes and the kernel is transparently recompiled instead of silently running
a binary that was tuned for a different machine. You normally don't touch this
file when adding an optimizer.

## Verifying That the Inner Loop Vectorized {#verify-vectorization}

To confirm the core loop was vectorized, RoundPipe **builds every kernel with
the vectorization report always on**, so every kernel compile leaves a `vec.log`.

### Find `vec.log`

The compile runs inside the Torch extensions build directory — by default
`~/.cache/torch_extensions/…/roundpipe_optim_adam_<hashes>/` (overridable with
the `TORCH_EXTENSIONS_DIR` environment variable). Because kernels are cached, the
file reflects the most recent compile. If the kernel is already built and you
want a fresh report, delete that module's build directory (or change the source,
flags, or host CPU) to trigger a recompile — for example:

```bash
python -c "import torch; from roundpipe.optim import Adam; \
p = torch.zeros(1024, requires_grad=True); p.grad = torch.randn(1024); \
Adam([p]).step()"
```

### Read the log

Look for a line pointing at the inner `for` loop in `adam.cpp`:

```
adam.cpp:20:29: optimized: loop vectorized using 32 byte vectors
```

`32 byte vectors` means 256-bit AVX; on an AVX-512 host you'll see `64 byte
vectors`.

Common reasons vectorization fails:

- **A branch in the loop** → lift the runtime flag to a `bool` template
  parameter (as with `amsgrad` / `maximize`) so the branch is compiled away.
- **Possible pointer aliasing** → mark the array pointers `__restrict`.
- **A function call in the loop** → move anything loop-invariant out of the loop,
  and make sure the called function can be inlined.

## Layer 1: The Python Optimizer Class

`roundpipe/optim/adam.py` is an ordinary `torch.optim.Optimizer` subclass. It
must match `torch.optim.Adam` exactly in interface and behavior, then forward the
computation to the kernel. It has three responsibilities.

**1. Compile the kernel when the optimizer is constructed.** The very first line
of `__init__` warms the cache so the first `step()` isn't slowed by a compile:

```python
def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), ...):
    load_optim_function("adam")
    ...
```

**2. Manage state exactly like PyTorch.** `_init_group` lazily allocates
`step`, `exp_avg`, `exp_avg_sq` (and `max_exp_avg_sq` for AMSGrad) and collects
per-parameter tensors into flat lists. `step()` iterates parameter groups and
calls the functional `adam(...)`, staying identical to PyTorch.

**3. Validate, then call the kernel.** The functional `adam()` checks that the
inputs meet the kernel's requirements before calling the C++ implementation,
typically including:

- Data types and memory layout the kernel can handle.
- Options PyTorch supports but this kernel doesn't (`fused`, `foreach`,
  `capturable`, `differentiable`) are accepted as **compatibility placeholders**
  — ignored, warned about, or rejected as appropriate — so existing training
  scripts keep working unchanged.

Finally, `__init__.py` exports the class:

```python
from .adam import Adam

__all__ = ["Adam"]
```

## Correctness: Test Against PyTorch

Every optimizer is verified against its PyTorch counterpart.
`tests/test_optim.py` builds a motley collection of tensors, runs both
optimizers for several steps under identical gradients, and checks the results.

Every optimizer needs a corresponding test that enumerates the various optimizer
arguments and calls `run_optim` to compare against the reference.

## Checklist

- `csrc/<name>.cpp`: templated branch-free inner loop, `__restrict` pointers,
  loop-invariants hoisted, OpenMP block slicing across threads.
- pybind binding named exactly `<name>`, with `py::gil_scoped_release`.
- Inner loop confirmed vectorized by checking `vec.log` (see
  [Verifying That the Inner Loop Vectorized](#verify-vectorization)).
- `<name>.py`: PyTorch-compatible API, `load_optim_function` in `__init__`,
  CPU/fp32/contiguous asserts, complex handled via `view_as_real`.
- Unsupported PyTorch options handled as compatibility placeholders.
- Exported from `optim/__init__.py`.
- Tested against the PyTorch reference across all flag combinations.
- Documented in `docs/API/optimizer.*.md`.

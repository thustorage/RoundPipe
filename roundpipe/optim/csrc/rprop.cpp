#include <omp.h>
#include <torch/extension.h>
using namespace std;
using namespace torch;

template <bool maximize>
void rprop_kernel(float *__restrict params, const float *__restrict grads,
                  float *__restrict prev, float *__restrict step_size, float f_etaminus,
                  float f_etaplus, float f_step_size_min, float f_step_size_max,
                  int64_t param_size) {
    for (int64_t i = 0; i < param_size; ++i) {
        float grad = !maximize ? grads[i] : -grads[i];
        // PyTorch keys the update off sign(grad * prev). Forming the product keeps the
        // tiny-gradient behavior identical (denormals are preserved, not flushed).
        float prod = grad * prev[i];
        // sign > 0 -> etaplus, sign < 0 -> etaminus, sign == 0 -> 1 (no change).
        float multiplier = prod > 0.0f ? f_etaplus : (prod < 0.0f ? f_etaminus : 1.0f);
        // Update and clamp the per-element step size.
        float ss =
            min(max(step_size[i] * multiplier, f_step_size_min), f_step_size_max);
        step_size[i] = ss;
        // On a sign reversal (prod < 0) the gradient is zeroed so the step is skipped
        // and prev is reset, matching PyTorch.
        grad = prod < 0.0f ? 0.0f : grad;
        float grad_sign = grad > 0.0f ? 1.0f : (grad < 0.0f ? -1.0f : 0.0f);
        params[i] -= grad_sign * ss;
        prev[i] = grad;
    }
}

// Helper to unpack boolean template parameters
template <bool... FixedBools, typename... Args>
void rprop_kernel(bool current_bool, Args... args) {
    if (current_bool) {
        rprop_kernel<FixedBools..., true>(args...);
    } else {
        rprop_kernel<FixedBools..., false>(args...);
    }
}

void rprop(vector<Tensor> params, vector<Tensor> grads, vector<Tensor> prev,
           vector<Tensor> step_size, vector<Tensor> state_steps, double etaminus,
           double etaplus, double step_size_min, double step_size_max, bool maximize) {
    vector<int64_t> numel(params.size());
    vector<float *> params_ptr(params.size());
    vector<const float *> grads_ptr(params.size());
    vector<float *> prev_ptr(params.size());
    vector<float *> step_size_ptr(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        numel[i] = params[i].numel();
        params_ptr[i] = params[i].mutable_data_ptr<float>();
        grads_ptr[i] = grads[i].const_data_ptr<float>();
        prev_ptr[i] = prev[i].mutable_data_ptr<float>();
        step_size_ptr[i] = step_size[i].mutable_data_ptr<float>();
        // The step is tracked in state for checkpoint fidelity but is unused in the
        // math.
        state_steps[i].add_(1);
    }
#pragma omp parallel
    {
        int rank = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        for (size_t i = 0; i < params.size(); ++i) {
            int64_t block_size = numel[i] / nthreads + (rank < (numel[i] % nthreads));
            int64_t offset =
                (numel[i] / nthreads) * rank + min<int64_t>(rank, numel[i] % nthreads);
            rprop_kernel(maximize, params_ptr[i] + offset, grads_ptr[i] + offset,
                         prev_ptr[i] + offset, step_size_ptr[i] + offset, etaminus,
                         etaplus, step_size_min, step_size_max, block_size);
        }
    }
}

#if PYBIND11_VERSION_HEX >= 0x020D0000 // pybind11 >= 2.13
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m, py::mod_gil_not_used())
#else
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
#endif
{
    m.def("rprop", &rprop, py::call_guard<py::gil_scoped_release>(),
          "Rprop optimizer step implementation in C++");
}

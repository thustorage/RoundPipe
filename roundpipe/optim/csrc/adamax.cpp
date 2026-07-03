#include <omp.h>
#include <torch/extension.h>
using namespace std;
using namespace torch;

template <bool maximize, bool zero_weight_decay>
void adamax_kernel(float *__restrict params, const float *__restrict grads,
                   float *__restrict exp_avg, float *__restrict exp_inf, double beta1,
                   float f_beta2, float f_eps, float f_weight_decay, float f_clr,
                   int64_t param_size) {
    float f_beta1 = beta1;
    float f_one_beta1 = 1.0 - beta1;
    for (int64_t i = 0; i < param_size; ++i) {
        float grad = !maximize ? grads[i] : -grads[i];
        if (!zero_weight_decay) {
            grad += f_weight_decay * params[i];
        }
        // Update biased first moment estimate.
        exp_avg[i] = f_beta1 * exp_avg[i] + f_one_beta1 * grad;
        // Update the exponentially weighted infinity norm.
        exp_inf[i] = max(f_beta2 * exp_inf[i], fabs(grad) + f_eps);
        // clr = lr / bias_correction1 is precomputed per-param in Python.
        params[i] -= f_clr * exp_avg[i] / exp_inf[i];
    }
}

// Helper to unpack boolean template parameters
template <bool... FixedBools, typename... Args>
void adamax_kernel(bool current_bool, Args... args) {
    if (current_bool) {
        adamax_kernel<FixedBools..., true>(args...);
    } else {
        adamax_kernel<FixedBools..., false>(args...);
    }
}

void adamax(vector<Tensor> params, vector<Tensor> grads, vector<Tensor> exp_avg,
            vector<Tensor> exp_inf, vector<Tensor> state_steps, double lr, double beta1,
            double beta2, double eps, double weight_decay, bool maximize) {
    vector<int64_t> numel(params.size());
    vector<float *> params_ptr(params.size());
    vector<const float *> grads_ptr(params.size());
    vector<float *> exp_avg_ptr(params.size());
    vector<float *> exp_inf_ptr(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        numel[i] = params[i].numel();
        params_ptr[i] = params[i].mutable_data_ptr<float>();
        grads_ptr[i] = grads[i].const_data_ptr<float>();
        exp_avg_ptr[i] = exp_avg[i].mutable_data_ptr<float>();
        exp_inf_ptr[i] = exp_inf[i].mutable_data_ptr<float>();
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
            // Per-param bias-corrected learning rate, computed in double straight from
            // the incremented step tensor and narrowed only in the kernel.
            double step = state_steps[i].item<double>();
            double clr = lr / (1.0 - pow(beta1, step));
            adamax_kernel(maximize, weight_decay == 0.0, params_ptr[i] + offset,
                          grads_ptr[i] + offset, exp_avg_ptr[i] + offset,
                          exp_inf_ptr[i] + offset, beta1, beta2, eps, weight_decay, clr,
                          block_size);
        }
    }
}

#if PYBIND11_VERSION_HEX >= 0x020D0000 // pybind11 >= 2.13
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m, py::mod_gil_not_used())
#else
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
#endif
{
    m.def("adamax", &adamax, py::call_guard<py::gil_scoped_release>(),
          "Adamax optimizer step implementation in C++");
}

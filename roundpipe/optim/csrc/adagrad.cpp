#include <omp.h>
#include <torch/extension.h>
using namespace std;
using namespace torch;

template <bool maximize, bool zero_weight_decay>
void adagrad_kernel(float *__restrict params, const float *__restrict grads,
                    float *__restrict state_sum, float f_clr, float f_eps,
                    float f_weight_decay, int64_t param_size) {
    for (int64_t i = 0; i < param_size; ++i) {
        float grad = !maximize ? grads[i] : -grads[i];
        if (!zero_weight_decay) {
            grad += f_weight_decay * params[i];
        }
        state_sum[i] += grad * grad;
        float std = sqrt(state_sum[i]) + f_eps;
        params[i] -= f_clr * grad / std;
    }
}

// Helper to unpack boolean template parameters
template <bool... FixedBools, typename... Args>
void adagrad_kernel(bool current_bool, Args... args) {
    if (current_bool) {
        adagrad_kernel<FixedBools..., true>(args...);
    } else {
        adagrad_kernel<FixedBools..., false>(args...);
    }
}

void adagrad(vector<Tensor> params, vector<Tensor> grads, vector<Tensor> state_sum,
             vector<Tensor> state_steps, double lr, double lr_decay, double eps,
             double weight_decay, bool maximize) {
    vector<int64_t> numel(params.size());
    vector<float *> params_ptr(params.size());
    vector<const float *> grads_ptr(params.size());
    vector<float *> state_sum_ptr(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        numel[i] = params[i].numel();
        params_ptr[i] = params[i].mutable_data_ptr<float>();
        grads_ptr[i] = grads[i].const_data_ptr<float>();
        state_sum_ptr[i] = state_sum[i].mutable_data_ptr<float>();
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
            double step = state_steps[i].item<double>();
            double clr = lr / (1.0 + (step - 1.0) * lr_decay);
            adagrad_kernel(maximize, weight_decay == 0.0, params_ptr[i] + offset,
                           grads_ptr[i] + offset, state_sum_ptr[i] + offset, clr, eps,
                           weight_decay, block_size);
        }
    }
}

#if PYBIND11_VERSION_HEX >= 0x020D0000 // pybind11 >= 2.13
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m, py::mod_gil_not_used())
#else
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
#endif
{
    m.def("adagrad", &adagrad, py::call_guard<py::gil_scoped_release>(),
          "Adagrad optimizer step implementation in C++");
}

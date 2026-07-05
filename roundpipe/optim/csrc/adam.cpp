#include <omp.h>
#include <torch/extension.h>
using namespace std;
using namespace torch;

template <bool amsgrad, bool maximize, bool zero_weight_decay,
          bool decoupled_weight_decay>
void adam_kernel(float *__restrict params, const float *__restrict grads,
                 float *__restrict exp_avg, float *__restrict exp_avg_sq,
                 float *__restrict max_exp_avg_sq, double lr, double beta1,
                 double beta2, float f_eps, double weight_decay, int64_t param_size,
                 int64_t step) {
    double bias_correction1 = 1.0 - pow(beta1, step);
    double bias_correction2 = 1.0 - pow(beta2, step);
    float f_beta1 = beta1;
    float f_one_beta1 = 1.0 - beta1;
    float f_beta2 = beta2;
    float f_one_beta2 = 1.0 - beta2;
    float f_weight_decay = weight_decay;
    float f_one_lr_weight_decay = 1.0 - lr * weight_decay;
    float f_step_size = lr / bias_correction1;
    float f_div_bias_correction2 = 1.0 / bias_correction2;

    for (int64_t i = 0; i < param_size; ++i) {
        float grad = !maximize ? grads[i] : -grads[i];
        if (!zero_weight_decay) {
            if (decoupled_weight_decay) {
                params[i] *= f_one_lr_weight_decay;
            } else {
                grad += f_weight_decay * params[i];
            }
        }
        exp_avg[i] = f_beta1 * exp_avg[i] + f_one_beta1 * grad;
        exp_avg_sq[i] = f_beta2 * exp_avg_sq[i] + f_one_beta2 * grad * grad;
        float denom;
        if (amsgrad) {
            max_exp_avg_sq[i] = max(max_exp_avg_sq[i], exp_avg_sq[i]);
            denom = sqrt(max_exp_avg_sq[i] * f_div_bias_correction2) + f_eps;
        } else {
            denom = sqrt(exp_avg_sq[i] * f_div_bias_correction2) + f_eps;
        }
        params[i] -= f_step_size * exp_avg[i] / denom;
    }
}

// Helper to unpack boolean template parameters
template <bool... FixedBools, typename... Args>
void adam_kernel(bool current_bool, Args... args) {
    if (current_bool) {
        adam_kernel<FixedBools..., true>(args...);
    } else {
        adam_kernel<FixedBools..., false>(args...);
    }
}

void adam(vector<Tensor> params, vector<Tensor> grads, vector<Tensor> exp_avg,
          vector<Tensor> exp_avg_sq, vector<Tensor> max_exp_avg_sq,
          vector<Tensor> state_steps, bool amsgrad, double beta1, double beta2,
          double lr, double weight_decay, double eps, bool maximize,
          bool decoupled_weight_decay) {
    for (size_t i = 0; i < params.size(); ++i) {
        state_steps[i].add_(1);
    }
#pragma omp parallel
    {
        int rank = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        for (size_t i = 0; i < params.size(); ++i) {
            int64_t numel = params[i].numel();
            int64_t block_size = numel / nthreads + (rank < (numel % nthreads));
            int64_t offset =
                (numel / nthreads) * rank + min<int64_t>(rank, numel % nthreads);
            float *params_ptr = params[i].mutable_data_ptr<float>() + offset;
            const float *grads_ptr = grads[i].const_data_ptr<float>() + offset;
            float *exp_avg_ptr = exp_avg[i].mutable_data_ptr<float>() + offset;
            float *exp_avg_sq_ptr = exp_avg_sq[i].mutable_data_ptr<float>() + offset;
            float *max_exp_avg_sq_ptr =
                amsgrad ? max_exp_avg_sq[i].mutable_data_ptr<float>() + offset
                        : nullptr;
            adam_kernel(amsgrad, maximize, weight_decay == 0.0, decoupled_weight_decay,
                        params_ptr, grads_ptr, exp_avg_ptr, exp_avg_sq_ptr,
                        max_exp_avg_sq_ptr, lr, beta1, beta2, eps, weight_decay,
                        block_size, state_steps[i].item<int64_t>());
        }
    }
}

#if PYBIND11_VERSION_HEX >= 0x020D0000 // pybind11 >= 2.13
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m, py::mod_gil_not_used())
#else
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
#endif
{
    m.def("adam", &adam, py::call_guard<py::gil_scoped_release>(),
          "Adam optimizer step implementation in C++");
}

#include <omp.h>
#include <torch/extension.h>
using namespace std;
using namespace torch;

template <bool maximize, bool zero_weight_decay, bool decoupled_weight_decay,
          bool rectified>
void radam_kernel(float *__restrict params, const float *__restrict grads,
                  float *__restrict exp_avg, float *__restrict exp_avg_sq, double lr,
                  double beta1, double beta2, float f_eps, double weight_decay,
                  double step, double rho_t, double rho_inf, double bias_correction2,
                  int64_t param_size) {
    // Per-param coefficients, computed in double from the step count and the
    // (rho_t, rho_inf, bias_correction2) passed in from the dispatch loop, mirroring
    // `_single_tensor_radam`. rect_coeff is loop-invariant (rectified only).
    double step_coeff = lr / (1.0 - pow(beta1, step));
    float f_rect_coeff = 0.0;
    if (rectified) {
        double rect = pow((rho_t - 4.0) * (rho_t - 2.0) * rho_inf /
                              ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t),
                          0.5);
        f_rect_coeff = step_coeff * rect * pow(bias_correction2, 0.5);
    }
    float f_beta2 = beta2;
    float f_one_beta1 = 1.0 - beta1;
    float f_one_beta2 = 1.0 - beta2;
    float f_weight_decay = weight_decay;
    float f_one_lr_weight_decay = 1.0 - lr * weight_decay;
    float f_step_coeff = step_coeff;

    for (int64_t i = 0; i < param_size; ++i) {
        float grad = !maximize ? grads[i] : -grads[i];
        if (!zero_weight_decay) {
            if (decoupled_weight_decay) {
                params[i] *= f_one_lr_weight_decay;
            } else {
                grad += f_weight_decay * params[i];
            }
        }
        // exp_avg.lerp_(grad, 1 - beta1); matches torch.lerp for |1-beta1| < 0.5.
        exp_avg[i] = exp_avg[i] + f_one_beta1 * (grad - exp_avg[i]);
        exp_avg_sq[i] = f_beta2 * exp_avg_sq[i] + f_one_beta2 * grad * grad;
        if (rectified) {
            params[i] -= f_rect_coeff * exp_avg[i] / (sqrt(exp_avg_sq[i]) + f_eps);
        } else {
            params[i] -= f_step_coeff * exp_avg[i];
        }
    }
}

// Helper to unpack boolean template parameters
template <bool... FixedBools, typename... Args>
void radam_kernel(bool current_bool, Args... args) {
    if (current_bool) {
        radam_kernel<FixedBools..., true>(args...);
    } else {
        radam_kernel<FixedBools..., false>(args...);
    }
}

void radam(vector<Tensor> params, vector<Tensor> grads, vector<Tensor> exp_avg,
           vector<Tensor> exp_avg_sq, vector<Tensor> state_steps, double lr,
           double beta1, double beta2, double weight_decay, double eps, bool maximize,
           bool decoupled_weight_decay) {
    // rho_inf is the maximum length of the approximated SMA (step-independent).
    double rho_inf = 2.0 / (1.0 - beta2) - 1.0;
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
            // rho_t drives the rectified template selector and is reused by the kernel
            // (together with rho_inf) to form the rectification coefficient.
            double step = state_steps[i].item<double>();
            double beta2_step = pow(beta2, step);
            double bias_correction2 = 1.0 - beta2_step;
            double rho_t = rho_inf - 2.0 * step * beta2_step / bias_correction2;
            float *params_ptr = params[i].mutable_data_ptr<float>() + offset;
            const float *grads_ptr = grads[i].const_data_ptr<float>() + offset;
            float *exp_avg_ptr = exp_avg[i].mutable_data_ptr<float>() + offset;
            float *exp_avg_sq_ptr = exp_avg_sq[i].mutable_data_ptr<float>() + offset;
            radam_kernel(maximize, weight_decay == 0.0, decoupled_weight_decay,
                         rho_t > 5.0, params_ptr, grads_ptr, exp_avg_ptr,
                         exp_avg_sq_ptr, lr, beta1, beta2, eps, weight_decay, step,
                         rho_t, rho_inf, bias_correction2, block_size);
        }
    }
}

#if PYBIND11_VERSION_HEX >= 0x020D0000 // pybind11 >= 2.13
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m, py::mod_gil_not_used())
#else
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
#endif
{
    m.def("radam", &radam, py::call_guard<py::gil_scoped_release>(),
          "RAdam optimizer step implementation in C++");
}

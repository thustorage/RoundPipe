#include <omp.h>
#include <torch/extension.h>
using namespace std;
using namespace torch;

template <bool maximize, bool zero_weight_decay, bool decoupled_weight_decay>
void nadam_kernel(float *__restrict params, const float *__restrict grads,
                  float *__restrict exp_avg, float *__restrict exp_avg_sq, double mu,
                  double mu_product, double beta1, double beta2, double lr,
                  double weight_decay, double momentum_decay, float f_eps, double step,
                  int64_t param_size) {
    // Per-param coefficients formed from the step-dependent intermediates passed in
    // from the dispatch loop, computed in double then narrowed once before the loop.
    // c1/c2 are the magnitudes of PyTorch's negated addcdiv values; the kernel
    // subtracts, which is bit-identical to adding the negated value.
    double mu_next = beta1 * (1.0 - 0.5 * pow(0.96, (step + 1.0) * momentum_decay));
    double mu_product_next = mu_product * mu_next;
    float f_c1 = lr * (1.0 - mu) / (1.0 - mu_product);
    float f_c2 = (lr * mu_next) / (1.0 - mu_product_next);
    float f_inv_bias_correction2 = 1.0 / (1.0 - pow(beta2, step));
    float f_beta1 = beta1;
    float f_one_beta1 = 1.0 - beta1;
    float f_beta2 = beta2;
    float f_one_beta2 = 1.0 - beta2;
    float f_weight_decay = weight_decay;
    float f_one_minus_lr_weight_decay = 1.0 - lr * weight_decay;

    for (int64_t i = 0; i < param_size; ++i) {
        float grad = !maximize ? grads[i] : -grads[i];
        if (!zero_weight_decay) {
            if (decoupled_weight_decay) {
                params[i] *= f_one_minus_lr_weight_decay;
            } else {
                grad += f_weight_decay * params[i];
            }
        }
        exp_avg[i] = f_beta1 * exp_avg[i] + f_one_beta1 * grad;
        exp_avg_sq[i] = f_beta2 * exp_avg_sq[i] + f_one_beta2 * grad * grad;
        float denom = sqrt(exp_avg_sq[i] * f_inv_bias_correction2) + f_eps;
        // Two separate updates mirror PyTorch's two addcdiv_ calls (each rounds).
        params[i] -= f_c1 * grad / denom;
        params[i] -= f_c2 * exp_avg[i] / denom;
    }
}

// Helper to unpack boolean template parameters
template <bool... FixedBools, typename... Args>
void nadam_kernel(bool current_bool, Args... args) {
    if (current_bool) {
        nadam_kernel<FixedBools..., true>(args...);
    } else {
        nadam_kernel<FixedBools..., false>(args...);
    }
}

void nadam(vector<Tensor> params, vector<Tensor> grads, vector<Tensor> exp_avg,
           vector<Tensor> exp_avg_sq, vector<Tensor> mu_products,
           vector<Tensor> state_steps, double beta1, double beta2, double lr,
           double weight_decay, double momentum_decay, double eps, bool maximize,
           bool decoupled_weight_decay) {
    vector<double> mus(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        // Update the mu_product running state in place (serially: exactly once per
        // parameter, not once per OpenMP thread) with the same ATen mul_ as PyTorch, so
        // its float rounding is identical. The OMP loop reads it back post-mutation.
        state_steps[i].add_(1);
        double step = state_steps[i].item<double>();
        mus[i] = beta1 * (1.0 - 0.5 * pow(0.96, step * momentum_decay));
        mu_products[i].mul_(mus[i]);
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
            nadam_kernel(maximize, weight_decay == 0.0, decoupled_weight_decay,
                         params_ptr, grads_ptr, exp_avg_ptr, exp_avg_sq_ptr, mus[i],
                         mu_products[i].item<double>(), beta1, beta2, lr, weight_decay,
                         momentum_decay, eps, state_steps[i].item<double>(),
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
    m.def("nadam", &nadam, py::call_guard<py::gil_scoped_release>(),
          "NAdam optimizer step implementation in C++");
}

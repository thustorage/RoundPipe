#include <omp.h>
#include <torch/extension.h>
using namespace std;
using namespace torch;

template <bool maximize, bool zero_weight_decay, bool centered, bool has_momentum,
          bool weight_small>
void rmsprop_kernel(float *__restrict params, const float *__restrict grads,
                    float *__restrict square_avg, float *__restrict grad_avg,
                    float *__restrict momentum_buffer, float f_lr, double alpha,
                    float f_eps, float f_weight_decay, float f_momentum,
                    int64_t param_size) {
    float f_alpha = alpha;
    float f_one_minus_alpha = 1.0 - alpha;
    for (int64_t i = 0; i < param_size; ++i) {
        float grad = !maximize ? grads[i] : -grads[i];
        if (!zero_weight_decay) {
            grad += f_weight_decay * params[i];
        }
        float sq = f_alpha * square_avg[i] + f_one_minus_alpha * grad * grad;
        square_avg[i] = sq;
        float avg;
        if (centered) {
            float ga = grad_avg[i];
            float new_ga = weight_small ? ga + f_one_minus_alpha * (grad - ga)
                                        : grad - (grad - ga) * f_alpha;
            grad_avg[i] = new_ga;
            avg = sqrt(sq - new_ga * new_ga) + f_eps;
        } else {
            avg = sqrt(sq) + f_eps;
        }
        if (has_momentum) {
            float buf = f_momentum * momentum_buffer[i] + grad / avg;
            momentum_buffer[i] = buf;
            params[i] -= f_lr * buf;
        } else {
            params[i] -= f_lr * grad / avg;
        }
    }
}

// Helper to unpack boolean template parameters
template <bool... FixedBools, typename... Args>
void rmsprop_kernel(bool current_bool, Args... args) {
    if (current_bool) {
        rmsprop_kernel<FixedBools..., true>(args...);
    } else {
        rmsprop_kernel<FixedBools..., false>(args...);
    }
}

void rmsprop(vector<Tensor> params, vector<Tensor> grads, vector<Tensor> square_avg,
             vector<Tensor> grad_avg, vector<Tensor> momentum_buffer,
             vector<Tensor> state_steps, double lr, double alpha, double eps,
             double weight_decay, double momentum, bool centered, bool maximize) {
    bool has_momentum = momentum != 0.0;
    for (size_t i = 0; i < params.size(); ++i) {
        // The step is tracked in state for checkpoint fidelity but is unused in the
        // math.
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
            float *square_avg_ptr = square_avg[i].mutable_data_ptr<float>() + offset;
            float *grad_avg_ptr =
                centered ? grad_avg[i].mutable_data_ptr<float>() + offset : nullptr;
            float *momentum_buffer_ptr =
                has_momentum ? momentum_buffer[i].mutable_data_ptr<float>() + offset
                             : nullptr;
            rmsprop_kernel(maximize, weight_decay == 0.0, centered, has_momentum,
                           1.0 - alpha < 0.5, params_ptr, grads_ptr, square_avg_ptr,
                           grad_avg_ptr, momentum_buffer_ptr, lr, alpha, eps,
                           weight_decay, momentum, block_size);
        }
    }
}

#if PYBIND11_VERSION_HEX >= 0x020D0000 // pybind11 >= 2.13
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m, py::mod_gil_not_used())
#else
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
#endif
{
    m.def("rmsprop", &rmsprop, py::call_guard<py::gil_scoped_release>(),
          "RMSprop optimizer step implementation in C++");
}

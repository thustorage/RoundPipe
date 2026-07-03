#include <omp.h>
#include <torch/extension.h>
using namespace std;
using namespace torch;

template <bool maximize, bool zero_weight_decay, bool has_momentum, bool nesterov>
void sgd_kernel(float *__restrict params, const float *__restrict grads,
                float *__restrict momentum_buffer, float lr, float momentum,
                float weight_decay, float one_minus_dampening, int64_t param_size) {
    for (int64_t i = 0; i < param_size; ++i) {
        float grad = !maximize ? grads[i] : -grads[i];
        if (!zero_weight_decay) {
            grad += weight_decay * params[i];
        }
        if (has_momentum) {
            // On the first step the buffer is zero and one_minus_dampening is 1, so
            // buf = grad, matching PyTorch (the first momentum is not damped).
            float buf = momentum * momentum_buffer[i] + one_minus_dampening * grad;
            momentum_buffer[i] = buf;
            if (nesterov) {
                grad += momentum * buf;
            } else {
                grad = buf;
            }
        }
        params[i] -= lr * grad;
    }
}

// Helper to unpack boolean template parameters
template <bool... FixedBools, typename... Args>
void sgd_kernel(bool current_bool, Args... args) {
    if (current_bool) {
        sgd_kernel<FixedBools..., true>(args...);
    } else {
        sgd_kernel<FixedBools..., false>(args...);
    }
}

void sgd(vector<Tensor> params, vector<Tensor> grads, vector<Tensor> momentum_buffer,
         vector<int64_t> is_first_step, float lr, float momentum, float dampening,
         float weight_decay, bool nesterov, bool maximize) {
    bool has_momentum = momentum != 0.0f;
    vector<int64_t> numel(params.size());
    vector<float *> params_ptr(params.size());
    vector<const float *> grads_ptr(params.size());
    vector<float *> momentum_buffer_ptr(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        numel[i] = params[i].numel();
        params_ptr[i] = params[i].mutable_data_ptr<float>();
        grads_ptr[i] = grads[i].const_data_ptr<float>();
        if (has_momentum) {
            momentum_buffer_ptr[i] = momentum_buffer[i].mutable_data_ptr<float>();
        } else {
            momentum_buffer_ptr[i] = nullptr;
        }
    }
#pragma omp parallel
    {
        int rank = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        for (size_t i = 0; i < params.size(); ++i) {
            int64_t block_size = numel[i] / nthreads + (rank < (numel[i] % nthreads));
            int64_t offset =
                (numel[i] / nthreads) * rank + min<int64_t>(rank, numel[i] % nthreads);
            float one_minus_dampening = is_first_step[i] ? 1.0f : (1.0f - dampening);
            sgd_kernel(maximize, weight_decay == 0.0f, has_momentum, nesterov,
                       params_ptr[i] + offset, grads_ptr[i] + offset,
                       momentum_buffer_ptr[i] + offset, lr, momentum, weight_decay,
                       one_minus_dampening, block_size);
        }
    }
}

#if PYBIND11_VERSION_HEX >= 0x020D0000 // pybind11 >= 2.13
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m, py::mod_gil_not_used())
#else
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
#endif
{
    m.def("sgd", &sgd, py::call_guard<py::gil_scoped_release>(),
          "SGD optimizer step implementation in C++");
}

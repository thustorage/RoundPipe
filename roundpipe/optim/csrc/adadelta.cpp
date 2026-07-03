#include <omp.h>
#include <torch/extension.h>
using namespace std;
using namespace torch;

template <bool maximize, bool zero_weight_decay>
void adadelta_kernel(float *__restrict params, const float *__restrict grads,
                     float *__restrict square_avg, float *__restrict acc_delta,
                     float f_lr, double rho, float f_eps, float f_weight_decay,
                     int64_t param_size) {
    float f_rho = rho;
    float f_one_minus_rho = 1.0 - rho;
    for (int64_t i = 0; i < param_size; ++i) {
        float grad = !maximize ? grads[i] : -grads[i];
        if (!zero_weight_decay) {
            grad += f_weight_decay * params[i];
        }
        float sq = f_rho * square_avg[i] + f_one_minus_rho * (grad * grad);
        square_avg[i] = sq;
        float std = sqrt(sq + f_eps);
        float delta = sqrt(acc_delta[i] + f_eps) / std * grad;
        acc_delta[i] = f_rho * acc_delta[i] + f_one_minus_rho * (delta * delta);
        params[i] -= f_lr * delta;
    }
}

// Helper to unpack boolean template parameters
template <bool... FixedBools, typename... Args>
void adadelta_kernel(bool current_bool, Args... args) {
    if (current_bool) {
        adadelta_kernel<FixedBools..., true>(args...);
    } else {
        adadelta_kernel<FixedBools..., false>(args...);
    }
}

void adadelta(vector<Tensor> params, vector<Tensor> grads, vector<Tensor> square_avg,
              vector<Tensor> acc_delta, vector<Tensor> state_steps, double lr,
              double rho, double eps, double weight_decay, bool maximize) {
    vector<int64_t> numel(params.size());
    vector<float *> params_ptr(params.size());
    vector<const float *> grads_ptr(params.size());
    vector<float *> square_avg_ptr(params.size());
    vector<float *> acc_delta_ptr(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        numel[i] = params[i].numel();
        params_ptr[i] = params[i].mutable_data_ptr<float>();
        grads_ptr[i] = grads[i].const_data_ptr<float>();
        square_avg_ptr[i] = square_avg[i].mutable_data_ptr<float>();
        acc_delta_ptr[i] = acc_delta[i].mutable_data_ptr<float>();
        // The step is tracked for state-dict compatibility but does not enter the math.
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
            adadelta_kernel(maximize, weight_decay == 0.0, params_ptr[i] + offset,
                            grads_ptr[i] + offset, square_avg_ptr[i] + offset,
                            acc_delta_ptr[i] + offset, lr, rho, eps, weight_decay,
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
    m.def("adadelta", &adadelta, py::call_guard<py::gil_scoped_release>(),
          "Adadelta optimizer step implementation in C++");
}

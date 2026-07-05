#include <omp.h>
#include <torch/extension.h>
using namespace std;
using namespace torch;

template <bool maximize, bool zero_weight_decay, bool mu_is_one>
void asgd_kernel(float *__restrict params, const float *__restrict grads,
                 float *__restrict ax, double lambd, double eta_value, float f_mu,
                 float f_weight_decay, int64_t param_size) {
    // eta_value and mu_value are the per-param scalars set on the PREVIOUS step
    // (step 1: eta=lr, mu=1). The decay factor is loop-invariant; compute the
    // scalar-only (1 - lambd * eta) in double, then narrow once before the loop.
    float f_decay = 1.0 - lambd * eta_value;
    float f_eta = eta_value;
    for (int64_t i = 0; i < param_size; ++i) {
        float grad = !maximize ? grads[i] : -grads[i];
        if (!zero_weight_decay) {
            grad += f_weight_decay * params[i];
        }
        // param = param * (1 - lambd * eta_value) - eta_value * grad
        // mirrors PyTorch's two ops: param.mul_(decay) then param.add_(grad, -eta).
        float p = params[i] * f_decay;
        p -= f_eta * grad;
        params[i] = p;
        // averaging: mu==1 is an exact copy (matches ax.copy_(param)); otherwise
        // ax += (param - ax) * mu_value.
        if (mu_is_one) {
            ax[i] = p;
        } else {
            ax[i] += (p - ax[i]) * f_mu;
        }
    }
}

// Helper to unpack boolean template parameters
template <bool... FixedBools, typename... Args>
void asgd_kernel(bool current_bool, Args... args) {
    if (current_bool) {
        asgd_kernel<FixedBools..., true>(args...);
    } else {
        asgd_kernel<FixedBools..., false>(args...);
    }
}

void asgd(vector<Tensor> params, vector<Tensor> grads, vector<Tensor> ax,
          vector<Tensor> mus, vector<Tensor> etas, vector<Tensor> state_steps,
          double lambd, double lr, double t0, double alpha, double weight_decay,
          bool maximize) {
    for (size_t i = 0; i < params.size(); ++i) {
        // Increment the step serially (exactly once per parameter). The eta/mu set on
        // the PREVIOUS step are read back in the OMP loop below (they are not mutated
        // until the serial write-back after the loop).
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
            // eta/mu set on the PREVIOUS step drive the current update (step 1 uses
            // eta=lr, mu=1); mu==1 selects the exact-copy averaging template.
            double mu_value = mus[i].item<double>();
            float *params_ptr = params[i].mutable_data_ptr<float>() + offset;
            const float *grads_ptr = grads[i].const_data_ptr<float>() + offset;
            float *ax_ptr = ax[i].mutable_data_ptr<float>() + offset;
            asgd_kernel(maximize, weight_decay == 0.0, mu_value == 1.0, params_ptr,
                        grads_ptr, ax_ptr, lambd, etas[i].item<double>(), mu_value,
                        weight_decay, block_size);
        }
    }
    // Recompute eta and mu for the next step and store them back (serial, matching
    // PyTorch's non-capturable branch: computed in double, rounded once into the
    // float32 state tensors).
    for (size_t i = 0; i < params.size(); ++i) {
        double step = state_steps[i].item<double>();
        etas[i].fill_(lr / pow(1.0 + lambd * lr * step, alpha));
        mus[i].fill_(1.0 / max(1.0, step - t0));
    }
}

#if PYBIND11_VERSION_HEX >= 0x020D0000 // pybind11 >= 2.13
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m, py::mod_gil_not_used())
#else
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
#endif
{
    m.def("asgd", &asgd, py::call_guard<py::gil_scoped_release>(),
          "ASGD optimizer step implementation in C++");
}

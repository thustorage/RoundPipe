#include <omp.h>
#include <torch/extension.h>
using namespace std;
using namespace torch;

// Adafactor with fp32 stepping on CPU. Unlike the previous revision (one thread
// per whole parameter), every parameter is split *within itself* across the whole
// thread team: the factored path parallelizes over the B*R rows of the matrix and
// the elementwise/reduction passes over all elements. A single large matrix now
// uses every core. The per-parameter reductions (RMS of the parameter and of the
// update, plus the column sums) force a handful of barriers per parameter, so the
// parameters are stepped one at a time -- the trade-off for within-tensor scaling.

// Process one matrix parameter (grad.dim() > 1). Tensors with more than two
// dimensions are a batch of B row-major R x C matrices; the second moment is
// factored along the last two dims, exactly like PyTorch's batched row_var @ col_var.
template <bool maximize, bool zero_weight_decay>
static void adafactor_matrix(float *__restrict param, const float *__restrict grad,
                             float *__restrict row_var, float *__restrict col_var,
                             float *__restrict cs, float *__restrict inv_rvm,
                             float *__restrict upd, int64_t B, int64_t R, int64_t C,
                             float f_omb2, double rho_t, double eps1, double eps2,
                             double d, float f_wd_factor) {
    const int64_t RC = R * C;
    const int64_t numel = B * RC;
    const float gsign = maximize ? -1.0f : 1.0f;
    float f_eps1 = eps1;
    float f_eps1sq = eps1 * eps1;
    const float f_inv_C = 1.0 / C;
    const float f_inv_R = 1.0 / R;
    const int64_t BR = B * R;
    const int64_t BC = B * C;

    // --- RMS(param) for the step size, computed before the weight decay. ---
    float param_sq = 0.0;
#pragma omp parallel for simd reduction(+ : param_sq) schedule(static)
    for (int64_t o = 0; o < numel; ++o) {
        param_sq += param[o] * param[o];
    }
    double rms_param = sqrt((double)param_sq / numel);
    double alpha = max(eps2, rms_param) * rho_t;

    // --- Row sums of grad^2 (and the row_var lerp) in parallel over rows; the
    // column sums are a reduction across rows, done with an OpenMP array-section
    // reduction so each thread accumulates its own colsum copy. ---
#pragma omp parallel for reduction(+ : cs[ : BC]) schedule(static)
    for (int64_t br = 0; br < BR; ++br) {
        const int64_t b = br / R;
        const float *__restrict grow = grad + br * C;
        float *__restrict csb = cs + b * C;
        float racc = 0.0f;
        for (int64_t c = 0; c < C; ++c) {
            float g2 = grow[c] * grow[c];
            racc += g2;
            csb[c] += g2;
        }
        row_var[br] += f_omb2 * (racc * f_inv_C - row_var[br]);
    }

    // --- col_var lerp (parallel over columns) and the per-matrix mean of row_var
    // that normalizes the outer product. ---
#pragma omp parallel
    {
#pragma omp for simd schedule(static) nowait
        for (int64_t bc = 0; bc < BC; ++bc) {
            col_var[bc] += f_omb2 * (cs[bc] * f_inv_R - col_var[bc]);
        }
#pragma omp for schedule(static)
        for (int64_t b = 0; b < B; ++b) {
            const float *__restrict rv = row_var + b * R;
            float rv_sum = 0.0;
#pragma omp simd reduction(+ : rv_sum)
            for (int64_t r = 0; r < R; ++r) {
                rv_sum += rv[r];
            }
            inv_rvm[b] = 1.0f / max(rv_sum * f_inv_R, f_eps1);
        }
    }

    // --- Build the update U = grad / sqrt(max(var_estimate, eps1^2)) and its RMS,
    // parallel over rows. var_estimate[b,r,c] = row_var[b,r] * col_var[b,c] *
    // inv_rvm[b].
    float update_sq = 0.0;
#pragma omp parallel for reduction(+ : update_sq) schedule(static)
    for (int64_t br = 0; br < BR; ++br) {
        const int64_t b = br / R;
        float pre = row_var[br] * inv_rvm[b];
        const float *__restrict grow = grad + br * C;
        const float *__restrict cvb = col_var + b * C;
        float *__restrict urow = upd + br * C;
        float usq = 0.0;
        for (int64_t c = 0; c < C; ++c) {
            float ve = pre * cvb[c];
            ve = ve > f_eps1sq ? ve : f_eps1sq;
            float u = gsign * grow[c] / sqrt(ve);
            urow[c] = u;
            usq += u * u;
        }
        update_sq += usq;
    }

    // --- Clip by RMS(update)/d and apply the (decoupled) weight decay in one pass. ---
    double rms_update = sqrt((double)update_sq / numel);
    double denom = max(1.0, rms_update / d);
    float f_coeff = -alpha / denom;
#pragma omp parallel for simd schedule(static)
    for (int64_t o = 0; o < numel; ++o) {
        param[o] =
            (zero_weight_decay ? param[o] : param[o] * f_wd_factor) + f_coeff * upd[o];
    }
}

// Process one vector parameter (grad.dim() == 1): no factorization, the full
// second moment is tracked in `variance`. Same within-tensor parallelism and the
// same elementwise-plus-two-reductions shape as the other RoundPipe optimizers.
template <bool maximize, bool zero_weight_decay>
static void adafactor_vector(float *__restrict param, const float *__restrict grad,
                             float *__restrict variance, float *__restrict upd,
                             int64_t numel, float f_omb2, double rho_t, double eps1,
                             double eps2, double d, float f_wd_factor) {
    if (numel == 0) {
        return;
    }
    float f_eps1sq = eps1 * eps1;
    const float gsign = maximize ? -1.0f : 1.0f;

    float param_sq = 0.0, update_sq = 0.0;
#pragma omp parallel for simd reduction(+ : param_sq, update_sq) schedule(static)
    for (int64_t o = 0; o < numel; ++o) {
        param_sq += param[o] * param[o];
        float g = gsign * grad[o];
        variance[o] += f_omb2 * (g * g - variance[o]);
        float u = g / sqrt(variance[o] > f_eps1sq ? variance[o] : f_eps1sq);
        upd[o] = u;
        update_sq += u * u;
    }

    double rms_param = sqrt((double)param_sq / numel);
    double alpha = max(eps2, rms_param) * rho_t;
    double rms_update = sqrt((double)update_sq / numel);
    double denom = max(1.0, rms_update / d);
    float f_coeff = -alpha / denom;
#pragma omp parallel for simd schedule(static)
    for (int64_t o = 0; o < numel; ++o) {
        param[o] =
            (zero_weight_decay ? param[o] : param[o] * f_wd_factor) + f_coeff * upd[o];
    }
}

// Unpack the runtime booleans into template parameters.
template <bool... FixedBools, typename... Args>
static void adafactor_matrix(bool current_bool, Args &&...args) {
    if (current_bool) {
        adafactor_matrix<FixedBools..., true>(std::forward<Args>(args)...);
    } else {
        adafactor_matrix<FixedBools..., false>(std::forward<Args>(args)...);
    }
}

template <bool... FixedBools, typename... Args>
static void adafactor_vector(bool current_bool, Args &&...args) {
    if (current_bool) {
        adafactor_vector<FixedBools..., true>(std::forward<Args>(args)...);
    } else {
        adafactor_vector<FixedBools..., false>(std::forward<Args>(args)...);
    }
}

// Dispatch. Parameters arrive in one flat list (like the other optimizers); the
// factored (dim > 1) and full-variance (dim == 1) cases are split here by
// inspecting each parameter's shape. row_vars/col_vars carry the factors for the
// matrices and variances the full moment for the vectors; the unused slots are
// empty placeholder tensors and are never dereferenced. The step counters are
// advanced here in C++, mirroring the other RoundPipe optimizers.
void adafactor(vector<Tensor> params, vector<Tensor> grads, vector<Tensor> row_vars,
               vector<Tensor> col_vars, vector<Tensor> variances,
               vector<Tensor> state_steps, double lr, double beta2_decay, double eps1,
               double eps2, double d, double weight_decay, bool maximize) {
    float f_wd_factor = 1.0 - (lr * weight_decay);
    bool zero_weight_decay = weight_decay == 0.0;
    vector<float> colsum;
    vector<float> inv_rvm;
    vector<float> update;

    for (size_t i = 0; i < params.size(); ++i) {
        Tensor &p = params[i];
        state_steps[i].add_(1);
        if (p.numel() == 0) {
            continue; // A zero-sized parameter is a no-op.
        }
        double s = state_steps[i].item<double>();
        double omb2 = pow(s, beta2_decay);
        double rho = min(lr, 1.0 / sqrt(s));
        if (p.dim() > 1) {
            int64_t nd = p.dim();
            int64_t Cc = p.size(nd - 1);
            int64_t Rr = p.size(nd - 2);
            int64_t rc = Rr * Cc;
            int64_t B = p.numel() / rc;
            colsum.assign(B * Cc, 0.0f);
            inv_rvm.reserve(B);
            update.reserve(p.numel());
            adafactor_matrix(
                maximize, zero_weight_decay, p.mutable_data_ptr<float>(),
                grads[i].const_data_ptr<float>(), row_vars[i].mutable_data_ptr<float>(),
                col_vars[i].mutable_data_ptr<float>(), colsum.data(), inv_rvm.data(),
                update.data(), B, Rr, Cc, omb2, rho, eps1, eps2, d, f_wd_factor);
        } else {
            update.reserve(p.numel());
            adafactor_vector(maximize, zero_weight_decay, p.mutable_data_ptr<float>(),
                             grads[i].const_data_ptr<float>(),
                             variances[i].mutable_data_ptr<float>(), update.data(),
                             p.numel(), omb2, rho, eps1, eps2, d, f_wd_factor);
        }
    }
}

#if PYBIND11_VERSION_HEX >= 0x020D0000 // pybind11 >= 2.13
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m, py::mod_gil_not_used())
#else
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
#endif
{
    m.def("adafactor", &adafactor, py::call_guard<py::gil_scoped_release>(),
          "Adafactor optimizer step implementation in C++");
}

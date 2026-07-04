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
                             int64_t B, int64_t R, int64_t C, float f_omb2,
                             double rho_t, float f_eps1, float f_eps1sq, float f_eps2,
                             float f_d, float f_wd_factor) {
    const int64_t RC = R * C;
    const int64_t numel = B * RC;
    if (numel == 0) {
        return; // A zero-sized dim leaves the empty parameter untouched.
    }
    const float gsign = maximize ? -1.0f : 1.0f;
    const float f_inv_C = 1.0f / (float)C;
    const float f_inv_R = 1.0f / (float)R;
    const int64_t BR = B * R;
    const int64_t BC = B * C;

    // --- RMS(param) for the step size, computed before the weight decay. ---
    double param_sq = 0.0;
#pragma omp parallel for simd reduction(+ : param_sq) schedule(static)
    for (int64_t o = 0; o < numel; ++o) {
        param_sq += (double)param[o] * param[o];
    }
    double rms_param = sqrt(param_sq / (double)numel);
    float f_alpha = (float)(max((double)f_eps2, rms_param) * rho_t);

    // --- Row sums of grad^2 (and the row_var lerp) in parallel over rows; the
    // column sums are a reduction across rows, done with an OpenMP array-section
    // reduction so each thread accumulates its own colsum copy. ---
    vector<float> colsum((size_t)BC, 0.0f);
    float *__restrict cs = colsum.data();
#pragma omp parallel for reduction(+ : cs[ : BC]) schedule(static)
    for (int64_t br = 0; br < BR; ++br) {
        const int64_t b = br / R;
        const float *__restrict grow = grad + br * C;
        float *__restrict csb = cs + b * C;
        float racc = 0.0f;
#pragma omp simd reduction(+ : racc)
        for (int64_t c = 0; c < C; ++c) {
            float g2 = grow[c] * grow[c];
            racc += g2;
            csb[c] += g2;
        }
        row_var[br] += f_omb2 * (racc * f_inv_C - row_var[br]);
    }

    // --- col_var lerp (parallel over columns) and the per-matrix mean of row_var
    // that normalizes the outer product. B is tiny, so inv_rvm is serial. ---
#pragma omp parallel for simd schedule(static)
    for (int64_t bc = 0; bc < BC; ++bc) {
        col_var[bc] += f_omb2 * (cs[bc] * f_inv_R - col_var[bc]);
    }
    vector<float> inv_rvm(B);
    for (int64_t b = 0; b < B; ++b) {
        const float *__restrict rv = row_var + b * R;
        double rv_sum = 0.0;
        for (int64_t r = 0; r < R; ++r) {
            rv_sum += rv[r];
        }
        inv_rvm[b] = 1.0f / max((float)(rv_sum * (double)f_inv_R), f_eps1);
    }

    // --- Build the update U = grad / sqrt(max(var_estimate, eps1^2)) and its RMS,
    // parallel over rows. var_estimate[b,r,c] = row_var[b,r] * col_var[b,c] *
    // inv_rvm[b].
    vector<float> update(numel);
    float *__restrict upd = update.data();
    double update_sq = 0.0;
#pragma omp parallel for reduction(+ : update_sq) schedule(static)
    for (int64_t br = 0; br < BR; ++br) {
        const int64_t b = br / R;
        float pre = row_var[br] * inv_rvm[b];
        const float *__restrict grow = grad + br * C;
        const float *__restrict cvb = col_var + b * C;
        float *__restrict urow = upd + br * C;
        double usq = 0.0;
#pragma omp simd reduction(+ : usq)
        for (int64_t c = 0; c < C; ++c) {
            float ve = pre * cvb[c];
            ve = ve > f_eps1sq ? ve : f_eps1sq;
            float u = gsign * grow[c] / sqrt(ve);
            urow[c] = u;
            usq += (double)u * u;
        }
        update_sq += usq;
    }

    // --- Clip by RMS(update)/d and apply the (decoupled) weight decay in one pass. ---
    double rms_update = sqrt(update_sq / (double)numel);
    float f_denom = (float)max(1.0, rms_update / (double)f_d);
    float f_coeff = -f_alpha / f_denom;
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
                             float *__restrict variance, int64_t numel, float f_omb2,
                             double rho_t, float f_eps1sq, float f_eps2, float f_d,
                             float f_wd_factor) {
    if (numel == 0) {
        return;
    }
    const float gsign = maximize ? -1.0f : 1.0f;

    double param_sq = 0.0;
#pragma omp parallel for simd reduction(+ : param_sq) schedule(static)
    for (int64_t o = 0; o < numel; ++o) {
        param_sq += (double)param[o] * param[o];
    }
    double rms_param = sqrt(param_sq / (double)numel);
    float f_alpha = (float)(max((double)f_eps2, rms_param) * rho_t);

    vector<float> update(numel);
    float *__restrict upd = update.data();
    double update_sq = 0.0;
#pragma omp parallel for simd reduction(+ : update_sq) schedule(static)
    for (int64_t o = 0; o < numel; ++o) {
        float g = gsign * grad[o];
        variance[o] += f_omb2 * (g * g - variance[o]);
        float ve = variance[o] > f_eps1sq ? variance[o] : f_eps1sq;
        float u = g / sqrt(ve);
        upd[o] = u;
        update_sq += (double)u * u;
    }

    double rms_update = sqrt(update_sq / (double)numel);
    float f_denom = (float)max(1.0, rms_update / (double)f_d);
    float f_coeff = -f_alpha / f_denom;
#pragma omp parallel for simd schedule(static)
    for (int64_t o = 0; o < numel; ++o) {
        param[o] =
            (zero_weight_decay ? param[o] : param[o] * f_wd_factor) + f_coeff * upd[o];
    }
}

// Step every parameter with the maximize / weight-decay branches resolved at
// compile time, matching the templated style of the other RoundPipe kernels.
template <bool maximize, bool zero_weight_decay>
static void adafactor_impl(const vector<float *> &mP, const vector<const float *> &mG,
                           const vector<float *> &mRV, const vector<float *> &mCV,
                           const vector<int64_t> &mB, const vector<int64_t> &mR,
                           const vector<int64_t> &mC, const vector<double> &m_omb2,
                           const vector<double> &m_rho, const vector<float *> &vP,
                           const vector<const float *> &vG, const vector<float *> &vVar,
                           const vector<int64_t> &vN, const vector<double> &v_omb2,
                           const vector<double> &v_rho, float f_eps1, float f_eps1sq,
                           float f_eps2, float f_d, float f_wd) {
    for (size_t i = 0; i < mP.size(); ++i) {
        adafactor_matrix<maximize, zero_weight_decay>(
            mP[i], mG[i], mRV[i], mCV[i], mB[i], mR[i], mC[i], (float)m_omb2[i],
            m_rho[i], f_eps1, f_eps1sq, f_eps2, f_d, f_wd);
    }
    for (size_t i = 0; i < vP.size(); ++i) {
        adafactor_vector<maximize, zero_weight_decay>(vP[i], vG[i], vVar[i], vN[i],
                                                      (float)v_omb2[i], v_rho[i],
                                                      f_eps1sq, f_eps2, f_d, f_wd);
    }
}

// Unpack the runtime booleans into template parameters.
template <bool... FixedBools, typename... Args>
static void adafactor_impl(bool current_bool, Args &&...args) {
    if (current_bool) {
        adafactor_impl<FixedBools..., true>(std::forward<Args>(args)...);
    } else {
        adafactor_impl<FixedBools..., false>(std::forward<Args>(args)...);
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
    vector<float *> mP, mRV, mCV, vP, vVar;
    vector<const float *> mG, vG;
    vector<int64_t> mB, mR, mC, vN;
    vector<double> m_omb2, m_rho, v_omb2, v_rho;

    for (size_t i = 0; i < params.size(); ++i) {
        state_steps[i].add_(1);
        double s = state_steps[i].item<double>();
        double omb2 = pow(s, beta2_decay);
        double rho = min(lr, 1.0 / sqrt(s));
        Tensor p = params[i];
        if (p.dim() > 1) {
            int64_t nd = p.dim();
            int64_t Cc = p.size(nd - 1);
            int64_t Rr = p.size(nd - 2);
            int64_t rc = Rr * Cc;
            // A zero-sized last dim makes rc == 0; guard the integer division.
            mB.push_back(rc == 0 ? 0 : p.numel() / rc);
            mR.push_back(Rr);
            mC.push_back(Cc);
            mP.push_back(params[i].mutable_data_ptr<float>());
            mG.push_back(grads[i].const_data_ptr<float>());
            mRV.push_back(row_vars[i].mutable_data_ptr<float>());
            mCV.push_back(col_vars[i].mutable_data_ptr<float>());
            m_omb2.push_back(omb2);
            m_rho.push_back(rho);
        } else {
            vN.push_back(p.numel());
            vP.push_back(params[i].mutable_data_ptr<float>());
            vG.push_back(grads[i].const_data_ptr<float>());
            vVar.push_back(variances[i].mutable_data_ptr<float>());
            v_omb2.push_back(omb2);
            v_rho.push_back(rho);
        }
    }

    float f_eps1 = eps1;
    float f_eps1sq = (float)eps1 * (float)eps1;
    float f_eps2 = eps2;
    float f_d = d;
    float f_wd = 1.0f - (float)(lr * weight_decay);
    bool zero_weight_decay = weight_decay == 0.0;

    adafactor_impl(maximize, zero_weight_decay, mP, mG, mRV, mCV, mB, mR, mC, m_omb2,
                   m_rho, vP, vG, vVar, vN, v_omb2, v_rho, f_eps1, f_eps1sq, f_eps2,
                   f_d, f_wd);
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

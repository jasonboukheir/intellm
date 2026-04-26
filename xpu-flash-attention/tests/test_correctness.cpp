#include <sycl/sycl.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>
#include <tuple>
#include "kernels/flash_attention_xe2.hpp"
#include "kernels/flash_attention_xmx.hpp"

// Naive attention on CPU in FP32 for reference.
static void naive_attention_cpu(
    const std::vector<float>& Q, const std::vector<float>& K, const std::vector<float>& V,
    std::vector<float>& O,
    int batch_heads, int seq_len, int head_dim, float scale, bool causal)
{
    std::vector<float> S(static_cast<size_t>(seq_len) * seq_len);
    for (int bh = 0; bh < batch_heads; ++bh) {
        const float* Qp = Q.data() + static_cast<size_t>(bh) * seq_len * head_dim;
        const float* Kp = K.data() + static_cast<size_t>(bh) * seq_len * head_dim;
        const float* Vp = V.data() + static_cast<size_t>(bh) * seq_len * head_dim;
        float* Op = O.data() + static_cast<size_t>(bh) * seq_len * head_dim;

        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                float sum = 0.0f;
                for (int d = 0; d < head_dim; ++d) sum += Qp[i * head_dim + d] * Kp[j * head_dim + d];
                S[i * seq_len + j] = sum * scale;
                if (causal && j > i) S[i * seq_len + j] = -1e30f;
            }
        }
        for (int i = 0; i < seq_len; ++i) {
            float m = -1e30f;
            for (int j = 0; j < seq_len; ++j) m = std::max(m, S[i * seq_len + j]);
            float sum = 0.0f;
            for (int j = 0; j < seq_len; ++j) {
                S[i * seq_len + j] = (S[i * seq_len + j] == -1e30f) ? 0.0f : std::exp(S[i * seq_len + j] - m);
                sum += S[i * seq_len + j];
            }
            float inv = 1.0f / sum;
            for (int j = 0; j < seq_len; ++j) S[i * seq_len + j] *= inv;
        }
        for (int i = 0; i < seq_len; ++i) {
            for (int d = 0; d < head_dim; ++d) {
                float sum = 0.0f;
                for (int j = 0; j < seq_len; ++j) sum += S[i * seq_len + j] * Vp[j * head_dim + d];
                Op[i * head_dim + d] = sum;
            }
        }
    }
}

struct Stats {
    float max_abs;
    float mean_abs;
    float ref_max_abs;
};

static Stats compare(const std::vector<float>& a, const std::vector<float>& b) {
    float max_abs = 0.0f, sum_abs = 0.0f, ref_max = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float d = std::abs(a[i] - b[i]);
        max_abs = std::max(max_abs, d);
        sum_abs += d;
        ref_max = std::max(ref_max, std::abs(b[i]));
    }
    return {max_abs, sum_abs / a.size(), ref_max};
}

enum class Variant { V1, V1_SLMQO, XMX };

struct Case {
    int batch;
    int heads;
    int seq_len;
    int head_dim;
    bool causal;
    Variant variant;
};

static const char* variant_name(Variant v) {
    switch (v) {
        case Variant::V1:       return "v1";
        case Variant::V1_SLMQO: return "v1.5";
        case Variant::XMX:      return "xmx";
    }
    return "?";
}

static int run_case(sycl::queue& q, const Case& c) {
    const int batch_heads = c.batch * c.heads;
    const size_t total = static_cast<size_t>(batch_heads) * c.seq_len * c.head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(c.head_dim));

    std::mt19937 gen(0xC0FFEE ^ (c.seq_len * 131 + c.head_dim));
    std::normal_distribution<float> dist(0.0f, 0.1f);

    std::vector<float> Qf(total), Kf(total), Vf(total), Oref(total);
    for (auto& x : Qf) x = dist(gen);
    for (auto& x : Kf) x = dist(gen);
    for (auto& x : Vf) x = dist(gen);

    naive_attention_cpu(Qf, Kf, Vf, Oref, batch_heads, c.seq_len, c.head_dim, scale, c.causal);

    std::vector<sycl::half> Qh(total), Kh(total), Vh(total), Oh(total);
    for (size_t i = 0; i < total; ++i) {
        Qh[i] = sycl::half(Qf[i]);
        Kh[i] = sycl::half(Kf[i]);
        Vh[i] = sycl::half(Vf[i]);
    }

    auto* dQ = sycl::malloc_device<sycl::half>(total, q);
    auto* dK = sycl::malloc_device<sycl::half>(total, q);
    auto* dV = sycl::malloc_device<sycl::half>(total, q);
    auto* dO = sycl::malloc_device<sycl::half>(total, q);
    auto* dL = sycl::malloc_device<float>(static_cast<size_t>(batch_heads) * c.seq_len, q);

    q.memcpy(dQ, Qh.data(), total * sizeof(sycl::half)).wait();
    q.memcpy(dK, Kh.data(), total * sizeof(sycl::half)).wait();
    q.memcpy(dV, Vh.data(), total * sizeof(sycl::half)).wait();

    switch (c.variant) {
        case Variant::V1:
            xfa::flash_attention_forward<sycl::half>(
                q, dQ, dK, dV, dO, dL,
                c.batch, c.heads, c.seq_len, c.head_dim, scale, c.causal).wait();
            break;
        case Variant::V1_SLMQO:
            xfa::flash_attention_forward_slmqo<sycl::half>(
                q, dQ, dK, dV, dO, dL,
                c.batch, c.heads, c.seq_len, c.head_dim, scale, c.causal).wait();
            break;
        case Variant::XMX:
            xfa::flash_attention_forward_xmx<sycl::half>(
                q, dQ, dK, dV, dO, dL,
                c.batch, c.heads, c.seq_len, c.head_dim, scale, c.causal).wait();
            break;
    }

    q.memcpy(Oh.data(), dO, total * sizeof(sycl::half)).wait();

    std::vector<float> Ogpu(total);
    for (size_t i = 0; i < total; ++i) Ogpu[i] = static_cast<float>(Oh[i]);

    auto stats = compare(Ogpu, Oref);

    sycl::free(dQ, q); sycl::free(dK, q); sycl::free(dV, q);
    sycl::free(dO, q); sycl::free(dL, q);

    // FP16 output, FP32 accum: ~5e-3 abs is generous for these small magnitudes.
    const float tol = 5e-3f;
    bool pass = (stats.max_abs <= tol);

    std::cout << "  [" << variant_name(c.variant) << "]"
              << " B=" << c.batch << " H=" << c.heads
              << " S=" << c.seq_len << " D=" << c.head_dim
              << " causal=" << (c.causal ? "y" : "n")
              << " | max|d|=" << std::scientific << std::setprecision(2) << stats.max_abs
              << " mean|d|=" << stats.mean_abs
              << " ref_max=" << stats.ref_max_abs
              << " | " << (pass ? "PASS" : "FAIL")
              << std::defaultfloat << std::endl;
    return pass ? 0 : 1;
}

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    auto dev = q.get_device();
    std::cout << "Device: " << dev.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "Local mem: " << dev.get_info<sycl::info::device::local_mem_size>() << " bytes" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    std::vector<Case> cases;
    auto seqs = {64, 128, 256, 512, 1024};
    // Variants to test. XMX (joint_matrix) is currently disabled — see
    // results/xmx_investigation.md: this Battlemage driver reports zero
    // matrix_combinations supported, so joint_matrix raises at runtime.
    std::vector<Variant> variants = {Variant::V1, Variant::V1_SLMQO};
    if (const char* e = std::getenv("XFA_TEST_XMX"); e && *e == '1') {
        variants.push_back(Variant::XMX);
    }
    for (auto v : variants) {
        for (int s : seqs) cases.push_back({1, 4, s, 128, true,  v});
        for (int s : seqs) cases.push_back({1, 4, s, 128, false, v});
        cases.push_back({2, 8, 1024, 128, true, v});
    }

    int failed = 0;
    for (const auto& c : cases) failed += run_case(q, c);

    std::cout << std::string(80, '-') << std::endl;
    if (failed == 0) {
        std::cout << "All " << cases.size() << " cases PASSED" << std::endl;
        return 0;
    }
    std::cout << failed << " / " << cases.size() << " cases FAILED" << std::endl;
    return 1;
}

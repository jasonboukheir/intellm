#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <random>
#include "kernels/flash_attention_xe2.hpp"

// Reference naive attention on CPU for correctness checking
void naive_attention_cpu(
    const std::vector<float>& Q, const std::vector<float>& K, const std::vector<float>& V,
    std::vector<float>& O,
    int seq_len, int head_dim, float scale, bool causal)
{
    // S = Q @ K^T
    std::vector<float> S(seq_len * seq_len);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            float sum = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                sum += Q[i * head_dim + d] * K[j * head_dim + d];
            }
            S[i * seq_len + j] = sum * scale;
            if (causal && j > i) {
                S[i * seq_len + j] = -1e9f;
            }
        }
    }

    // softmax per row
    for (int i = 0; i < seq_len; ++i) {
        float max_val = -1e30f;
        for (int j = 0; j < seq_len; ++j) {
            max_val = std::max(max_val, S[i * seq_len + j]);
        }
        float sum = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            S[i * seq_len + j] = std::exp(S[i * seq_len + j] - max_val);
            sum += S[i * seq_len + j];
        }
        for (int j = 0; j < seq_len; ++j) {
            S[i * seq_len + j] /= sum;
        }
    }

    // O = softmax(S) @ V
    for (int i = 0; i < seq_len; ++i) {
        for (int d = 0; d < head_dim; ++d) {
            float sum = 0.0f;
            for (int j = 0; j < seq_len; ++j) {
                sum += S[i * seq_len + j] * V[j * head_dim + d];
            }
            O[i * head_dim + d] = sum;
        }
    }
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float d = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        d = std::max(d, std::abs(a[i] - b[i]));
    }
    return d;
}

int main() {
    int seq_len = 128;
    int head_dim = 128;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    size_t n = seq_len * head_dim;

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.1f);

    std::vector<float> Q(n), K(n), V(n), O_ref(n), O_gpu(n);
    for (auto& x : Q) x = dist(gen);
    for (auto& x : K) x = dist(gen);
    for (auto& x : V) x = dist(gen);

    naive_attention_cpu(Q, K, V, O_ref, seq_len, head_dim, scale, true);

    // TODO: Run GPU kernel and compare
    // For now, just verify the reference implementation is sane
    float sum = 0.0f;
    for (auto& x : O_ref) sum += x * x;
    assert(sum > 0.0f && "Reference output should be non-zero");
    assert(!std::isnan(sum) && "Reference output should not contain NaN");

    std::cout << "Reference attention output L2 norm: " << std::sqrt(sum / n) << std::endl;
    std::cout << "PASS (reference only — GPU kernel not yet integrated)" << std::endl;

    return 0;
}

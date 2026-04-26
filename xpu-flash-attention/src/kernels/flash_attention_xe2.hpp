#pragma once

// Flash Attention v2 forward for Intel Xe2 (Battlemage) GPUs.
//
// Reference target: Intel Arc Pro B70 (Xe2 / Battlemage)
//   - 32 Xe-cores, 256 XMX engines
//   - 32GB GDDR6 @ 608 GB/s peak
//   - Sub-group sizes: 16 (preferred) or 32
//   - SLM per work-group: 64 KB
//
// Algorithm: Dao et al. (2022) tiling-based FA2 forward with online softmax.
//
// Variant v1 (this file): correctness-first pure SYCL.
//   * 1 work-item per query row inside a BLOCK_M tile
//   * K/V tiles staged through SLM and reused across all queries in the tile
//   * Q row + O accumulator + softmax stats kept in private memory (FP32)
//   * No XMX / joint_matrix yet -- explicit FMA loops, deliberately simple
//
// The XMX (joint_matrix) variant lives in flash_attention_xe2_xmx.hpp.

#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdint>
#include <vector>

namespace xfa {

struct FlashAttentionConfig {
    int head_dim = 128;
    int block_m = 64;
    int block_n = 64;
    float softmax_scale = 0.0f; // 0 = auto (1/sqrt(head_dim))
    bool causal = true;
};

template <typename T, int HEAD_DIM = 128, int BLOCK_M = 64, int BLOCK_N = 64>
class FlashAttentionKernel {
public:
    static constexpr int kHeadDim = HEAD_DIM;
    static constexpr int kBlockM = BLOCK_M;
    static constexpr int kBlockN = BLOCK_N;
    static constexpr int kSubGroupSize = 16;

    static constexpr size_t kSlmKBytes = static_cast<size_t>(BLOCK_N) * HEAD_DIM * sizeof(T);
    static constexpr size_t kSlmVBytes = static_cast<size_t>(BLOCK_N) * HEAD_DIM * sizeof(T);
    static constexpr size_t kSlmBytes  = kSlmKBytes + kSlmVBytes;
    static_assert(kSlmBytes <= 65536, "SLM tiles exceed 64KB Xe2 limit");

    static float default_scale() {
        return 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
    }

    void operator()(sycl::nd_item<2> item,
                    sycl::local_accessor<T, 1> slm_K,
                    sycl::local_accessor<T, 1> slm_V,
                    const T* __restrict Q,
                    const T* __restrict K,
                    const T* __restrict V,
                    T* __restrict O,
                    float* __restrict L,
                    int seq_len,
                    float scale,
                    bool causal) const
    {
        const int local_id = static_cast<int>(item.get_local_id(1));
        const int wg_size  = static_cast<int>(item.get_local_range(1));
        const int m_block  = static_cast<int>(item.get_group(1));
        const int q_row    = m_block * BLOCK_M + local_id;
        const bool q_valid = (q_row < seq_len);

        // 1. Load query row into private regs (promote to FP32 for accumulation accuracy)
        float q_reg[HEAD_DIM];
        if (q_valid) {
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                q_reg[d] = static_cast<float>(Q[q_row * HEAD_DIM + d]);
            }
        } else {
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) q_reg[d] = 0.0f;
        }

        // 2. Online softmax state and output accumulator
        float m_i = -INFINITY;
        float l_i = 0.0f;
        float o_reg[HEAD_DIM];
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) o_reg[d] = 0.0f;

        // 3. Determine the range of K/V blocks this work-group needs to visit.
        //    Causal mask: queries at row q only attend to keys 0..q. The whole
        //    work-group iterates up to the *last* query's last block so that we
        //    can keep the loop body branch-free per work-item.
        const int num_n_blocks = (seq_len + BLOCK_N - 1) / BLOCK_N;
        int max_n_block = num_n_blocks;
        if (causal) {
            int wg_last_q = sycl::min(seq_len - 1, m_block * BLOCK_M + BLOCK_M - 1);
            max_n_block = sycl::min(num_n_blocks, wg_last_q / BLOCK_N + 1);
        }

        // 4. Outer loop over K/V tiles
        for (int n_block = 0; n_block < max_n_block; ++n_block) {
            const int n_start = n_block * BLOCK_N;
            const int n_count = sycl::min(BLOCK_N, seq_len - n_start);

            // 4a. Cooperative load K and V tiles into SLM
            const int tile_elems = BLOCK_N * HEAD_DIM;
            for (int idx = local_id; idx < tile_elems; idx += wg_size) {
                int n = idx / HEAD_DIM;
                int d = idx % HEAD_DIM;
                if (n < n_count) {
                    slm_K[idx] = K[(n_start + n) * HEAD_DIM + d];
                    slm_V[idx] = V[(n_start + n) * HEAD_DIM + d];
                } else {
                    slm_K[idx] = T(0);
                    slm_V[idx] = T(0);
                }
            }
            sycl::group_barrier(item.get_group());

            // 4b. S = Q @ K^T (one row per work-item)
            float s_row[BLOCK_N];
            #pragma unroll
            for (int n = 0; n < BLOCK_N; ++n) {
                float dot = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dot += q_reg[d] * static_cast<float>(slm_K[n * HEAD_DIM + d]);
                }
                s_row[n] = dot * scale;
            }

            // 4c. Apply causal + bounds mask
            #pragma unroll
            for (int n = 0; n < BLOCK_N; ++n) {
                int key_idx = n_start + n;
                bool masked = (key_idx >= seq_len) || (causal && key_idx > q_row);
                if (masked) s_row[n] = -INFINITY;
            }

            // 4d. Online softmax update
            float m_new = m_i;
            #pragma unroll
            for (int n = 0; n < BLOCK_N; ++n) {
                m_new = sycl::fmax(m_new, s_row[n]);
            }
            float alpha = (m_i == -INFINITY) ? 0.0f : sycl::exp(m_i - m_new);
            float l_new = alpha * l_i;
            float p_row[BLOCK_N];
            #pragma unroll
            for (int n = 0; n < BLOCK_N; ++n) {
                p_row[n] = (s_row[n] == -INFINITY) ? 0.0f : sycl::exp(s_row[n] - m_new);
                l_new   += p_row[n];
            }

            // 4e. O = alpha * O + P @ V (V from SLM)
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                float pv = 0.0f;
                #pragma unroll
                for (int n = 0; n < BLOCK_N; ++n) {
                    pv += p_row[n] * static_cast<float>(slm_V[n * HEAD_DIM + d]);
                }
                o_reg[d] = alpha * o_reg[d] + pv;
            }

            m_i = m_new;
            l_i = l_new;

            sycl::group_barrier(item.get_group());
        }

        // 5. Final write: divide accumulator by softmax denominator
        if (q_valid) {
            float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                O[q_row * HEAD_DIM + d] = static_cast<T>(o_reg[d] * inv_l);
            }
            if (L != nullptr) {
                L[q_row] = (l_i > 0.0f) ? (m_i + sycl::log(l_i)) : -INFINITY;
            }
        }
    }
};

// Convenience launcher: lays out one work-group per (batch*head, m-block) tile.
// Pointers are assumed contiguous in [batch, head, seq, dim] layout.
template <typename T>
sycl::event flash_attention_forward(
    sycl::queue& q,
    const T* Q, const T* K, const T* V,
    T* O, float* L,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale = 0.0f,
    bool causal = true,
    const std::vector<sycl::event>& deps = {})
{
    if (scale == 0.0f) {
        scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    }

    using Kernel = FlashAttentionKernel<T, 128, 64, 64>;
    constexpr int BM = Kernel::kBlockM;
    constexpr int BN = Kernel::kBlockN;
    constexpr int HD = Kernel::kHeadDim;

    const int batch_heads  = batch_size * num_heads;
    const int num_m_blocks = (seq_len + BM - 1) / BM;

    sycl::range<2> global(batch_heads, num_m_blocks * BM);
    sycl::range<2> local(1, BM);

    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<T, 1> slm_K(BN * HD, h);
        sycl::local_accessor<T, 1> slm_V(BN * HD, h);

        Kernel kernel;
        h.parallel_for(sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> item) {
                int bh = static_cast<int>(item.get_group(0));
                size_t qkv_offset = static_cast<size_t>(bh) * seq_len * head_dim;
                size_t l_offset   = static_cast<size_t>(bh) * seq_len;
                kernel(item, slm_K, slm_V,
                       Q + qkv_offset, K + qkv_offset, V + qkv_offset,
                       O + qkv_offset,
                       (L != nullptr) ? (L + l_offset) : nullptr,
                       seq_len, scale, causal);
            });
    });
}

} // namespace xfa

#pragma once

// Flash Attention v2 forward for Intel Xe2 (Battlemage) GPUs.
//
// Reference target: Intel Arc Pro B70 (Xe2 / Battlemage)
//   - 32 Xe-cores, 256 XMX engines
//   - 32GB GDDR6 @ 608 GB/s peak
//   - Sub-group sizes: 16 (preferred) or 32
//   - SLM per work-group: up to 128 KB on Battlemage (64 KB on Xe2 client SKUs)
//
// Algorithm: Dao et al. (2022) tiling-based FA2 forward with online softmax.
//
// This file: SIMT (no XMX) reference. Two configs share one body via the
// USE_SLM_QO compile-time switch:
//
//   v1   (USE_SLM_QO = false): Q and O live in private regs (FAST kernel,
//        but spills heavily on Xe2 — HEAD_DIM=128 * 2 floats = 256+ live regs).
//
//   v1.5 (USE_SLM_QO = true):  Q and O staged into SLM. SLM cost rises to
//        BLOCK_M*HEAD_DIM*8 bytes (Q fp16 + O fp32) but per-WI register
//        pressure drops by ~256 floats — eliminates the v1 spill on Battlemage.
//
// The XMX (joint_matrix) variant lives in flash_attention_xmx.hpp; that path
// currently hits a runtime "no matrix hardware" error on this driver — see
// results/xmx_investigation.md.

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

template <typename T,
          int  HEAD_DIM   = 128,
          int  BLOCK_M    = 64,
          int  BLOCK_N    = 64,
          bool USE_SLM_QO = true>
class FlashAttentionKernel {
public:
    static constexpr int  kHeadDim     = HEAD_DIM;
    static constexpr int  kBlockM      = BLOCK_M;
    static constexpr int  kBlockN      = BLOCK_N;
    static constexpr int  kSubGroupSize = 16;
    static constexpr bool kUseSlmQO     = USE_SLM_QO;

    // SLM sizing (when USE_SLM_QO):
    //   K, V tiles                     : BLOCK_N * HEAD_DIM   (T)
    //   Q tile                         : BLOCK_M * HEAD_DIM   (T)
    //   O accumulator + per-row s scratch : BLOCK_M * HEAD_DIM + BLOCK_M * BLOCK_N (float)
    static constexpr size_t kSlmK_T = static_cast<size_t>(BLOCK_N) * HEAD_DIM;
    static constexpr size_t kSlmV_T = static_cast<size_t>(BLOCK_N) * HEAD_DIM;
    static constexpr size_t kSlmQ_T = USE_SLM_QO ? static_cast<size_t>(BLOCK_M) * HEAD_DIM : 0;
    static constexpr size_t kSlmTotalT = kSlmK_T + kSlmV_T + kSlmQ_T;

    static constexpr size_t kSlmO_F = USE_SLM_QO ? static_cast<size_t>(BLOCK_M) * HEAD_DIM : 0;
    static constexpr size_t kSlmS_F = USE_SLM_QO ? static_cast<size_t>(BLOCK_M) * BLOCK_N  : 0;
    static constexpr size_t kSlmTotalF = kSlmO_F + kSlmS_F;

    static constexpr size_t kSlmBytes =
        kSlmTotalT * sizeof(T) + kSlmTotalF * sizeof(float);
    static_assert(kSlmBytes <= 131072, "SLM exceeds 128KB Battlemage limit");

    static float default_scale() {
        return 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
    }

    void operator()(sycl::nd_item<2> item,
                    sycl::local_accessor<T, 1>     slm_t,
                    sycl::local_accessor<float, 1> slm_f,
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

        T*     slm_K = slm_t.template get_multi_ptr<sycl::access::decorated::no>().get();
        T*     slm_V = slm_K + kSlmK_T;
        T*     slm_Q = slm_V + kSlmV_T;                     // unused if USE_SLM_QO=false
        float* slm_O = slm_f.template get_multi_ptr<sycl::access::decorated::no>().get();
        float* slm_S = slm_O + kSlmO_F;                     // unused if USE_SLM_QO=false

        // -------- Q load --------
        // USE_SLM_QO=false: each WI loads its own Q row into private regs.
        // USE_SLM_QO=true : WG cooperatively loads BLOCK_M*HEAD_DIM into SLM.
        constexpr int kHD = HEAD_DIM;
        float q_reg[USE_SLM_QO ? 1 : HEAD_DIM];

        if constexpr (!USE_SLM_QO) {
            if (q_valid) {
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    q_reg[d] = static_cast<float>(Q[q_row * HEAD_DIM + d]);
                }
            } else {
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) q_reg[d] = 0.0f;
            }
        } else {
            const int q_elems = BLOCK_M * HEAD_DIM;
            for (int idx = local_id; idx < q_elems; idx += wg_size) {
                int row = idx / HEAD_DIM;
                int d   = idx % HEAD_DIM;
                int g_row = m_block * BLOCK_M + row;
                slm_Q[idx] = (g_row < seq_len) ? Q[g_row * HEAD_DIM + d] : T(0);
            }
            // O init to 0 in SLM
            const int o_elems = BLOCK_M * HEAD_DIM;
            for (int idx = local_id; idx < o_elems; idx += wg_size) {
                slm_O[idx] = 0.0f;
            }
            sycl::group_barrier(item.get_group());
        }

        // -------- O accumulator (private only when USE_SLM_QO=false) --------
        float o_reg[USE_SLM_QO ? 1 : HEAD_DIM];
        if constexpr (!USE_SLM_QO) {
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) o_reg[d] = 0.0f;
        }

        // -------- per-WI softmax state --------
        float m_i = -INFINITY;
        float l_i = 0.0f;

        // Per-row Q/O pointers in SLM (only when USE_SLM_QO)
        T*     my_q_slm = slm_Q + local_id * HEAD_DIM;
        float* my_o_slm = slm_O + local_id * HEAD_DIM;

        // -------- N-block range --------
        const int num_n_blocks = (seq_len + BLOCK_N - 1) / BLOCK_N;
        int max_n_block = num_n_blocks;
        if (causal) {
            int wg_last_q = sycl::min(seq_len - 1, m_block * BLOCK_M + BLOCK_M - 1);
            max_n_block = sycl::min(num_n_blocks, wg_last_q / BLOCK_N + 1);
        }

        // -------- Outer loop --------
        for (int n_block = 0; n_block < max_n_block; ++n_block) {
            const int n_start = n_block * BLOCK_N;
            const int n_count = sycl::min(BLOCK_N, seq_len - n_start);

            // Cooperative load K and V tiles into SLM
            const int tile_elems = BLOCK_N * HEAD_DIM;
            for (int idx = local_id; idx < tile_elems; idx += wg_size) {
                int n = idx / HEAD_DIM;
                int d = idx % HEAD_DIM;
                if (n < n_count) {
                    size_t g = static_cast<size_t>(n_start + n) * HEAD_DIM + d;
                    slm_K[idx] = K[g];
                    slm_V[idx] = V[g];
                } else {
                    slm_K[idx] = T(0);
                    slm_V[idx] = T(0);
                }
            }
            sycl::group_barrier(item.get_group());

            // Pass 1: compute s[n] for n in BLOCK_N, find row max.
            // We DON'T materialize a full s_row[BLOCK_N] when USE_SLM_QO; instead
            // we do two passes. The cost is recomputing the dot products, but
            // that's matmul work the GPU is good at — and we save BLOCK_N floats
            // of register pressure.
            float row_max = m_i;
            if constexpr (!USE_SLM_QO) {
                // Single-pass version using s_row[] (matches v1). Inner loops
                // MUST be unrolled — without #pragma unroll the compiler keeps
                // q_reg / s_row / p_row / o_reg in stack memory (huge slowdown).
                float s_row[BLOCK_N];
                #pragma unroll
                for (int n = 0; n < BLOCK_N; ++n) {
                    float dot = 0.0f;
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; ++d) {
                        dot += q_reg[d] * static_cast<float>(slm_K[n * HEAD_DIM + d]);
                    }
                    int key_idx = n_start + n;
                    bool masked = (key_idx >= seq_len) || (causal && key_idx > q_row);
                    s_row[n] = masked ? -INFINITY : dot * scale;
                    row_max  = sycl::fmax(row_max, s_row[n]);
                }

                float m_new = row_max;
                float alpha = (m_i == -INFINITY) ? 0.0f : sycl::exp(m_i - m_new);
                float l_new = alpha * l_i;
                float p_row[BLOCK_N];
                #pragma unroll
                for (int n = 0; n < BLOCK_N; ++n) {
                    p_row[n] = (s_row[n] == -INFINITY) ? 0.0f : sycl::exp(s_row[n] - m_new);
                    l_new   += p_row[n];
                }
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
            } else {
                // SLM-Q/O path: stage s_row[] in SLM scratch instead of registers.
                // Pass 1: compute s[n] -> slm_S, find row_max.
                // Pass 2: read s from SLM, exp -> p, accumulate l_new and O += p*V.
                float* my_s_slm = slm_S + local_id * BLOCK_N;
                for (int n = 0; n < BLOCK_N; ++n) {
                    float dot = 0.0f;
                    const T* k_row = slm_K + n * HEAD_DIM;
                    for (int d = 0; d < HEAD_DIM; ++d) {
                        dot += static_cast<float>(my_q_slm[d]) * static_cast<float>(k_row[d]);
                    }
                    int key_idx = n_start + n;
                    bool masked = (key_idx >= seq_len) || (causal && key_idx > q_row);
                    float s = masked ? -INFINITY : dot * scale;
                    my_s_slm[n] = s;
                    row_max = sycl::fmax(row_max, s);
                }

                float m_new = row_max;
                float alpha = (m_i == -INFINITY) ? 0.0f : sycl::exp(m_i - m_new);
                float l_new = alpha * l_i;

                // Rescale O by alpha (this WI's row only, in SLM)
                for (int d = 0; d < HEAD_DIM; ++d) my_o_slm[d] *= alpha;

                // Pass 2: O += P @ V using s from SLM scratch
                for (int n = 0; n < BLOCK_N; ++n) {
                    float s = my_s_slm[n];
                    float p = (s == -INFINITY) ? 0.0f : sycl::exp(s - m_new);
                    l_new += p;
                    const T* v_row = slm_V + n * HEAD_DIM;
                    for (int d = 0; d < HEAD_DIM; ++d) {
                        my_o_slm[d] += p * static_cast<float>(v_row[d]);
                    }
                }

                m_i = m_new;
                l_i = l_new;
            }

            sycl::group_barrier(item.get_group());
        }

        // -------- Write output --------
        if constexpr (!USE_SLM_QO) {
            if (q_valid) {
                float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
                for (int d = 0; d < HEAD_DIM; ++d) {
                    O[q_row * HEAD_DIM + d] = static_cast<T>(o_reg[d] * inv_l);
                }
                if (L != nullptr) {
                    L[q_row] = (l_i > 0.0f) ? (m_i + sycl::log(l_i)) : -INFINITY;
                }
            }
        } else {
            if (q_valid) {
                float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
                for (int d = 0; d < HEAD_DIM; ++d) {
                    O[q_row * HEAD_DIM + d] = static_cast<T>(my_o_slm[d] * inv_l);
                }
                if (L != nullptr) {
                    L[q_row] = (l_i > 0.0f) ? (m_i + sycl::log(l_i)) : -INFINITY;
                }
            }
        }
    }
};

namespace detail {

template <typename T, bool USE_SLM_QO>
sycl::event launch_fa2_simt(
    sycl::queue& q,
    const T* Q, const T* K, const T* V,
    T* O, float* L,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, bool causal,
    const std::vector<sycl::event>& deps)
{
    if (scale == 0.0f) scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    using Kernel = FlashAttentionKernel<T, 128, 64, 64, USE_SLM_QO>;
    constexpr int BM = Kernel::kBlockM;

    const int batch_heads  = batch_size * num_heads;
    const int num_m_blocks = (seq_len + BM - 1) / BM;

    sycl::range<2> global(batch_heads, num_m_blocks * BM);
    sycl::range<2> local(1, BM);

    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<T, 1>     slm_t(Kernel::kSlmTotalT, h);
        sycl::local_accessor<float, 1> slm_f(Kernel::kSlmTotalF == 0 ? 1 : Kernel::kSlmTotalF, h);

        Kernel kernel;
        h.parallel_for(sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> item) {
                int bh = static_cast<int>(item.get_group(0));
                size_t qkv_offset = static_cast<size_t>(bh) * seq_len * head_dim;
                size_t l_offset   = static_cast<size_t>(bh) * seq_len;
                kernel(item, slm_t, slm_f,
                       Q + qkv_offset, K + qkv_offset, V + qkv_offset,
                       O + qkv_offset,
                       (L != nullptr) ? (L + l_offset) : nullptr,
                       seq_len, scale, causal);
            });
    });
}

} // namespace detail

// Default v1: Q/O in private regs (matches the original implementation).
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
    return detail::launch_fa2_simt<T, /*USE_SLM_QO=*/false>(
        q, Q, K, V, O, L,
        batch_size, num_heads, seq_len, head_dim, scale, causal, deps);
}

// v1.5: Q and O staged through SLM. Eliminates the register spill that hurts
// v1 on Battlemage. Higher SLM cost, but Battlemage has 128KB SLM/WG.
template <typename T>
sycl::event flash_attention_forward_slmqo(
    sycl::queue& q,
    const T* Q, const T* K, const T* V,
    T* O, float* L,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale = 0.0f,
    bool causal = true,
    const std::vector<sycl::event>& deps = {})
{
    return detail::launch_fa2_simt<T, /*USE_SLM_QO=*/true>(
        q, Q, K, V, O, L,
        batch_size, num_heads, seq_len, head_dim, scale, causal, deps);
}

} // namespace xfa

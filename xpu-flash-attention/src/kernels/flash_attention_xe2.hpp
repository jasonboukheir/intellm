#pragma once

// Flash Attention v2 for Intel Xe2 (Battlemage) GPUs
//
// Reference implementation targeting Intel Arc Pro B70:
//   - 32 Xe-cores, 256 XMX engines
//   - 32GB GDDR6 @ 608 GB/s
//   - Xe2 architecture with XMX (matrix) engines
//
// Algorithm: Tiling-based flash attention following Dao et al. (2022),
// adapted for Xe2's sub-group (SIMD) width and SLM (shared local memory).
//
// Key Xe2 considerations:
//   - Sub-group size: 16 (preferred) or 32
//   - SLM per work-group: 64KB (Xe2)
//   - XMX engines: BF16/FP16 matrix multiply-accumulate
//   - Memory hierarchy: registers -> SLM -> L1 -> L2 -> GDDR6

#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdint>

namespace xfa {

struct FlashAttentionConfig {
    int head_dim = 128;
    int block_m = 64;     // tile size along sequence dim (queries)
    int block_n = 64;     // tile size along sequence dim (keys)
    float softmax_scale = 0.0f; // 0 = auto (1/sqrt(head_dim))
    bool causal = true;
};

// Forward declaration — implementation depends on SYCL*TLA availability
//
// Q, K, V: [batch, num_heads, seq_len, head_dim] in BF16 or FP16
// O:       [batch, num_heads, seq_len, head_dim] output
// L:       [batch, num_heads, seq_len] logsumexp for backward pass
//
// This kernel processes one (batch, head) tile per work-group.

template <typename T, int HEAD_DIM = 128, int BLOCK_M = 64, int BLOCK_N = 64>
class FlashAttentionKernel {
public:
    static constexpr int kHeadDim = HEAD_DIM;
    static constexpr int kBlockM = BLOCK_M;
    static constexpr int kBlockN = BLOCK_N;
    static constexpr int kSubGroupSize = 16; // Xe2 preferred

    // SLM tile sizes — must fit in 64KB SLM
    // Q tile: BLOCK_M * HEAD_DIM * sizeof(T)  = 64 * 128 * 2 = 16KB
    // K tile: BLOCK_N * HEAD_DIM * sizeof(T)  = 64 * 128 * 2 = 16KB
    // V tile: BLOCK_N * HEAD_DIM * sizeof(T)  = 64 * 128 * 2 = 16KB
    // Total SLM: ~48KB (fits in 64KB with room for accumulators)
    static constexpr size_t kSlmBytes = (kBlockM + 2 * kBlockN) * kHeadDim * sizeof(T);
    static_assert(kSlmBytes <= 65536, "SLM tiles exceed 64KB Xe2 limit");

    static float default_scale() {
        return 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    }

    static sycl::nd_range<2> launch_params(int batch_heads, int seq_len) {
        int num_m_blocks = (seq_len + kBlockM - 1) / kBlockM;
        return sycl::nd_range<2>(
            sycl::range<2>(batch_heads, num_m_blocks * kSubGroupSize),
            sycl::range<2>(1, kSubGroupSize)
        );
    }

    // TODO: Implement the kernel body
    // The outer loop iterates over K/V blocks (BLOCK_N tiles).
    // For each K block:
    //   1. Load K tile into SLM
    //   2. Compute S = Q @ K^T (using XMX DPAS instructions via sycl::joint_matrix)
    //   3. Apply causal mask if needed
    //   4. Online softmax: track running max and sum
    //   5. Load V tile into SLM
    //   6. Accumulate O += softmax(S) @ V
    // After all blocks: rescale O by final logsumexp

    void operator()(sycl::nd_item<2> item,
                    const T* __restrict Q,
                    const T* __restrict K,
                    const T* __restrict V,
                    T* __restrict O,
                    float* __restrict L,
                    int seq_len,
                    float scale,
                    bool causal) const {
        // Placeholder — real implementation will use SYCL*TLA primitives
        // and Xe2-specific sub-group operations
        (void)item;
        (void)Q; (void)K; (void)V; (void)O; (void)L;
        (void)seq_len; (void)scale; (void)causal;
    }
};

// Convenience launcher
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

    // TODO: Dispatch based on head_dim to select compile-time template params
    // For now, assume head_dim=128
    using Kernel = FlashAttentionKernel<T, 128, 64, 64>;
    auto nd_range = Kernel::launch_params(batch_size * num_heads, seq_len);

    Kernel kernel;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(nd_range, [=](sycl::nd_item<2> item) {
            int bh = item.get_global_id(0);
            int offset = bh * seq_len * head_dim;
            kernel(item,
                   Q + offset, K + offset, V + offset,
                   O + offset, L + bh * seq_len,
                   seq_len, scale, causal);
        });
    });
}

} // namespace xfa

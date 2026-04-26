#pragma once

// Flash Attention v2 forward for Intel Xe2 — XMX (joint_matrix / DPAS) variant.
//
// Layout for HEAD_DIM=128, FP16:
//   Work-group: BLOCK_M = 32 query rows -> 4 sub-groups, each owning 8 rows.
//   Sub-group: 16 lanes (Xe2 native), processes one DPAS tile (M=8, N=16) at a time.
//   K/V tiles staged in SLM (BLOCK_N rows, HEAD_DIM cols).
//   S/P scratch + O scratch + per-row softmax state in SLM.
//   Q is loaded once per work-group into joint_matrix tiles in registers.
//
// Per sub-group joint_matrix tiles (kept in regs across the n-loop):
//   Q_tiles[8] : (8, 16) FP16 row-major  (HEAD_DIM/16)
//   O_acc[8]   : (8, 16) FP32 accumulator (HEAD_DIM/16)
// Per iteration, transient:
//   S_acc[4]   : (8, 16) FP32 acc tiles (BLOCK_N/16) — softmax done in SLM
//   P_tiles[4] : (8, 16) FP16 A tiles loaded from SLM after softmax
//
// Softmax + alpha-rescale of O are done via cooperative SLM passes — this
// avoids needing implementation-specific per-WI joint_matrix element access.

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <cmath>
#include <cstdint>
#include <vector>

namespace xfa {

namespace exp_mat = sycl::ext::oneapi::experimental::matrix;

template <typename T, int HEAD_DIM = 128, int BLOCK_M = 32, int BLOCK_N = 64>
struct FlashAttentionXmxConfig {
    static_assert(HEAD_DIM % 16 == 0, "HEAD_DIM must be multiple of 16");
    static_assert(BLOCK_M % 8 == 0,   "BLOCK_M must be multiple of 8");
    static_assert(BLOCK_N % 16 == 0,  "BLOCK_N must be multiple of 16");

    static constexpr int kHeadDim = HEAD_DIM;
    static constexpr int kBlockM  = BLOCK_M;
    static constexpr int kBlockN  = BLOCK_N;

    static constexpr int kSubGroupSize = 16;
    static constexpr int kSGsPerWG     = BLOCK_M / 8;
    static constexpr int kWGSize       = kSGsPerWG * kSubGroupSize;

    // DPAS shape: (M=8, N=16, K=16) for FP16/BF16 -> FP32
    static constexpr int kMTile = 8;
    static constexpr int kNTile = 16;
    static constexpr int kKTile = 16;

    static constexpr int kQTiles = HEAD_DIM / kKTile;   // along K-dim of QK^T
    static constexpr int kSTiles = BLOCK_N  / kNTile;   // along N-dim (cols of S)
    static constexpr int kOTiles = HEAD_DIM / kNTile;   // along N-dim (cols of O)

    // SLM layout (in elements of T unless noted):
    //   K tile        : BLOCK_N * HEAD_DIM   (T)
    //   V tile        : BLOCK_N * HEAD_DIM   (T)
    //   P scratch     : BLOCK_M * BLOCK_N    (T)         FP16
    //   S scratch     : BLOCK_M * BLOCK_N    (float)     FP32, aliased with O scratch
    //   O scratch     : BLOCK_M * HEAD_DIM   (float)     FP32
    //   stats (m,l)   : BLOCK_M * 2          (float)
    //
    // We keep S and O scratches separate to avoid awkward aliasing during the alpha rescale.
    static constexpr size_t kSlmK_T   = static_cast<size_t>(BLOCK_N) * HEAD_DIM;
    static constexpr size_t kSlmV_T   = static_cast<size_t>(BLOCK_N) * HEAD_DIM;
    static constexpr size_t kSlmP_T   = static_cast<size_t>(BLOCK_M) * BLOCK_N;
    static constexpr size_t kSlmTotalT = kSlmK_T + kSlmV_T + kSlmP_T;

    static constexpr size_t kSlmS_F   = static_cast<size_t>(BLOCK_M) * BLOCK_N;
    static constexpr size_t kSlmO_F   = static_cast<size_t>(BLOCK_M) * HEAD_DIM;
    static constexpr size_t kSlmStats = static_cast<size_t>(BLOCK_M) * 2;
    static constexpr size_t kSlmTotalF = kSlmS_F + kSlmO_F + kSlmStats;
};

template <typename T, typename Cfg>
inline void flash_attention_xmx_kernel(
    sycl::nd_item<2> item,
    sycl::local_accessor<T, 1>      slm_t,
    sycl::local_accessor<float, 1>  slm_f,
    const T* __restrict Q, const T* __restrict K, const T* __restrict V,
    T* __restrict O, float* __restrict L,
    int seq_len, float scale, bool causal)
{
    namespace mat = exp_mat;
    using sg_t   = sycl::sub_group;
    using AccT   = mat::joint_matrix<sg_t, float, mat::use::accumulator, Cfg::kMTile, Cfg::kNTile>;
    using ATile  = mat::joint_matrix<sg_t, T, mat::use::a, Cfg::kMTile, Cfg::kKTile, mat::layout::row_major>;
    using BTile  = mat::joint_matrix<sg_t, T, mat::use::b, Cfg::kKTile, Cfg::kNTile, mat::layout::row_major>;
    using BTileT = mat::joint_matrix<sg_t, T, mat::use::b, Cfg::kKTile, Cfg::kNTile, mat::layout::col_major>;

    constexpr int HD = Cfg::kHeadDim;
    constexpr int BM = Cfg::kBlockM;
    constexpr int BN = Cfg::kBlockN;
    constexpr int M  = Cfg::kMTile;
    constexpr int N  = Cfg::kNTile;
    constexpr int KT = Cfg::kKTile;

    sg_t sg = item.get_sub_group();
    const int sg_id   = static_cast<int>(sg.get_group_linear_id());
    const int local   = static_cast<int>(item.get_local_linear_id());
    const int wg_size = Cfg::kWGSize;
    const int m_block = static_cast<int>(item.get_group(1));
    const int sg_row0 = m_block * BM + sg_id * M;

    // ---- SLM partitions ----------------------------------------------------
    auto slm_t_ptr = slm_t.template get_multi_ptr<sycl::access::decorated::no>().get();
    auto slm_f_ptr = slm_f.template get_multi_ptr<sycl::access::decorated::no>().get();
    T*     slm_K = slm_t_ptr;
    T*     slm_V = slm_K + Cfg::kSlmK_T;
    T*     slm_P = slm_V + Cfg::kSlmV_T;
    float* slm_S = slm_f_ptr;
    float* slm_O = slm_S + Cfg::kSlmS_F;
    float* slm_m = slm_O + Cfg::kSlmO_F;
    float* slm_l = slm_m + BM;

    // ---- Initialize per-row softmax state and SLM O accumulator -----------
    for (int idx = local; idx < BM; idx += wg_size) {
        slm_m[idx] = -INFINITY;
        slm_l[idx] = 0.0f;
    }
    for (int idx = local; idx < BM * HD; idx += wg_size) {
        slm_O[idx] = 0.0f;
    }

    // ---- Load Q tiles into registers (per sub-group) ----------------------
    ATile q_tiles[Cfg::kQTiles];
    {
        const int safe_row0 = sycl::min(sg_row0, sycl::max(0, seq_len - M));
        const T* base = Q + static_cast<size_t>(safe_row0) * HD;
        for (int t = 0; t < Cfg::kQTiles; ++t) {
            mat::joint_matrix_load(sg, q_tiles[t],
                sycl::address_space_cast<sycl::access::address_space::global_space,
                                         sycl::access::decorated::no, T>(
                    const_cast<T*>(base + t * KT)),
                HD);
        }
    }
    sycl::group_barrier(item.get_group());

    // ---- Determine n-block range -----------------------------------------
    const int num_n_blocks = (seq_len + BN - 1) / BN;
    int max_n_block = num_n_blocks;
    if (causal) {
        int wg_last_q = sycl::min(seq_len - 1, m_block * BM + BM - 1);
        max_n_block = sycl::min(num_n_blocks, wg_last_q / BN + 1);
    }

    // ---- Outer loop -------------------------------------------------------
    for (int n_block = 0; n_block < max_n_block; ++n_block) {
        const int n_start = n_block * BN;
        const int n_count = sycl::min(BN, seq_len - n_start);

        // 1. Cooperative load K and V tiles into SLM.
        const int tile_elems = BN * HD;
        for (int idx = local; idx < tile_elems; idx += wg_size) {
            int n = idx / HD;
            int d = idx % HD;
            if (n < n_count) {
                size_t g = static_cast<size_t>(n_start + n) * HD + d;
                slm_K[idx] = K[g];
                slm_V[idx] = V[g];
            } else {
                slm_K[idx] = T(0);
                slm_V[idx] = T(0);
            }
        }
        sycl::group_barrier(item.get_group());

        // 2. S = Q @ K^T via DPAS. Each sub-group computes M=8 rows of S
        //    (rows sg_id*8 .. sg_id*8+8 within the BM tile), all BN cols.
        AccT s_acc[Cfg::kSTiles];
        for (int t = 0; t < Cfg::kSTiles; ++t) mat::joint_matrix_fill(sg, s_acc[t], 0.0f);

        // K^T (HEAD_DIM x BLOCK_N) loaded as col_major B with stride HEAD_DIM
        // from SLM: address(k_in, j_in) = base + j_in*HD + k_in
        //                                = (n_tile*N + j_in)*HD + (k_tile*KT + k_in)
        // base for (n_tile, k_tile) = slm_K + n_tile*N*HD + k_tile*KT
        for (int k_tile = 0; k_tile < Cfg::kQTiles; ++k_tile) {
            for (int n_tile = 0; n_tile < Cfg::kSTiles; ++n_tile) {
                BTileT k_b;
                T* k_ptr = slm_K + static_cast<size_t>(n_tile * N) * HD + (k_tile * KT);
                mat::joint_matrix_load(sg, k_b,
                    sycl::address_space_cast<sycl::access::address_space::local_space,
                                             sycl::access::decorated::no, T>(k_ptr),
                    HD);
                mat::joint_matrix_mad(sg, s_acc[n_tile], q_tiles[k_tile], k_b, s_acc[n_tile]);
            }
        }

        // 3. Store S to SLM (FP32, row-major).
        for (int t = 0; t < Cfg::kSTiles; ++t) {
            float* s_ptr = slm_S + static_cast<size_t>(sg_id * M) * BN + (t * N);
            mat::joint_matrix_store(sg, s_acc[t],
                sycl::address_space_cast<sycl::access::address_space::local_space,
                                         sycl::access::decorated::no, float>(s_ptr),
                BN, mat::layout::row_major);
        }

        // 4. Store O acc tiles to SLM (FP32, row-major) so we can rescale by alpha per row.
        AccT o_acc[Cfg::kOTiles];  // we always reload from SLM below — declare here for clarity
        // (We'll fill o_acc fresh from SLM after rescale; declare to keep scope tidy.)
        sycl::group_barrier(item.get_group());

        // 5. Cooperative softmax in SLM. Each WI handles whole rows it owns.
        //    32 rows, 64 work-items: rows 0..31 owned by WIs 0..31; WIs 32..63 idle here.
        if (local < BM) {
            int row = local;
            int q_idx = m_block * BM + row;
            float* s_row = slm_S + static_cast<size_t>(row) * BN;

            // 5a. Apply scale + causal/bounds mask, find row max
            float local_max = -INFINITY;
            for (int j = 0; j < BN; ++j) {
                float v = s_row[j] * scale;
                int key_idx = n_start + j;
                bool masked = (key_idx >= seq_len) || (q_idx >= seq_len) ||
                              (causal && key_idx > q_idx);
                if (masked) v = -INFINITY;
                s_row[j] = v;
                if (v > local_max) local_max = v;
            }

            // 5b. Online softmax update
            float m_old = slm_m[row];
            float l_old = slm_l[row];
            float m_new = sycl::fmax(m_old, local_max);
            float alpha = (m_old == -INFINITY) ? 0.0f : sycl::exp(m_old - m_new);

            float row_sum = 0.0f;
            // Convert to P (FP16) and write to slm_P; also accumulate row_sum
            T* p_row = slm_P + static_cast<size_t>(row) * BN;
            for (int j = 0; j < BN; ++j) {
                float p = (s_row[j] == -INFINITY) ? 0.0f : sycl::exp(s_row[j] - m_new);
                p_row[j] = static_cast<T>(p);
                row_sum += p;
            }

            slm_m[row] = m_new;
            slm_l[row] = alpha * l_old + row_sum;

            // 5c. Rescale this row of O by alpha (in place).
            float* o_row = slm_O + static_cast<size_t>(row) * HD;
            for (int d = 0; d < HD; ++d) o_row[d] *= alpha;
        }
        sycl::group_barrier(item.get_group());

        // 6. Reload O from SLM into accumulator tiles, then DPAS-add P @ V.
        for (int t = 0; t < Cfg::kOTiles; ++t) {
            float* o_ptr = slm_O + static_cast<size_t>(sg_id * M) * HD + (t * N);
            mat::joint_matrix_load(sg, o_acc[t],
                sycl::address_space_cast<sycl::access::address_space::local_space,
                                         sycl::access::decorated::no, float>(o_ptr),
                HD, mat::layout::row_major);
        }

        // 7. P @ V — load P as A row-major from SLM, V as B row-major from SLM.
        //    P sub-tile for sub-group: rows sg_id*M..sg_id*M+M, cols k_tile*KT..k_tile*KT+KT
        //    V sub-tile: rows k_tile*KT..k_tile*KT+KT, cols n_tile*N..n_tile*N+N
        ATile p_tiles[Cfg::kSTiles];
        for (int k_tile = 0; k_tile < Cfg::kSTiles; ++k_tile) {
            T* p_ptr = slm_P + static_cast<size_t>(sg_id * M) * BN + (k_tile * KT);
            mat::joint_matrix_load(sg, p_tiles[k_tile],
                sycl::address_space_cast<sycl::access::address_space::local_space,
                                         sycl::access::decorated::no, T>(p_ptr),
                BN);
        }
        for (int n_tile = 0; n_tile < Cfg::kOTiles; ++n_tile) {
            for (int k_tile = 0; k_tile < Cfg::kSTiles; ++k_tile) {
                BTile v_b;
                T* v_ptr = slm_V + static_cast<size_t>(k_tile * KT) * HD + (n_tile * N);
                mat::joint_matrix_load(sg, v_b,
                    sycl::address_space_cast<sycl::access::address_space::local_space,
                                             sycl::access::decorated::no, T>(v_ptr),
                    HD);
                mat::joint_matrix_mad(sg, o_acc[n_tile], p_tiles[k_tile], v_b, o_acc[n_tile]);
            }
        }

        // 8. Store updated O back to SLM (so the next iter can rescale by next alpha).
        for (int t = 0; t < Cfg::kOTiles; ++t) {
            float* o_ptr = slm_O + static_cast<size_t>(sg_id * M) * HD + (t * N);
            mat::joint_matrix_store(sg, o_acc[t],
                sycl::address_space_cast<sycl::access::address_space::local_space,
                                         sycl::access::decorated::no, float>(o_ptr),
                HD, mat::layout::row_major);
        }
        sycl::group_barrier(item.get_group());
    }

    // ---- Final normalize and write back -----------------------------------
    if (local < BM) {
        int row = local;
        int q_row = m_block * BM + row;
        float l   = slm_l[row];
        float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
        if (q_row < seq_len) {
            float* o_row = slm_O + static_cast<size_t>(row) * HD;
            for (int d = 0; d < HD; ++d) {
                O[static_cast<size_t>(q_row) * HD + d] = static_cast<T>(o_row[d] * inv);
            }
            if (L != nullptr) {
                L[q_row] = (l > 0.0f) ? (slm_m[row] + sycl::log(l)) : -INFINITY;
            }
        }
    }
}

template <typename T>
sycl::event flash_attention_forward_xmx(
    sycl::queue& q,
    const T* Q, const T* K, const T* V,
    T* O, float* L,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale = 0.0f,
    bool causal = true,
    const std::vector<sycl::event>& deps = {})
{
    if (scale == 0.0f) scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    using Cfg = FlashAttentionXmxConfig<T, 128, 32, 64>;
    constexpr int BM = Cfg::kBlockM;

    const int batch_heads  = batch_size * num_heads;
    const int num_m_blocks = (seq_len + BM - 1) / BM;

    sycl::range<2> global(batch_heads, num_m_blocks * Cfg::kWGSize);
    sycl::range<2> local(1, Cfg::kWGSize);

    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<T, 1>     slm_t(Cfg::kSlmTotalT, h);
        sycl::local_accessor<float, 1> slm_f(Cfg::kSlmTotalF, h);

        h.parallel_for(sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(Cfg::kSubGroupSize)]] {
                int bh = static_cast<int>(item.get_group(0));
                size_t qkv_offset = static_cast<size_t>(bh) * seq_len * head_dim;
                size_t l_offset   = static_cast<size_t>(bh) * seq_len;
                flash_attention_xmx_kernel<T, Cfg>(
                    item, slm_t, slm_f,
                    Q + qkv_offset, K + qkv_offset, V + qkv_offset,
                    O + qkv_offset,
                    (L != nullptr) ? (L + l_offset) : nullptr,
                    seq_len, scale, causal);
            });
    });
}

} // namespace xfa

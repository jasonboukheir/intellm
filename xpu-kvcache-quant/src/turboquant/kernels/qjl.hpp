#pragma once

// QJL (Quantized Johnson-Lindenstrauss) Projection for TurboQuant
//
// After quantization, some information is lost. QJL captures the sign of the
// quantization residual using a random projection, providing 1 extra bit per
// element as an error-correction signal during dequantization.
//
// This is based on the Johnson-Lindenstrauss lemma: random projections preserve
// distances approximately. The sign of the projection captures the dominant
// direction of the residual error.
//
// The projection matrix is a random {+1, -1} matrix generated from a fixed seed,
// so it doesn't need to be stored — just regenerated on both quantize and dequantize.

#include <sycl/sycl.hpp>
#include <cstdint>

namespace kvq::turboquant {

// Simple PRNG for generating {+1, -1} projection entries from a seed
// Uses xorshift64 for speed — quality doesn't matter much since JL projections
// are robust to the choice of random matrix.
struct QJLProjection {
    static uint64_t xorshift64(uint64_t state) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    }

    // Get the sign (+1 or -1) for projection matrix entry (row, col)
    static float sign(uint64_t seed, int row, int col, int dim) {
        uint64_t state = seed ^ (static_cast<uint64_t>(row) * dim + col);
        state = xorshift64(state);
        return (state & 1) ? 1.0f : -1.0f;
    }

    // Compute QJL sign bit for a residual vector
    // residual[dim] -> 1 bit (sign of random projection)
    static uint8_t compute_sign_bit(
        const float* residual,
        int dim,
        int projection_idx,
        uint64_t seed)
    {
        float proj = 0.0f;
        for (int d = 0; d < dim; ++d) {
            proj += residual[d] * sign(seed, projection_idx, d, dim);
        }
        return proj >= 0.0f ? 1 : 0;
    }
};

// SYCL kernel: Compute QJL sign bits for a batch of residual vectors
// Input:  float[batch_size, dim] (quantization residuals)
// Output: uint8_t[batch_size, packed_bits] (1 bit per element, packed into bytes)
sycl::event compute_qjl_bits(
    sycl::queue& q,
    const float* residuals,    // [batch_size, dim]
    uint8_t* sign_bits,        // [batch_size, (dim+7)/8]
    int batch_size,
    int dim,
    uint64_t seed = 0xDEADBEEF42ULL,
    const std::vector<sycl::event>& deps = {});

} // namespace kvq::turboquant

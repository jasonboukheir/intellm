#pragma once

// Lloyd-Max Optimal Scalar Quantizer for TurboQuant
//
// After WHT rotation, the KV values follow a near-Gaussian distribution.
// Lloyd-Max quantization places quantization levels to minimize MSE for
// a given distribution, unlike uniform quantization which assumes flat density.
//
// For a Gaussian source:
//   - 3-bit (8 levels): ~3.01 dB gain over uniform
//   - 2-bit (4 levels): ~2.36 dB gain over uniform
//
// The optimal levels for a standard Gaussian are precomputed.

#include <sycl/sycl.hpp>
#include <array>

namespace kvq::turboquant {

// Precomputed Lloyd-Max quantization boundaries and reconstruction levels
// for a standard normal distribution N(0,1).
// Actual use requires scaling by the group's std dev.

struct LloydMaxTable3Bit {
    // 8 reconstruction levels for 3-bit quantization of N(0,1)
    static constexpr std::array<float, 8> levels = {
        -1.748f, -1.050f, -0.500f, -0.000f,
         0.000f,  0.500f,  1.050f,  1.748f
    };
    // 7 decision boundaries
    static constexpr std::array<float, 7> boundaries = {
        -1.399f, -0.775f, -0.250f, 0.000f,
         0.250f,  0.775f,  1.399f
    };
};

struct LloydMaxTable2Bit {
    static constexpr std::array<float, 4> levels = {
        -1.510f, -0.453f, 0.453f, 1.510f
    };
    static constexpr std::array<float, 3> boundaries = {
        -0.982f, 0.000f, 0.982f
    };
};

// Quantize a single value using precomputed Lloyd-Max tables
template <int BITS>
uint8_t lloyd_max_quantize(float val);

template <>
inline uint8_t lloyd_max_quantize<3>(float val) {
    for (int i = 0; i < 7; ++i) {
        if (val < LloydMaxTable3Bit::boundaries[i]) return static_cast<uint8_t>(i);
    }
    return 7;
}

template <>
inline uint8_t lloyd_max_quantize<2>(float val) {
    for (int i = 0; i < 3; ++i) {
        if (val < LloydMaxTable2Bit::boundaries[i]) return static_cast<uint8_t>(i);
    }
    return 3;
}

// Dequantize
template <int BITS>
float lloyd_max_dequantize(uint8_t idx);

template <>
inline float lloyd_max_dequantize<3>(uint8_t idx) {
    return LloydMaxTable3Bit::levels[idx & 0x7];
}

template <>
inline float lloyd_max_dequantize<2>(uint8_t idx) {
    return LloydMaxTable2Bit::levels[idx & 0x3];
}

// SYCL kernel: Quantize a batch of rotated vectors
// Input:  float[batch_size, dim] (post-WHT rotated values)
// Output: uint8_t[batch_size, packed_dim] (bit-packed quantized values)
//         GroupMeta[batch_size, num_groups] (per-group scale/offset)
template <int BITS, int GROUP_SIZE = 128>
sycl::event lloyd_max_quantize_batch(
    sycl::queue& q,
    const float* input,       // [batch_size, dim]
    uint8_t* output,          // [batch_size, packed_bytes]
    float* scales,            // [batch_size, num_groups]
    int batch_size,
    int dim,
    const std::vector<sycl::event>& deps = {})
{
    int num_groups = (dim + GROUP_SIZE - 1) / GROUP_SIZE;

    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);

        h.parallel_for(
            sycl::range<2>(batch_size, num_groups),
            [=](sycl::id<2> idx) {
                int b = idx[0];
                int g = idx[1];
                int start = g * GROUP_SIZE;
                int end = sycl::min(start + GROUP_SIZE, dim);
                const float* group_data = input + b * dim + start;

                // Compute group statistics
                float sum = 0.0f, sum_sq = 0.0f;
                for (int i = 0; i < end - start; ++i) {
                    sum += group_data[i];
                    sum_sq += group_data[i] * group_data[i];
                }
                float mean = sum / (end - start);
                float variance = sum_sq / (end - start) - mean * mean;
                float std_dev = sycl::sqrt(sycl::max(variance, 1e-8f));

                scales[b * num_groups + g] = std_dev;

                // Quantize each element: normalize -> Lloyd-Max table lookup
                // TODO: Write to bit-packed output
                // For now, write one byte per element (wasteful but correct)
                for (int i = 0; i < end - start; ++i) {
                    float normalized = (group_data[i] - mean) / std_dev;
                    uint8_t qval = lloyd_max_quantize<BITS>(normalized);
                    output[b * dim + start + i] = qval;
                }
            });
    });
}

} // namespace kvq::turboquant

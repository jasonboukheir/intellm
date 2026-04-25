#pragma once

// Walsh-Hadamard Transform for TurboQuant rotation step
//
// The WHT distributes information uniformly across all dimensions of the KV
// vectors before quantization. This ensures no single dimension carries
// disproportionate information that would be destroyed by uniform quantization.
//
// For d=128 head dimensions, the WHT is a 128x128 orthogonal matrix that can be
// applied in O(d log d) = 896 operations via the butterfly factorization,
// vs O(d^2) = 16,384 for naive matrix multiply.
//
// SYCL implementation uses sub-group shuffles for the butterfly steps.

#include <sycl/sycl.hpp>
#include <cmath>

namespace kvq::turboquant {

// In-place Walsh-Hadamard Transform via iterative butterfly
// Input: float[dim] where dim is power of 2
// Output: transformed float[dim], scaled by 1/sqrt(dim)
template <int DIM>
struct WalshHadamardTransform {
    static_assert((DIM & (DIM - 1)) == 0, "DIM must be power of 2");
    static constexpr int kLogDim = __builtin_ctz(DIM);
    static constexpr float kScale = 1.0f / std::sqrt(static_cast<float>(DIM));

    // CPU reference implementation
    static void apply_cpu(float* data) {
        for (int step = 1; step < DIM; step <<= 1) {
            for (int i = 0; i < DIM; i += step * 2) {
                for (int j = i; j < i + step; ++j) {
                    float a = data[j];
                    float b = data[j + step];
                    data[j] = a + b;
                    data[j + step] = a - b;
                }
            }
        }
        for (int i = 0; i < DIM; ++i) {
            data[i] *= kScale;
        }
    }

    // SYCL kernel: transform a batch of vectors
    // Each work-item handles one vector element, using sub-group operations
    // for the butterfly communication pattern.
    //
    // TODO: For DIM > sub_group_size, need SLM-based implementation
    // For DIM=128 on Xe2 (sub_group_size=16), we need 8 elements per work-item
    // with SLM shuffles for cross-item butterfly steps.
    static sycl::event apply_batch(
        sycl::queue& q,
        float* data,           // [batch_size, DIM]
        int batch_size,
        const std::vector<sycl::event>& deps = {})
    {
        return q.submit([&](sycl::handler& h) {
            h.depends_on(deps);
            // Each work-group handles one vector
            // Work-group size = DIM (or padded to sub-group boundary)
            constexpr int WG_SIZE = DIM >= 16 ? DIM : 16;
            sycl::local_accessor<float, 1> slm(sycl::range<1>(DIM), h);

            h.parallel_for(
                sycl::nd_range<1>(batch_size * WG_SIZE, WG_SIZE),
                [=](sycl::nd_item<1> item) {
                    int batch_idx = item.get_group(0);
                    int local_id = item.get_local_id(0);

                    if (local_id < DIM) {
                        slm[local_id] = data[batch_idx * DIM + local_id];
                    }
                    sycl::group_barrier(item.get_group());

                    // Butterfly steps
                    for (int step = 1; step < DIM; step <<= 1) {
                        if (local_id < DIM) {
                            int pair = local_id ^ step;
                            float a = slm[local_id];
                            float b = slm[pair];
                            sycl::group_barrier(item.get_group());
                            if (local_id < pair) {
                                slm[local_id] = a + b;
                                slm[pair] = a - b;
                            }
                        }
                        sycl::group_barrier(item.get_group());
                    }

                    if (local_id < DIM) {
                        data[batch_idx * DIM + local_id] = slm[local_id] * kScale;
                    }
                });
        });
    }
};

} // namespace kvq::turboquant

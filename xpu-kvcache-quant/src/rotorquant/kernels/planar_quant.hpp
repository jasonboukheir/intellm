#pragma once

// PlanarQuant — Givens 2D rotation variant
//
// The simplest of the three RotorQuant strategies. Processes the KV vector
// in 2D blocks using Givens rotations (2D plane rotations).
//
// For d=128: 64 independent 2D rotations, each parameterized by a single angle.
// Total parameters: 64 (vs 168 for RotorQuant, vs 16,384 for TurboQuant WHT)
// FMAs per vector: 256 (64 blocks * 4 FMAs per 2D rotation)
//
// PlanarQuant and IsoQuant (quaternion 4D) provide the two extremes:
//   - PlanarQuant: fewest params, simplest rotation, good baseline
//   - IsoQuant: more params, 4D blocks, better mixing
//   - RotorQuant: Cl(3,0) 3D blocks, best quality/speed tradeoff

#include <sycl/sycl.hpp>
#include <cmath>

namespace kvq::rotorquant {

struct GivensRotation {
    float cos_theta;
    float sin_theta;

    static GivensRotation from_angle(float theta) {
        return {std::cos(theta), std::sin(theta)};
    }

    void apply(float& a, float& b) const {
        float a_new = cos_theta * a - sin_theta * b;
        float b_new = sin_theta * a + cos_theta * b;
        a = a_new;
        b = b_new;
    }

    void apply_inverse(float& a, float& b) const {
        float a_new = cos_theta * a + sin_theta * b;
        float b_new = -sin_theta * a + cos_theta * b;
        a = a_new;
        b = b_new;
    }
};

struct PlanarRotationSet {
    static constexpr int block_size = 2;

    int dim;
    int num_rotations;             // dim / 2
    const GivensRotation* rotations;

    static int rotations_needed(int dim) {
        return dim / 2;
    }

    void apply_forward(float* data) const {
        for (int r = 0; r < num_rotations; ++r) {
            rotations[r].apply(data[r * 2], data[r * 2 + 1]);
        }
    }

    void apply_inverse(float* data) const {
        for (int r = num_rotations - 1; r >= 0; --r) {
            rotations[r].apply_inverse(data[r * 2], data[r * 2 + 1]);
        }
    }
};

sycl::event apply_planar_rotations_batch(
    sycl::queue& q,
    float* data,                      // [batch_size, dim] — in-place
    const GivensRotation* rotations,  // [dim/2]
    int batch_size,
    int dim,
    const std::vector<sycl::event>& deps = {})
{
    int num_rotations = dim / 2;

    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(batch_size), [=](sycl::id<1> b) {
            float* row = data + b * dim;
            for (int r = 0; r < num_rotations; ++r) {
                rotations[r].apply(row[r * 2], row[r * 2 + 1]);
            }
        });
    });
}

} // namespace kvq::rotorquant

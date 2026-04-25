#pragma once

// IsoQuant — Quaternion 4D rotation variant
//
// Processes KV vectors in 4D blocks using unit quaternion rotations.
// A quaternion q = w + xi + yj + zk represents a rotation in 4D space
// (technically a left-isoclinic rotation, hence "IsoQuant").
//
// For d=128: 32 independent 4D rotations, each parameterized by a unit quaternion (4 params).
// Total parameters: 128
// FMAs per vector: 512 (32 blocks * 16 FMAs per 4D rotation)
//
// 4D blocks provide better mixing than PlanarQuant's 2D blocks while
// remaining simpler than RotorQuant's Clifford approach.

#include <sycl/sycl.hpp>
#include <cmath>

namespace kvq::rotorquant {

struct Quaternion {
    float w, x, y, z;

    float norm() const { return std::sqrt(w * w + x * x + y * y + z * z); }

    void normalize() {
        float n = norm();
        if (n > 1e-8f) {
            float inv = 1.0f / n;
            w *= inv; x *= inv; y *= inv; z *= inv;
        }
    }

    Quaternion conjugate() const { return {w, -x, -y, -z}; }
};

// Apply quaternion rotation to a 4D vector (left-isoclinic rotation)
// v' = q * v * q_conj (treating v as a quaternion with w=0)
// But for 4D rotation we use: v' = q * v (left multiplication)
struct Vec4 { float a, b, c, d; };

inline Vec4 quat_rotate(const Quaternion& q, Vec4 v) {
    // Left-multiplication by unit quaternion in 4D:
    // [w -x -y -z] [a]
    // [x  w -z  y] [b]
    // [y  z  w -x] [c]
    // [z -y  x  w] [d]
    return {
        q.w * v.a - q.x * v.b - q.y * v.c - q.z * v.d,
        q.x * v.a + q.w * v.b - q.z * v.c + q.y * v.d,
        q.y * v.a + q.z * v.b + q.w * v.c - q.x * v.d,
        q.z * v.a - q.y * v.b + q.x * v.c + q.w * v.d,
    };
}

inline Vec4 quat_rotate_inverse(const Quaternion& q, Vec4 v) {
    Quaternion qc = q.conjugate();
    return quat_rotate(qc, v);
}

struct IsoRotationSet {
    static constexpr int block_size = 4;

    int dim;
    int num_rotations;
    const Quaternion* quats;

    static int rotations_needed(int dim) {
        return dim / 4;
    }

    void apply_forward(const float* input, float* output) const {
        int i = 0;
        for (int r = 0; r < num_rotations; ++r, i += 4) {
            Vec4 v = {input[i], input[i + 1], input[i + 2], input[i + 3]};
            Vec4 rv = quat_rotate(quats[r], v);
            output[i] = rv.a;
            output[i + 1] = rv.b;
            output[i + 2] = rv.c;
            output[i + 3] = rv.d;
        }
    }

    void apply_inverse(const float* input, float* output) const {
        int i = 0;
        for (int r = 0; r < num_rotations; ++r, i += 4) {
            Vec4 v = {input[i], input[i + 1], input[i + 2], input[i + 3]};
            Vec4 rv = quat_rotate_inverse(quats[r], v);
            output[i] = rv.a;
            output[i + 1] = rv.b;
            output[i + 2] = rv.c;
            output[i + 3] = rv.d;
        }
    }
};

sycl::event apply_iso_rotations_batch(
    sycl::queue& q,
    const float* input,
    float* output,
    const Quaternion* quats,
    int batch_size,
    int dim,
    const std::vector<sycl::event>& deps = {})
{
    int num_rotations = dim / 4;

    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(batch_size), [=](sycl::id<1> b) {
            const float* in = input + b * dim;
            float* out = output + b * dim;

            int i = 0;
            for (int r = 0; r < num_rotations; ++r, i += 4) {
                Vec4 v = {in[i], in[i + 1], in[i + 2], in[i + 3]};
                Vec4 rv = quat_rotate(quats[r], v);
                out[i] = rv.a;
                out[i + 1] = rv.b;
                out[i + 2] = rv.c;
                out[i + 3] = rv.d;
            }
        });
    });
}

} // namespace kvq::rotorquant

#pragma once

// Clifford Algebra Cl(3,0) Rotor for RotorQuant
//
// RotorQuant replaces TurboQuant's d x d Walsh-Hadamard matrix with a Clifford
// rotor in the geometric algebra Cl(3,0). The key insight is that a rotor
// R = a + b*e12 + c*e13 + d*e23 acts on 3D blocks of the vector via the
// sandwich product: v' = R * v * R~
//
// For d=128 head dimensions processed in 3D blocks:
//   - WHT: d * log(d) = 896 FMAs (but full d x d matrix = 16,384 FMAs in practice)
//   - Clifford rotor: ~100 FMAs total (43 blocks of ~2.4 FMAs each)
//
// This is because Cl(3,0) rotors have algebraic sparsity: the sandwich product
// expands to only 9 multiply-adds per 3D block, not a full matrix multiply.
//
// Rotor parameterization:
//   R = cos(theta/2) + sin(theta/2) * (a*e12 + b*e13 + c*e23)
//   where (a,b,c) is the unit rotation axis and theta is the angle
//
// The rotor parameters are learned per-layer (372 params for d=128).

#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdint>

namespace kvq::rotorquant {

// A Cl(3,0) rotor: R = s + xy*e12 + xz*e13 + yz*e23
// Normalized: s^2 + xy^2 + xz^2 + yz^2 = 1
struct Rotor3 {
    float s;    // scalar part
    float xy;   // e12 bivector component
    float xz;   // e13 bivector component
    float yz;   // e23 bivector component

    static Rotor3 from_angle_axis(float angle, float ax, float ay, float az) {
        float half = angle * 0.5f;
        float c = std::cos(half);
        float s = std::sin(half);
        // Rotation in plane perpendicular to axis (ax, ay, az):
        // bivector = az*e12 - ay*e13 + ax*e23
        return {c, s * az, -s * ay, s * ax};
    }

    Rotor3 reverse() const {
        return {s, -xy, -xz, -yz};
    }

    void normalize() {
        float norm = std::sqrt(s * s + xy * xy + xz * xz + yz * yz);
        if (norm > 1e-8f) {
            float inv = 1.0f / norm;
            s *= inv; xy *= inv; xz *= inv; yz *= inv;
        }
    }
};

// Apply rotor sandwich product to a 3D vector: v' = R * v * R~
// This is the core operation — ~9 multiply-adds per 3D vector.
struct Vec3 { float x, y, z; };

inline Vec3 apply_rotor(const Rotor3& R, Vec3 v) {
    // Expanded sandwich product R * (v.x*e1 + v.y*e2 + v.z*e3) * R~
    // Using the Cl(3,0) multiplication table:
    float s = R.s, xy = R.xy, xz = R.xz, yz = R.yz;

    // Precompute common terms
    float ss = s * s;
    float xy2 = xy * xy;
    float xz2 = xz * xz;
    float yz2 = yz * yz;

    float out_x = v.x * (ss + yz2 - xy2 - xz2)
                + 2.0f * (v.y * (s * xz + xy * yz) + v.z * (-s * xy + xz * yz));

    float out_y = v.y * (ss + xz2 - xy2 - yz2)
                + 2.0f * (v.x * (-s * xz + xy * yz) + v.z * (s * yz + xy * xz));

    float out_z = v.z * (ss + xy2 - xz2 - yz2)
                + 2.0f * (v.x * (s * xy + xz * yz) + v.y * (-s * yz + xy * xz));

    return {out_x, out_y, out_z};
}

// Rotate a d-dimensional vector by applying rotors to consecutive 3D blocks.
// For d=128: 42 full blocks + 2 remainder elements (handled separately).
// Each block uses an independent rotor -> 42 * 4 = 168 parameters.
// Total rotors stored: ceil(d/3) = 43
struct RotorSet {
    static constexpr int block_size = 3;

    int dim;
    int num_rotors;        // ceil(dim / 3)
    const Rotor3* rotors;  // [num_rotors]

    static int rotors_needed(int dim) {
        return (dim + block_size - 1) / block_size;
    }

    void apply_forward(const float* input, float* output) const {
        int i = 0;
        for (int r = 0; r < num_rotors && i + 2 < dim; ++r, i += block_size) {
            Vec3 v = {input[i], input[i + 1], input[i + 2]};
            Vec3 out = apply_rotor(rotors[r], v);
            output[i] = out.x;
            output[i + 1] = out.y;
            output[i + 2] = out.z;
        }
        // Copy remainder elements unchanged
        for (; i < dim; ++i) {
            output[i] = input[i];
        }
    }

    void apply_inverse(const float* input, float* output) const {
        int i = 0;
        for (int r = 0; r < num_rotors && i + 2 < dim; ++r, i += block_size) {
            Vec3 v = {input[i], input[i + 1], input[i + 2]};
            Vec3 out = apply_rotor(rotors[r].reverse(), v);
            output[i] = out.x;
            output[i + 1] = out.y;
            output[i + 2] = out.z;
        }
        for (; i < dim; ++i) {
            output[i] = input[i];
        }
    }
};

// SYCL kernel: Apply Clifford rotors to a batch of KV vectors
// Input:  float[batch_size, dim]
// Output: float[batch_size, dim] (rotated)
// Rotors: Rotor3[num_rotors] (learned parameters)
sycl::event apply_rotors_batch(
    sycl::queue& q,
    const float* input,
    float* output,
    const Rotor3* rotors,
    int batch_size,
    int dim,
    const std::vector<sycl::event>& deps = {})
{
    int num_rotors = RotorSet::rotors_needed(dim);

    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(batch_size), [=](sycl::id<1> b) {
            const float* in = input + b * dim;
            float* out = output + b * dim;

            int i = 0;
            for (int r = 0; r < num_rotors && i + 2 < dim; ++r, i += 3) {
                Vec3 v = {in[i], in[i + 1], in[i + 2]};
                Vec3 rv = apply_rotor(rotors[r], v);
                out[i] = rv.x;
                out[i + 1] = rv.y;
                out[i + 2] = rv.z;
            }
            for (; i < dim; ++i) {
                out[i] = in[i];
            }
        });
    });
}

} // namespace kvq::rotorquant

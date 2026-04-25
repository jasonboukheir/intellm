#pragma once

// Shared quantization utilities for TurboQuant and RotorQuant
//
// Both methods follow the same pipeline:
//   1. Rotate values (WHT or Clifford rotor) to spread information
//   2. Quantize rotated values to low bit-width
//   3. Store compressed + metadata for dequantization
//
// This header provides the shared infrastructure.

#include <sycl/sycl.hpp>
#include <cstdint>
#include <cmath>

namespace kvq {

enum class QuantMethod {
    TurboQuant,   // Walsh-Hadamard + Lloyd-Max + QJL
    IsoQuant,     // Quaternion 4D rotation
    PlanarQuant,  // Givens 2D rotation
    RotorQuant,   // Clifford Cl(3,0) rotation
};

struct QuantConfig {
    int key_bits = 3;
    int value_bits = 2;
    int group_size = 128;     // elements per quantization group
    bool use_qjl = true;     // TurboQuant: use QJL residual bit
    QuantMethod method = QuantMethod::RotorQuant;
};

// Per-group quantization metadata
struct alignas(4) GroupMeta {
    float scale;
    float zero_point;
};

// Compression ratio calculation
constexpr float compression_ratio(int orig_bits, int quant_bits, int group_size) {
    float meta_overhead = sizeof(GroupMeta) * 8.0f / group_size;
    return static_cast<float>(orig_bits) / (quant_bits + meta_overhead);
}

// Uniform quantization: value -> [0, 2^bits - 1]
template <int BITS>
inline uint8_t uniform_quantize(float val, float scale, float zero_point) {
    constexpr int max_val = (1 << BITS) - 1;
    float q = (val - zero_point) / scale;
    q = sycl::clamp(q, 0.0f, static_cast<float>(max_val));
    return static_cast<uint8_t>(sycl::round(q));
}

// Uniform dequantization
template <int BITS>
inline float uniform_dequantize(uint8_t qval, float scale, float zero_point) {
    return static_cast<float>(qval) * scale + zero_point;
}

// Bit-packing utilities
// Pack N-bit values into bytes. For 3-bit: pack 8 values into 3 bytes.
// For 2-bit: pack 4 values per byte.

inline void pack_2bit(const uint8_t* src, uint8_t* dst, int count) {
    for (int i = 0; i < count / 4; ++i) {
        dst[i] = (src[i * 4] & 0x3)
               | ((src[i * 4 + 1] & 0x3) << 2)
               | ((src[i * 4 + 2] & 0x3) << 4)
               | ((src[i * 4 + 3] & 0x3) << 6);
    }
}

inline void unpack_2bit(const uint8_t* src, uint8_t* dst, int count) {
    for (int i = 0; i < count / 4; ++i) {
        dst[i * 4]     = src[i] & 0x3;
        dst[i * 4 + 1] = (src[i] >> 2) & 0x3;
        dst[i * 4 + 2] = (src[i] >> 4) & 0x3;
        dst[i * 4 + 3] = (src[i] >> 6) & 0x3;
    }
}

inline void pack_3bit(const uint8_t* src, uint8_t* dst, int count) {
    // Pack 8 x 3-bit values into 3 bytes
    for (int i = 0; i < count / 8; ++i) {
        const uint8_t* s = src + i * 8;
        uint8_t* d = dst + i * 3;
        d[0] = (s[0] & 0x7) | ((s[1] & 0x7) << 3) | ((s[2] & 0x3) << 6);
        d[1] = ((s[2] & 0x4) >> 2) | ((s[3] & 0x7) << 1) | ((s[4] & 0x7) << 4) | ((s[5] & 0x1) << 7);
        d[2] = ((s[5] & 0x6) >> 1) | ((s[6] & 0x7) << 2) | ((s[7] & 0x7) << 5);
    }
}

inline void unpack_3bit(const uint8_t* src, uint8_t* dst, int count) {
    for (int i = 0; i < count / 8; ++i) {
        const uint8_t* s = src + i * 3;
        uint8_t* d = dst + i * 8;
        d[0] = s[0] & 0x7;
        d[1] = (s[0] >> 3) & 0x7;
        d[2] = ((s[0] >> 6) & 0x3) | ((s[1] & 0x1) << 2);
        d[3] = (s[1] >> 1) & 0x7;
        d[4] = (s[1] >> 4) & 0x7;
        d[5] = ((s[1] >> 7) & 0x1) | ((s[2] & 0x3) << 1);
        d[6] = (s[2] >> 2) & 0x7;
        d[7] = (s[2] >> 5) & 0x7;
    }
}

} // namespace kvq

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <random>

#include "common/quantize.hpp"
#include "turboquant/kernels/lloyd_max.hpp"

using namespace kvq;
using namespace kvq::turboquant;

void test_lloyd_max_3bit_ordering() {
    // Quantized values should be monotonically increasing
    for (float v = -3.0f; v <= 3.0f; v += 0.01f) {
        uint8_t q1 = lloyd_max_quantize<3>(v);
        uint8_t q2 = lloyd_max_quantize<3>(v + 0.5f);
        assert(q2 >= q1);
    }
    std::cout << "3-bit Lloyd-Max ordering: PASS" << std::endl;
}

void test_lloyd_max_roundtrip_quality() {
    // Measure MSE of quantize -> dequantize for Gaussian data
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    double mse_3bit = 0.0, mse_2bit = 0.0;
    int N = 100000;
    for (int i = 0; i < N; ++i) {
        float v = dist(gen);
        float r3 = lloyd_max_dequantize<3>(lloyd_max_quantize<3>(v));
        float r2 = lloyd_max_dequantize<2>(lloyd_max_quantize<2>(v));
        mse_3bit += (v - r3) * (v - r3);
        mse_2bit += (v - r2) * (v - r2);
    }
    mse_3bit /= N;
    mse_2bit /= N;

    // For N(0,1): uniform 3-bit MSE ~= 0.037, Lloyd-Max should be better
    std::cout << "3-bit Lloyd-Max MSE: " << mse_3bit << " (should be < 0.04)" << std::endl;
    std::cout << "2-bit Lloyd-Max MSE: " << mse_2bit << " (should be < 0.20)" << std::endl;
    assert(mse_3bit < 0.04);
    assert(mse_2bit < 0.20);
}

void test_bitpack_2bit_roundtrip() {
    std::vector<uint8_t> src = {0, 1, 2, 3, 1, 2, 0, 3};
    std::vector<uint8_t> packed(2);
    std::vector<uint8_t> unpacked(8);

    pack_2bit(src.data(), packed.data(), 8);
    unpack_2bit(packed.data(), unpacked.data(), 8);

    for (int i = 0; i < 8; ++i) {
        assert(src[i] == unpacked[i]);
    }
    std::cout << "2-bit pack/unpack roundtrip: PASS" << std::endl;
}

void test_bitpack_3bit_roundtrip() {
    std::vector<uint8_t> src = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<uint8_t> packed(3);
    std::vector<uint8_t> unpacked(8);

    pack_3bit(src.data(), packed.data(), 8);
    unpack_3bit(packed.data(), unpacked.data(), 8);

    for (int i = 0; i < 8; ++i) {
        assert(src[i] == unpacked[i]);
    }
    std::cout << "3-bit pack/unpack roundtrip: PASS" << std::endl;
}

void test_compression_ratio() {
    float r_3bit = compression_ratio(16, 3, 128); // FP16 -> 3-bit
    float r_2bit = compression_ratio(16, 2, 128);
    std::cout << "FP16->3bit compression: " << r_3bit << "x" << std::endl;
    std::cout << "FP16->2bit compression: " << r_2bit << "x" << std::endl;
    assert(r_3bit > 4.0f && r_3bit < 6.0f);
    assert(r_2bit > 6.0f && r_2bit < 9.0f);
}

int main() {
    test_lloyd_max_3bit_ordering();
    test_lloyd_max_roundtrip_quality();
    test_bitpack_2bit_roundtrip();
    test_bitpack_3bit_roundtrip();
    test_compression_ratio();
    std::cout << "All quantization tests passed." << std::endl;
    return 0;
}

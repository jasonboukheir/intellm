#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <random>
#include "turboquant/kernels/walsh_hadamard.hpp"

using WHT128 = kvq::turboquant::WalshHadamardTransform<128>;

void test_orthogonality() {
    // WHT is an involution (self-inverse) up to scaling.
    // Applying it twice should return the original vector.
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> orig(128), data(128);
    for (int i = 0; i < 128; ++i) {
        orig[i] = dist(gen);
        data[i] = orig[i];
    }

    WHT128::apply_cpu(data.data());
    WHT128::apply_cpu(data.data());

    float max_err = 0.0f;
    for (int i = 0; i < 128; ++i) {
        max_err = std::max(max_err, std::abs(data[i] - orig[i]));
    }
    std::cout << "WHT orthogonality max error: " << max_err << std::endl;
    assert(max_err < 1e-4f);
}

void test_energy_preservation() {
    // WHT is orthogonal -> preserves L2 norm
    std::mt19937 gen(123);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data(128);
    for (auto& x : data) x = dist(gen);

    float norm_before = 0.0f;
    for (auto x : data) norm_before += x * x;

    WHT128::apply_cpu(data.data());

    float norm_after = 0.0f;
    for (auto x : data) norm_after += x * x;

    float rel_err = std::abs(norm_before - norm_after) / norm_before;
    std::cout << "WHT energy preservation relative error: " << rel_err << std::endl;
    assert(rel_err < 1e-5f);
}

void test_distribution_spreading() {
    // After WHT, a sparse input should be spread across all dimensions
    std::vector<float> data(128, 0.0f);
    data[0] = 1.0f; // impulse

    WHT128::apply_cpu(data.data());

    // All elements should have the same magnitude (1/sqrt(128))
    float expected = 1.0f / std::sqrt(128.0f);
    float max_err = 0.0f;
    for (int i = 0; i < 128; ++i) {
        max_err = std::max(max_err, std::abs(std::abs(data[i]) - expected));
    }
    std::cout << "WHT impulse response max error: " << max_err << std::endl;
    assert(max_err < 1e-6f);
}

int main() {
    test_orthogonality();
    test_energy_preservation();
    test_distribution_spreading();
    std::cout << "All WHT tests passed." << std::endl;
    return 0;
}

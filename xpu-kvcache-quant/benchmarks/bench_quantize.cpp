#include <sycl/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>
#include <random>

#include "common/quantize.hpp"
#include "turboquant/kernels/lloyd_max.hpp"

using Clock = std::chrono::high_resolution_clock;

int main() {
    constexpr int DIM = 128;
    constexpr int BATCH = 8192;
    constexpr int WARMUP = 10;
    constexpr int ITERS = 100;

    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "Batch: " << BATCH << ", Dim: " << DIM << "\n\n";

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> h_data(BATCH * DIM);
    for (auto& x : h_data) x = dist(gen);

    auto* d_data = sycl::malloc_device<float>(BATCH * DIM, q);
    auto* d_qout = sycl::malloc_device<uint8_t>(BATCH * DIM, q);
    int num_groups = DIM / 128;
    auto* d_scales = sycl::malloc_device<float>(BATCH * std::max(num_groups, 1), q);

    q.memcpy(d_data, h_data.data(), BATCH * DIM * sizeof(float)).wait();

    // Benchmark 3-bit Lloyd-Max quantization
    {
        for (int i = 0; i < WARMUP; ++i) {
            kvq::turboquant::lloyd_max_quantize_batch<3, 128>(
                q, d_data, d_qout, d_scales, BATCH, DIM).wait();
        }
        auto start = Clock::now();
        for (int i = 0; i < ITERS; ++i) {
            kvq::turboquant::lloyd_max_quantize_batch<3, 128>(
                q, d_data, d_qout, d_scales, BATCH, DIM).wait();
        }
        double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count() / ITERS;

        size_t input_bytes = BATCH * DIM * sizeof(float);
        size_t output_bytes = BATCH * DIM; // 1 byte per element (unpacked)
        double gbs = (input_bytes + output_bytes) / (ms * 1e6);
        double ratio = kvq::compression_ratio(32, 3, 128);
        std::cout << "3-bit Lloyd-Max: " << ms << " ms, " << gbs << " GB/s, "
                  << ratio << "x compression\n";
    }

    // Benchmark 2-bit Lloyd-Max quantization
    {
        for (int i = 0; i < WARMUP; ++i) {
            kvq::turboquant::lloyd_max_quantize_batch<2, 128>(
                q, d_data, d_qout, d_scales, BATCH, DIM).wait();
        }
        auto start = Clock::now();
        for (int i = 0; i < ITERS; ++i) {
            kvq::turboquant::lloyd_max_quantize_batch<2, 128>(
                q, d_data, d_qout, d_scales, BATCH, DIM).wait();
        }
        double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count() / ITERS;

        double ratio = kvq::compression_ratio(32, 2, 128);
        std::cout << "2-bit Lloyd-Max: " << ms << " ms, "
                  << ratio << "x compression\n";
    }

    sycl::free(d_data, q);
    sycl::free(d_qout, q);
    sycl::free(d_scales, q);

    return 0;
}

#include <sycl/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>
#include <random>
#include "kernels/flash_attention_xe2.hpp"

using namespace xfa;
using Clock = std::chrono::high_resolution_clock;

template <typename T>
void fill_random(std::vector<T>& v) {
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.02f);
    for (auto& x : v) {
        x = static_cast<T>(dist(gen));
    }
}

struct BenchConfig {
    int batch_size;
    int num_heads;
    int seq_len;
    int head_dim;
};

enum class Variant { V1, V1_SLMQO };
static const char* variant_label(Variant v) {
    return v == Variant::V1 ? "v1" : "v1.5";
}

void run_benchmark(sycl::queue& q, const BenchConfig& cfg, Variant variant, int warmup, int iters) {
    size_t total = cfg.batch_size * cfg.num_heads * cfg.seq_len * cfg.head_dim;

    std::vector<sycl::half> h_Q(total), h_K(total), h_V(total), h_O(total);
    std::vector<float> h_L(cfg.batch_size * cfg.num_heads * cfg.seq_len);

    fill_random(h_Q);
    fill_random(h_K);
    fill_random(h_V);

    auto* d_Q = sycl::malloc_device<sycl::half>(total, q);
    auto* d_K = sycl::malloc_device<sycl::half>(total, q);
    auto* d_V = sycl::malloc_device<sycl::half>(total, q);
    auto* d_O = sycl::malloc_device<sycl::half>(total, q);
    auto* d_L = sycl::malloc_device<float>(h_L.size(), q);

    q.memcpy(d_Q, h_Q.data(), total * sizeof(sycl::half)).wait();
    q.memcpy(d_K, h_K.data(), total * sizeof(sycl::half)).wait();
    q.memcpy(d_V, h_V.data(), total * sizeof(sycl::half)).wait();

    auto launch = [&]() {
        if (variant == Variant::V1) {
            return flash_attention_forward(q, d_Q, d_K, d_V, d_O, d_L,
                cfg.batch_size, cfg.num_heads, cfg.seq_len, cfg.head_dim);
        }
        return flash_attention_forward_slmqo(q, d_Q, d_K, d_V, d_O, d_L,
            cfg.batch_size, cfg.num_heads, cfg.seq_len, cfg.head_dim);
    };
    for (int i = 0; i < warmup; ++i) launch().wait();
    auto start = Clock::now();
    for (int i = 0; i < iters; ++i) launch().wait();
    auto end = Clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;

    // Compute FLOPs: 2 * batch * heads * seq^2 * head_dim (for QK^T)
    //              + 2 * batch * heads * seq^2 * head_dim (for softmax(QK^T) @ V)
    double flops = 4.0 * cfg.batch_size * cfg.num_heads
                   * static_cast<double>(cfg.seq_len) * cfg.seq_len * cfg.head_dim;
    double tflops = flops / (ms * 1e9);

    // Memory bandwidth: read Q + K + V, write O (each seq_len * head_dim * sizeof(half))
    double bytes = 4.0 * total * sizeof(sycl::half);
    double bw_gbs = bytes / (ms * 1e6);

    std::cout << "  [" << variant_label(variant) << "] "
              << "B=" << cfg.batch_size << " H=" << cfg.num_heads
              << " S=" << cfg.seq_len << " D=" << cfg.head_dim
              << " | " << ms << " ms"
              << " | " << tflops << " TFLOPS"
              << " | " << bw_gbs << " GB/s"
              << std::endl;

    sycl::free(d_Q, q);
    sycl::free(d_K, q);
    sycl::free(d_V, q);
    sycl::free(d_O, q);
    sycl::free(d_L, q);
}

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};

    auto dev = q.get_device();
    std::cout << "Device: " << dev.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "Max compute units: " << dev.get_info<sycl::info::device::max_compute_units>() << std::endl;
    std::cout << "Max work group: " << dev.get_info<sycl::info::device::max_work_group_size>() << std::endl;
    std::cout << "Local mem: " << dev.get_info<sycl::info::device::local_mem_size>() << " bytes" << std::endl;
    std::cout << std::endl;

    std::vector<BenchConfig> configs = {
        // Typical LLM attention shapes
        {1, 32, 512, 128},    // single batch, 32 heads (Llama-style)
        {1, 32, 2048, 128},
        {1, 32, 4096, 128},
        {1, 32, 8192, 128},
        {4, 32, 2048, 128},   // batched
        {1, 8, 2048, 128},    // GQA (8 KV heads)
        {1, 8, 8192, 128},
    };

    int warmup = 5;
    int iters = 20;

    std::cout << "Flash Attention v2 (Xe2) Benchmark" << std::endl;
    std::cout << "Warmup: " << warmup << " | Iterations: " << iters << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    for (auto v : {Variant::V1, Variant::V1_SLMQO}) {
        for (const auto& cfg : configs) {
            run_benchmark(q, cfg, v, warmup, iters);
        }
        std::cout << std::endl;
    }

    return 0;
}

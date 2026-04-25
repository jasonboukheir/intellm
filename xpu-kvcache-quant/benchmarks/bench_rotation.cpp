#include <sycl/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>
#include <random>

#include "turboquant/kernels/walsh_hadamard.hpp"
#include "rotorquant/kernels/clifford_rotor.hpp"
#include "rotorquant/kernels/planar_quant.hpp"
#include "rotorquant/kernels/iso_quant.hpp"

using Clock = std::chrono::high_resolution_clock;

template <typename F>
double time_ms(sycl::queue& q, F fn, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) { fn(); q.wait(); }
    auto start = Clock::now();
    for (int i = 0; i < iters; ++i) { fn(); }
    q.wait();
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count() / iters;
}

int main() {
    constexpr int DIM = 128;
    constexpr int BATCH = 8192; // typical KV cache: 8K tokens
    constexpr int WARMUP = 10;
    constexpr int ITERS = 100;

    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    auto dev = q.get_device();
    std::cout << "Device: " << dev.get_info<sycl::info::device::name>() << "\n";
    std::cout << "Batch: " << BATCH << " vectors, Dim: " << DIM << "\n\n";

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> h_data(BATCH * DIM);
    for (auto& x : h_data) x = dist(gen);

    auto* d_data = sycl::malloc_device<float>(BATCH * DIM, q);
    auto* d_out = sycl::malloc_device<float>(BATCH * DIM, q);
    q.memcpy(d_data, h_data.data(), BATCH * DIM * sizeof(float)).wait();

    // Benchmark Walsh-Hadamard Transform (TurboQuant)
    {
        q.memcpy(d_out, d_data, BATCH * DIM * sizeof(float)).wait();
        double ms = time_ms(q, [&]() {
            kvq::turboquant::WalshHadamardTransform<DIM>::apply_batch(q, d_out, BATCH);
        }, WARMUP, ITERS);
        double gbs = BATCH * DIM * sizeof(float) * 2.0 / (ms * 1e6); // read + write
        std::cout << "Walsh-Hadamard (TurboQuant): " << ms << " ms, " << gbs << " GB/s\n";
    }

    // Benchmark Clifford Rotor (RotorQuant)
    {
        int num_rotors = kvq::rotorquant::RotorSet::rotors_needed(DIM);
        std::vector<kvq::rotorquant::Rotor3> h_rotors(num_rotors);
        for (auto& r : h_rotors) {
            r = kvq::rotorquant::Rotor3::from_angle_axis(dist(gen), dist(gen), dist(gen), dist(gen));
            r.normalize();
        }
        auto* d_rotors = sycl::malloc_device<kvq::rotorquant::Rotor3>(num_rotors, q);
        q.memcpy(d_rotors, h_rotors.data(), num_rotors * sizeof(kvq::rotorquant::Rotor3)).wait();

        double ms = time_ms(q, [&]() {
            kvq::rotorquant::apply_rotors_batch(q, d_data, d_out, d_rotors, BATCH, DIM);
        }, WARMUP, ITERS);
        double gbs = BATCH * DIM * sizeof(float) * 2.0 / (ms * 1e6);
        std::cout << "Clifford Rotor (RotorQuant): " << ms << " ms, " << gbs << " GB/s\n";

        sycl::free(d_rotors, q);
    }

    // Benchmark Planar Rotation (PlanarQuant)
    {
        int num_rot = kvq::rotorquant::PlanarRotationSet::rotations_needed(DIM);
        std::vector<kvq::rotorquant::GivensRotation> h_rots(num_rot);
        for (auto& r : h_rots) {
            r = kvq::rotorquant::GivensRotation::from_angle(dist(gen));
        }
        auto* d_rots = sycl::malloc_device<kvq::rotorquant::GivensRotation>(num_rot, q);
        q.memcpy(d_rots, h_rots.data(), num_rot * sizeof(kvq::rotorquant::GivensRotation)).wait();

        q.memcpy(d_out, d_data, BATCH * DIM * sizeof(float)).wait();
        double ms = time_ms(q, [&]() {
            kvq::rotorquant::apply_planar_rotations_batch(q, d_out, d_rots, BATCH, DIM);
        }, WARMUP, ITERS);
        double gbs = BATCH * DIM * sizeof(float) * 2.0 / (ms * 1e6);
        std::cout << "Planar Givens (PlanarQuant): " << ms << " ms, " << gbs << " GB/s\n";

        sycl::free(d_rots, q);
    }

    // Benchmark IsoQuant (quaternion)
    {
        int num_rot = kvq::rotorquant::IsoRotationSet::rotations_needed(DIM);
        std::vector<kvq::rotorquant::Quaternion> h_quats(num_rot);
        for (auto& q_ : h_quats) {
            q_ = {dist(gen), dist(gen), dist(gen), dist(gen)};
            q_.normalize();
        }
        auto* d_quats = sycl::malloc_device<kvq::rotorquant::Quaternion>(num_rot, q);
        q.memcpy(d_quats, h_quats.data(), num_rot * sizeof(kvq::rotorquant::Quaternion)).wait();

        double ms = time_ms(q, [&]() {
            kvq::rotorquant::apply_iso_rotations_batch(q, d_data, d_out, d_quats, BATCH, DIM);
        }, WARMUP, ITERS);
        double gbs = BATCH * DIM * sizeof(float) * 2.0 / (ms * 1e6);
        std::cout << "Quaternion 4D (IsoQuant):    " << ms << " ms, " << gbs << " GB/s\n";

        sycl::free(d_quats, q);
    }

    sycl::free(d_data, q);
    sycl::free(d_out, q);

    return 0;
}

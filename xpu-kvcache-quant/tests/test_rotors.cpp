#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <random>

#include "rotorquant/kernels/clifford_rotor.hpp"
#include "rotorquant/kernels/planar_quant.hpp"
#include "rotorquant/kernels/iso_quant.hpp"

using namespace kvq::rotorquant;

void test_clifford_roundtrip() {
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    constexpr int DIM = 128;
    int num_rotors = RotorSet::rotors_needed(DIM);
    std::vector<Rotor3> rotors(num_rotors);
    for (auto& r : rotors) {
        r = Rotor3::from_angle_axis(dist(gen), dist(gen), dist(gen), dist(gen));
        r.normalize();
    }

    RotorSet rset{DIM, num_rotors, rotors.data()};

    std::vector<float> orig(DIM), rotated(DIM), recovered(DIM);
    for (auto& x : orig) x = dist(gen);

    rset.apply_forward(orig.data(), rotated.data());
    rset.apply_inverse(rotated.data(), recovered.data());

    float max_err = 0.0f;
    for (int i = 0; i < DIM; ++i) {
        max_err = std::max(max_err, std::abs(orig[i] - recovered[i]));
    }
    std::cout << "Clifford rotor roundtrip max error: " << max_err << std::endl;
    assert(max_err < 1e-5f);
}

void test_clifford_norm_preservation() {
    std::mt19937 gen(99);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    constexpr int DIM = 128;
    int num_rotors = RotorSet::rotors_needed(DIM);
    std::vector<Rotor3> rotors(num_rotors);
    for (auto& r : rotors) {
        r = Rotor3::from_angle_axis(dist(gen), dist(gen), dist(gen), dist(gen));
        r.normalize();
    }

    RotorSet rset{DIM, num_rotors, rotors.data()};

    std::vector<float> data(DIM), rotated(DIM);
    for (auto& x : data) x = dist(gen);

    float norm_before = 0.0f;
    for (auto x : data) norm_before += x * x;

    rset.apply_forward(data.data(), rotated.data());

    float norm_after = 0.0f;
    for (auto x : rotated) norm_after += x * x;

    float rel_err = std::abs(norm_before - norm_after) / norm_before;
    std::cout << "Clifford rotor norm preservation error: " << rel_err << std::endl;
    assert(rel_err < 1e-5f);
}

void test_planar_roundtrip() {
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    constexpr int DIM = 128;
    int num_rot = PlanarRotationSet::rotations_needed(DIM);
    std::vector<GivensRotation> rots(num_rot);
    for (auto& r : rots) r = GivensRotation::from_angle(dist(gen));

    PlanarRotationSet pset{DIM, num_rot, rots.data()};

    std::vector<float> orig(DIM), data(DIM);
    for (int i = 0; i < DIM; ++i) { orig[i] = dist(gen); data[i] = orig[i]; }

    pset.apply_forward(data.data());
    pset.apply_inverse(data.data());

    float max_err = 0.0f;
    for (int i = 0; i < DIM; ++i) {
        max_err = std::max(max_err, std::abs(orig[i] - data[i]));
    }
    std::cout << "Planar roundtrip max error: " << max_err << std::endl;
    assert(max_err < 1e-5f);
}

void test_iso_roundtrip() {
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    constexpr int DIM = 128;
    int num_rot = IsoRotationSet::rotations_needed(DIM);
    std::vector<Quaternion> quats(num_rot);
    for (auto& q : quats) { q = {dist(gen), dist(gen), dist(gen), dist(gen)}; q.normalize(); }

    IsoRotationSet iset{DIM, num_rot, quats.data()};

    std::vector<float> orig(DIM), rotated(DIM), recovered(DIM);
    for (auto& x : orig) x = dist(gen);

    iset.apply_forward(orig.data(), rotated.data());
    iset.apply_inverse(rotated.data(), recovered.data());

    float max_err = 0.0f;
    for (int i = 0; i < DIM; ++i) {
        max_err = std::max(max_err, std::abs(orig[i] - recovered[i]));
    }
    std::cout << "IsoQuant roundtrip max error: " << max_err << std::endl;
    assert(max_err < 1e-5f);
}

int main() {
    test_clifford_roundtrip();
    test_clifford_norm_preservation();
    test_planar_roundtrip();
    test_iso_roundtrip();
    std::cout << "All rotation tests passed." << std::endl;
    return 0;
}

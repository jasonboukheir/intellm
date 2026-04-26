// Probe what joint_matrix combinations the device reports as supported.
#include <sycl/sycl.hpp>
#include <iostream>

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    auto dev = q.get_device();
    std::cout << "Device: " << dev.get_info<sycl::info::device::name>() << "\n";
    auto arch = dev.get_info<sycl::ext::oneapi::experimental::info::device::architecture>();
    std::cout << "Arch enum: " << static_cast<int>(arch) << "\n";

    using comb_info = sycl::ext::oneapi::experimental::info::device::matrix_combinations;
    try {
        auto combos = dev.get_info<comb_info>();
        std::cout << "Matrix combinations reported: " << combos.size() << "\n";
        for (size_t i = 0; i < combos.size(); ++i) {
            const auto& c = combos[i];
            std::cout << "  [" << i << "] M=" << c.msize << " N=" << c.nsize
                      << " K=" << c.ksize
                      << " mmax=" << c.max_msize << " nmax=" << c.max_nsize
                      << " kmax=" << c.max_ksize
                      << " atype=" << static_cast<int>(c.atype)
                      << " btype=" << static_cast<int>(c.btype)
                      << " ctype=" << static_cast<int>(c.ctype)
                      << " dtype=" << static_cast<int>(c.dtype)
                      << "\n";
        }
    } catch (sycl::exception& e) {
        std::cerr << "matrix_combinations query failed: " << e.what() << "\n";
    }
    return 0;
}

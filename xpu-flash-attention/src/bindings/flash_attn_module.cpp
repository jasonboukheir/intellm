#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// Python bindings for xpu-flash-attention
//
// Provides a Python interface for benchmarking and testing the SYCL kernels
// without requiring PyTorch XPU integration. Uses numpy arrays as the
// interchange format.
//
// TODO: Add torch XPU tensor support once torch+xpu is available in the env

namespace py = pybind11;

PYBIND11_MODULE(xpu_flash_attn, m) {
    m.doc() = "Flash attention for Intel Xe2 GPUs (SYCL)";

    // TODO: Bind flash_attention_forward with numpy array wrappers
    // The binding needs to:
    //   1. Accept numpy arrays for Q, K, V
    //   2. Create SYCL buffers from the array data
    //   3. Submit the kernel
    //   4. Return output as numpy array
    //
    // For torch integration, also provide:
    //   flash_attn_forward_torch(Q, K, V) accepting torch.Tensor on XPU device

    m.def("available", []() { return false; },
          "Check if XPU flash attention is compiled and available");
}

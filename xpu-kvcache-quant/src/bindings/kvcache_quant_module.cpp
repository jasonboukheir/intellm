#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(xpu_kvcache_quant, m) {
    m.doc() = "KV cache quantization for Intel XPU (TurboQuant + RotorQuant)";

    py::enum_<int>(m, "QuantMethod")
        .value("TurboQuant", 0)
        .value("IsoQuant", 1)
        .value("PlanarQuant", 2)
        .value("RotorQuant", 3);

    // TODO: Bind SYCL kernel wrappers
    // - quantize_kv_cache(keys, values, method, key_bits, value_bits) -> compressed
    // - dequantize_kv_cache(compressed, method) -> keys, values
    // - benchmark_rotation(data, method, iters) -> timing results

    m.def("available", []() { return false; },
          "Check if XPU KV cache quantization is compiled and available");
}

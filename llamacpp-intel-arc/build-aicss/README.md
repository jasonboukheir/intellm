# build-aicss/

Local checkout of `aicss-genai/llama.cpp@aicss-genai-fork` (the umbrella PR
ggml-org/llama.cpp#22066 = the seven sub-PRs #22147–#22156). Built here
to test the Battlemage SYCL optimizations against our Qwen3.6-35B-A3B
workload before they land upstream.

## Layout

- `llama.cpp/` — git clone of the fork, branch `aicss-genai-fork`
- `aicss-fork-merge-fixes.patch` — local fixes for merge artifacts in
  the fork (see "Fork bugs we fixed" below). Apply with
  `cd llama.cpp && git apply ../aicss-fork-merge-fixes.patch`.

## Build

Inside the project (where this README lives):

```sh
podman run --rm \
    --device /dev/dri --group-add keep-groups \
    -v "$(pwd)/llama.cpp:/work:z" -w /work \
    -e SETVARS_COMPLETED=0 \
    docker.io/intel/vllm:0.17.0-xpu \
    bash -lc '. /opt/intel/oneapi/setvars.sh --force >/dev/null && \
      cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
        -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL \
        -DGGML_SYCL_DEVICE_ARCH=bmg-g31 -DGGML_SYCL_F16=ON \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF && \
      cmake --build build --target llama-server -j$(nproc)'
```

Resulting binary: `build-aicss/llama.cpp/build/bin/llama-server`.

To run with this binary, use `scripts/run-server-aicss.sh` from the
project root.

## Fork bugs we fixed

The fork contains a leftover merge artifact: a "fix duplicates" commit
landed but didn't fully de-duplicate. We needed three local patches to
build:

1. `ggml/src/ggml-sycl/dequantize.hpp` — duplicate definitions of
   `dequantize_q8_0_reorder` (lines 147 & 182) and
   `dequantize_block_q8_0_reorder` (lines 201 & 290). Deleted the second
   pair.
2. `ggml/src/ggml-sycl/ggml-sycl.cpp` — duplicate `reorder_qw_q5_k`,
   inconsistent `static bool` vs `static void` signatures across the
   `reorder_qw_q*` family, and a leftover bool-return check in
   `opt_for_reorder`. Made all five `reorder_qw_q*` return void; changed
   `reorder_qw`'s switch to use plain calls + break; dropped
   `if (reorder_qw(...))` to an unconditional call.
3. `ggml/src/ggml-sycl/mmvq.cpp` — duplicate
   `reorder_mul_mat_vec_q8_0_q8_1_sycl` (lines 682 & 725, identical
   bodies). Deleted the second.

These all cluster around split-PR #5 (Q5_K + Q8_0 reorder MMVQ)
re-introducing definitions that were already in upstream from earlier
commits. Worth filing an issue against the fork — but we don't need it
to validate the optimizations on our card.

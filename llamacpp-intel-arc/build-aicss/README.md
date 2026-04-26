# build-aicss/

Local llama.cpp build with the open Intel SYCL PRs cherry-picked. Built
here to test their effect against our Qwen3.6-35B-A3B workload before
they land upstream.

## Status: 6 of 7 PRs in use

| PR | Title | Status |
|----|-------|--------|
| [#22147](https://github.com/ggml-org/llama.cpp/pull/22147) | Battlemage AOT + MMQ subgroup | applied |
| [#22148](https://github.com/ggml-org/llama.cpp/pull/22148) | PAD non-contiguous stride | applied |
| [#22149](https://github.com/ggml-org/llama.cpp/pull/22149) | FILL/CUMSUM/DIAG/SOLVE_TRI/SSM_SCAN/GATED_DELTA_NET | applied |
| [#22150](https://github.com/ggml-org/llama.cpp/pull/22150) | small f32 matmul → oneMKL | applied |
| [#22152](https://github.com/ggml-org/llama.cpp/pull/22152) | Q5_K + Q8_0 reorder MMVQ | **skipped** — stale; upstream master independently added the same `dequantize_q8_0_reorder` / `dequantize_block_q8_0_reorder` / `reorder_mul_mat_vec_q8_0_q8_1_sycl` symbols, so cherry-picking creates duplicate definitions |
| [#22153](https://github.com/ggml-org/llama.cpp/pull/22153) | GGML_SYCL_USE_ASYNC_MEM_OP env | applied |
| [#22156](https://github.com/ggml-org/llama.cpp/pull/22156) | Q6_K SWAR byte-subtract | applied |

## How the checkout was built

```sh
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git llama-pr-only
cd llama-pr-only
git remote add aicss https://github.com/aicss-genai/llama.cpp.git
git fetch aicss aicss-genai/sycl-bmg-upstream-pr-{1,2,3,4,5,6,7}

# cherry-pick each PR's change commit onto current master,
# skipping pr-5 (already merged in spirit)
git cherry-pick \
    4065b65  # PR #22147
    5496728  # PR #22148
    ad7cabe  # PR #22149
    1dd517d  # PR #22150
    210de27  # PR #22153
    4f611ae  # PR #22156

# build (JIT — AOT works too with -DGGML_SYCL_DEVICE_ARCH=bmg-g31)
podman run --rm --device /dev/dri --group-add keep-groups \
  -v "$(pwd):/work:z" -w /work \
  -e SETVARS_COMPLETED=0 \
  docker.io/intel/vllm:0.17.0-xpu bash -lc '
    . /opt/intel/oneapi/setvars.sh --force >/dev/null
    cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
      -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF
    cmake --build build --target llama-server -j$(nproc)'
```

Resulting binary: `build-aicss/llama-pr-only/build/bin/llama-server`.

## Why not the full `aicss-genai-fork` branch?

The fork at `aicss-genai/llama.cpp@aicss-genai-fork` carries 12 patches —
the 7 in their open PRs *plus* 5 more (RMS_NORM+MUL fusion, native
shuffles, scratchpad pool, UNARY+MUL fusion, 3-op fusion, etc) that
the team is testing privately. We built the full fork (`llama.cpp/`
sibling dir, also gitignored) and it produced gibberish output on
Qwen3.6 — so one of those private patches is broken on this card. The
narrow cherry-pick reproduces only what's been publicly proposed.

## Run

```sh
nix run .#server -- configs/models/qwen3.6-35b-a3b-q4km.yaml   # vanilla container
scripts/run-server-aicss.sh configs/models/qwen3.6-35b-a3b-q4km.yaml   # this build
```

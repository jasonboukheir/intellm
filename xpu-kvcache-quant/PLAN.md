# Plan: serve TurboQuant on Intel XPU via vLLM

Status as of 2026-04-29.

## Strategy

vLLM PR #38479 merged TurboQuant as a first-class attention backend on 2026-04-15 (commit `f4b42df0`), and `vllm/platforms/xpu.py` already dispatches `--kv-cache-dtype turboquant_*` to it. The merged kernels are Triton; the merged shim (`vllm/v1/attention/backends/turboquant_attn.py`) has no CUDA-only primitives outside the kernel call sites, and QJL is intentionally omitted upstream.

Therefore: do not write a custom monkey-patch and do not port a third-party fork. Reuse the upstream shim and replace its Triton kernel calls with SYCL ops registered through a `vllm-xpu-kernels`-style plugin.

Tradeoff: we follow upstream's preset list (`k8v4`, `k3v4nc`, `t3nc`, `t4nc`) and inherit upstream's no-QJL decision. Locally-developed RotorQuant variants (Clifford / Iso / Planar rotations) and the QJL stage are deferred to phase E or later, after the WHT + Lloyd-Max baseline ships.

## Current local state (xpu-kvcache-quant/)

Mature: SYCL rotation kernels — WHT (`src/turboquant/kernels/walsh_hadamard.hpp`), Clifford rotor, quaternion iso, Givens planar. Tests + benchmarks pass.

Partial: Lloyd-Max codebooks done (`src/turboquant/kernels/lloyd_max.hpp`) but writes one byte per element — bit-packing is TODO.

Stubs only — DO NOT assume working:
- QJL (`src/turboquant/kernels/qjl.hpp`) — header-only, no body
- pybind11 bindings — `available()` returns false
- `vllm_adapter/monkey_patch.py` — logs a warning and no-ops

## Upstream files we will lift / reference

From `vllm-project/vllm@f4b42df0`:
- `vllm/v1/attention/backends/turboquant_attn.py` (812 LOC) — shim to keep
- `vllm/v1/attention/ops/triton_turboquant_store.py` (441 LOC) — kernel to replace
- `vllm/v1/attention/ops/triton_turboquant_decode.py` (617 LOC) — kernel to replace
- `vllm/model_executor/layers/quantization/turboquant/{config,centroids,quantizer}.py`
- `vllm/platforms/xpu.py` (6-line dispatch — already in place)
- `tests/quantization/test_turboquant.py` (570 LOC) — reuse as correctness fixture
- `tests/evals/gsm8k/configs/Qwen3-4B-TQ-{k8v4,k3v4nc,t3nc,t4nc}.yaml` — reuse for end-to-end eval

## Phases

### Phase A — verify upstream works on XPU (Triton-on-XPU baseline)

Goal: determine whether the merged Triton kernels JIT on Intel XPU as-is.

- Pin the intel/vllm container or build to a commit that includes PR #38479.
- Run `vllm serve Qwen/Qwen3-4B --kv-cache-dtype turboquant_k8v4` on Arc / BMG.
- Outcome A1: it works — we have a baseline; phase B becomes optional perf work.
- Outcome A2: Triton-XPU fails to JIT one or more kernels — phase B is mandatory; capture the failure for kernel-by-kernel substitution priority.

Acceptance: a single GSM8K eval run completes (any score) with `turboquant_k8v4` on XPU, OR a documented JIT failure trace.

### Phase B — SYCL ops behind the upstream shim

Goal: replace `triton_turboquant_store.py` and `triton_turboquant_decode.py` calls with SYCL ops without modifying `turboquant_attn.py`.

Subtasks:
1. Finish Lloyd-Max bit-packing in `src/turboquant/kernels/lloyd_max.hpp` (currently writes 1 byte/elem). Output layout must match upstream's expectation in `triton_turboquant_store.py` — read that file to lock the contract.
2. Implement real pybind11 bindings (`src/bindings/`) — drop the `available() == false` stub.
3. Register SYCL ops via `torch.library.define` so they appear under `torch.ops.vllm_xpu_kernels.*`. Use upstream `vllm-project/vllm-xpu-kernels` FP8/MxFP4 op registration as the template.
4. Provide `xpu_turboquant_store` and `xpu_turboquant_decode` Python modules that mirror the Triton ops' signatures and dispatch to the SYCL ops.
5. Patch `turboquant_attn.py` import block to use the SYCL ops on XPU. Smallest possible diff against upstream — track this as a maintained patch, not a fork.

Acceptance: same GSM8K eval as phase A, but kernel calls hit SYCL ops (verify via op profiler / logging). Score within ±2 points of phase A baseline (or phase A's expected score from upstream's own evals if A2 occurred).

### Phase C — validate with KL harness

Goal: catch quality regressions that GSM8K masks.

- Reuse `../llamacpp-intel-arc/harness/capture_logprobs.py` against the vLLM-XPU server.
- Reuse `../llamacpp-intel-arc/harness/compute_kl.py` for FP16 baseline vs `turboquant_k8v4` and `turboquant_k3v4_nc`.
- Long-context check via `build_long_prompts.py` — TurboQuant is sold on long-context, so this is the headline metric.

Acceptance: KL divergence vs FP16 baseline within paper-reported bounds for each preset.

### Phase D — autoresearch loop (only after A–C)

Premature today. Becomes viable once a working baseline exists, because each variant is a config flag rather than a kernel build:

- Search axis: bit allocation across layers / heads, rotation choice (WHT vs random Hadamard vs orthogonal), group sizes for scales, codebook training corpus.
- Eval loop: pre-capture (K, V) tensors + FP16 reference logprobs once; per iteration only `quantize → dequantize → KL`. This makes the 30–60s/iteration target plausible.
- Use Sonnet (not Haiku) if iterations touch op registration or pybind11; Haiku is fine for pure-config sweeps.
- See `notes/autoresearch.md` if/when this phase is launched.

### Phase E — RotorQuant and QJL (out of scope for v1)

Deferred. Local SYCL kernels for Clifford / Iso / Planar rotations and the QJL projection exist but are not integrated. Revisit after upstream signals (vllm issue #38291 is the RotorQuant tracker); contributing rotation choice as an opt-in `turboquant_rot=<name>` config flag upstream is one path forward.

## Validation infrastructure (already in place)

- `../llamacpp-intel-arc/harness/compute_kl.py` — top-K logprob KL with vocab tail mass handling
- `../llamacpp-intel-arc/harness/capture_logprobs.py` — vLLM logprob capture (greedy, top-K)
- `../llamacpp-intel-arc/harness/build_long_prompts.py` — long-context prompts

These were built for the llama.cpp WHT KV-rotation evaluation but are vLLM-compatible.

## Build environment reminder

SYCL kernels need DPC++ from the FHS oneAPI shell:

```
nix run ../nix-intel-xpu#oneapi-env
source /opt/intel/oneapi/setvars.sh
```

Then `icpx -fsycl` works. The flake here does not provide DPC++ directly — it only checks that `icpx` is on PATH.

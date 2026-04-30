# Phase A result — 2026-04-29

Outcome: **A1 — works.** TurboQuant Triton kernels JIT cleanly on Intel XPU, decode runs, server returns coherent generations.

## Build

Path A executed: forked `vllm-project/vllm` to `jasonboukheir/vllm`, cloned at `/home/jasonbk/Projects/vllm`, built from main HEAD (`6841f5dc7`, ~3000 commits past PR #38479).

```
cd /home/jasonbk/Projects/vllm
podman build --format=docker --pull=missing \
  -f docker/Dockerfile.xpu --target vllm-openai \
  -t vllm-xpu-tq:main-6841f5dc7 .
```

Two notes for reproducing:
- `--format=docker` is required. Without it, podman uses OCI format which doesn't honor the Dockerfile's `SHELL ["bash", "-c"]` directive, and the `source /opt/intel/oneapi/setvars.sh` line during the vllm pip install fails with `/bin/sh: 1: source: not found`.
- First build is fully cold — apt + oneCCL + UMD + UCX + NIXL + vllm C++/SYCL compile. ~30 min on this machine. Subsequent builds with the same `--format` reuse layer cache cheaply.

Final image: 35.4 GB. Includes vllm `0.20.1rc1.dev83+g6841f5dc7`, torch 2.11.0+xpu, IPEX, triton-xpu 3.7.0, vllm-xpu-kernels 0.1.7.

## Boot

```
podman run -d --rm --name vllm-tq-test \
  --device /dev/dri --group-add keep-groups --shm-size=16g \
  -p 8000:8000 \
  -v /home/jasonbk/.cache/huggingface:/root/.cache/huggingface:z \
  vllm-xpu-tq:main-6841f5dc7 \
  Qwen/Qwen3-4B \
  --kv-cache-dtype turboquant_k8v4 \
  --max-model-len 4096 --gpu-memory-utilization 0.85 \
  --enforce-eager --max-num-seqs 4 \
  --host 0.0.0.0 --port 8000
```

Boot signals:

```
[xpu.py:67] Using TurboQuant attention backend.
[default_loader.py:391] Loading weights took 1.76 seconds
[gpu_model_runner.py:4883] Model loading took 7.56 GiB memory and 31.19 seconds
[gpu_worker.py:433] Available KV cache memory: 17.57 GiB
[kv_cache_utils.py:1710] GPU KV cache size: 283,392 tokens
[kv_cache_utils.py:1711] Maximum concurrency for 4,096 tokens per request: 69.19x
[core.py:306] init engine (profile, create kv cache, warmup model) took 10.47 s
INFO:     Application startup complete.
```

The xpu.py:67 line confirms upstream's 6-line dispatch fired and the `TurboQuant` attention backend was selected. KV cache profiled to 283,392 tokens in 17.57 GiB ≈ 0.062 KiB/tok, vs FP16 baseline for Qwen3-4B (36 layers × 8 KV heads × 128 head dim × 2 K/V × 2 bytes = 144 KiB/tok)... wait that's 1000x off. Let me recheck: 36 × 8 × 128 × 2 × 2 = 147,456 bytes/tok = 144 KiB/tok ≈ 0.144 MiB/tok. 17.57 GiB ÷ 144 KiB ≈ 122K tokens at FP16. We got 283K, so ~2.3x compression. Matches K8V4's expected ratio (~2.7x at 8/4 bit average) within profiling overhead.

## End-to-end decode

```
$ curl -s -4 -X POST http://127.0.0.1:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "Qwen/Qwen3-4B", "prompt": "Q: 2+2 equals what?\nA:",
         "max_tokens": 16, "temperature": 0}'
{"choices":[{"text":" 4\n\nQ: 2+2 equals what?\nA: 4", ...}], ...}
```

Coherent output. The decode path exercises `triton_turboquant_decode.py` (and the Triton runtime JIT'd it on XPU without complaint).

Gotcha: curl POST over `localhost` (which resolves IPv6 first) gets `Connection reset by peer` — rootless podman's passt networking is IPv4-only. Use `curl -4` or `127.0.0.1` explicitly.

## Implication for Phase B

Phase B (replace Triton kernels with SYCL ops) becomes **optional perf work**, not mandatory. Triton-on-XPU works. We keep Phase B as a path to lower latency / better Arc/BMG utilization, but the baseline already serves correctness.

## Caveats / next questions

- **Long-context not yet exercised.** Phase A used a 16-token decode at 4K max. PR #38479's selling point is long context; need to drive a 32K+ prompt before claiming the kernel handles real workloads. That's Phase C's KL harness territory.
- **GSM8K eval not run.** Acceptance was "any score OR JIT trace" — we got a single coherent decode, which is stronger than "any score" but weaker than a full GSM8K sweep. Cheap to add: vllm has the eval configs from PR #38479 at `tests/evals/gsm8k/configs/Qwen3-4B-TQ-k8v4.yaml`.
- **Hybrid models unproven.** Qwen3.6 (the actual target) is a Gated DeltaNet + gated full-attention hybrid (3:1). PR #38479's `single_type_kv_cache_manager.py` change targets uniform-type KV caches. Hybrid models go through a different manager. Whether `--kv-cache-dtype turboquant_k8v4` even loads on a hybrid Qwen3.6 model is an open question — should be tested as Phase A.5 before committing to that target.
- **Compression ratio undermeasured.** vLLM's profiler reports the *available* KV memory as 17.57 GiB, but how much of that is the TurboQuant overhead (centroids, scales, rotation matrices) vs raw quantized tensor is not visible. Worth instrumenting before tuning bit allocation.

## Status of Phase A subtasks

- [x] Container / build verified — built from upstream main
- [x] Boot vllm with `turboquant_k8v4` — clean
- [x] Smoke decode — coherent
- [x] Document outcome (this file)

Phase A ✅ done.

# Running Qwen3.6-35B-A3B on an Intel Arc Pro B70 with vLLM

A self-hosted vLLM image with INT4 MoE + TurboQuant KV cache, baked on top of
`vllm-project/vllm` with five XPU patches that aren't upstream yet. Validated
on the Intel Arc Pro B70 (Battlemage, 32 GiB GDDR6).

**Public image:** `ghcr.io/jasonboukheir/vllm-xpu-int4-tq:fcc0c8365`
**Source:** branch `tq-hybrid-allow` at `github.com/jasonboukheir/vllm` (head `fcc0c8365`)
**Model:** `palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4` (24 GiB on disk, GPTQv2 sym int4 group=128)

## What you get

| metric | value |
|---|---:|
| Model VRAM | 20.15 GiB (INT4 routed experts, BF16 attention/embed/lm_head) |
| KV cache headroom @ 32k ctx, util=0.85 | **208k tokens** (6.36× concurrency) |
| Single-stream decode | **~58 tok/s** |
| 8-way aggregate decode (batched) | ~100–130 tok/s |
| KL(FP16-KV ‖ TurboQuant k3v4) at 4096 ctx, top-2000 | 0.018 — top-1 100%, top-5 93% |

For comparison: the older `intel/llm-scaler-vllm:0.14.0-b8.2 + sym_int4 + fp8`
path got ~20 tok/s single-stream and 103k KV tokens at the same context. The
new path is **~3× single-stream + 2× KV headroom** on the same hardware,
matching `llama.cpp` Q4_K_M's hand-tuned SYCL pipeline (60 tok/s).

The single-stream win comes from `torch.compile` + XPU graph capture: the
captured Level Zero command list collapses hundreds of per-kernel CPU
dispatches into one submission per token. The kernels are unchanged, vLLM
just stops paying framework tax. See
[PERF-INVESTIGATION.md](../results/PERF-INVESTIGATION.md) for the bandwidth
math.

## Prerequisites

- Intel Arc Pro B70 (Battlemage / Xe2 / `bmg_g31`)
- A recent kernel + Level Zero stack with `/dev/dri/renderD*` exposed
- ~30 GiB free disk for the image, ~25 GiB for the HF model cache
- `podman` (or `docker` — adjust the flags)

## Quick run (any Linux distro)

```bash
mkdir -p ~/.cache/huggingface

podman run --rm -d --name vllm \
  --device /dev/dri \
  --group-add keep-groups \
  --ipc=host \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/cache:Z \
  -e HF_HOME=/cache \
  -e CCL_ZE_IPC_EXCHANGE=sockets \
  -e CCL_PROCESS_LAUNCHER=none \
  -e CCL_LOCAL_RANK=0 \
  -e CCL_LOCAL_SIZE=1 \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  --entrypoint /bin/bash \
  ghcr.io/jasonboukheir/vllm-xpu-int4-tq:fcc0c8365 \
  -lc 'cd /workspace/vllm && exec vllm serve palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4 \
    --served-model-name qwen3.6-35b-a3b \
    --dtype bfloat16 \
    --port 8000 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 32768 \
    --quantization inc \
    --kv-cache-dtype turboquant_k3v4_nc \
    --max-num-seqs 8 \
    --reasoning-parser qwen3 \
    --compilation-config '"'"'{"cudagraph_capture_sizes":[1,2,4,8]}'"'"' \
    --limit-mm-per-prompt '"'"'{"image":0,"video":0}'"'"''
```

First boot:
- ~12 min to pull the 27 GiB image from GHCR (one-time)
- ~90 s to download `palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4` from HuggingFace (one-time, cached in `~/.cache/huggingface`)
- ~12 s to load weights into VRAM
- ~40 s torch.compile (cached on disk after first boot, subsequent boots ~5 s)
- ~7 s graph capture
- **Total: ~16 min first boot, ~40 s on subsequent boots**

Verify it's alive:

```bash
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-35b-a3b","prompt":"The capital of France is","max_tokens":20,"temperature":0}'
```

## What the flags mean

| flag | why |
|---|---|
| `--quantization inc` | Routes the GPTQv2 int4 weights through INC (Intel Neural Compressor) → `xpu_fused_moe(is_int4=True)` for the routed experts. |
| `--kv-cache-dtype turboquant_k3v4_nc` | TurboQuant: 3-bit MSE-Lloyd-Max keys + 4-bit uniform values. ~2× the KV slots vs FP16 KV; KL divergence vs FP16 is 0.018 with 100% top-1 token agreement (functionally lossless for greedy decoding). |
| `--gpu-memory-utilization 0.85` | Leaves ~5 GiB headroom for compile workspace + graph capture + L0 runtime. Drop to 0.80 if you're co-resident with another vLLM instance. |
| `--max-num-seqs 8` | Caps concurrent requests at 8. With graph capture, this also bounds the profile-pass workspace; set higher if you want more concurrency but you'll need to capture more graph sizes (and that costs VRAM). |
| `--max-model-len 32768` | 32k context. The model supports more, but the KV pool sizes proportionally — pick what your workload needs. |
| `--reasoning-parser qwen3` | Bundled parser; correctly handles the chat template's `enable_thinking` switch. |
| `VLLM_XPU_ENABLE_XPU_GRAPH=1` | The env var that actually enables Level Zero graph capture. Without it `torch.compile` runs but per-kernel CPU dispatch stays in the loop and you get ~20 tok/s. |
| `--compilation-config '{"cudagraph_capture_sizes":[1,2,4,8]}'` | Limits captured batch sizes. The default (`[1,2,…,128]`) costs ~7 GiB of VRAM and OOMs the KV cache budget on a 32 GiB B70. `[1,2,4,8]` keeps it ~1.4 GiB. Beyond batch=8 vLLM falls back to eager submission. |
| `CCL_*` env vars | OneCCL needs these to complete single-card init in seconds rather than hanging on `all_reduce`. |
| `--ipc=host` | Required for the level-zero shared memory path. |

## NixOS recipe

If you're already running NixOS, here's the gist of the
`services.local-llm.vllm` config:

```nix
{ config, lib, pkgs, ... }: {
  services.local-llm = {
    enable = true;
    backend = "vllm";

    vllm = {
      containerImage = pkgs.callPackage ./vllm-xpu-int4-tq-image.nix {};
      workingDir = "/workspace/vllm";
      model = "palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4";
      alias = "qwen3.6-35b-a3b";
      dtype = "bfloat16";
      quantization = "inc";
      kvCacheDtype = "turboquant_k3v4_nc";
      maxModelLen = 32768;
      gpuMemoryUtilization = 0.85;
      enforceEager = false;
      enableXpuGraph = true;            # sets VLLM_XPU_ENABLE_XPU_GRAPH=1
      cudagraphCaptureSizes = [ 1 2 4 8 ];
      reasoningParser = "qwen3";
      limitMmPerPrompt = { image = 0; video = 0; };
      extraArgs = [ "--max-num-seqs" "8" ];
    };
  };
}
```

The image package itself:

```nix
# vllm-xpu-int4-tq-image.nix
{ dockerTools, ... }:
dockerTools.pullImage {
  imageName = "ghcr.io/jasonboukheir/vllm-xpu-int4-tq";
  imageDigest = "sha256:8c51da522fd8296cd31d18361e8a909139f06c4de1c406e185afed593183876d";
  hash = "sha256-/yOKKVROJQoLmR2JhrWgeuFw1GkkPOFoMj/5AoG0DuI=";
  finalImageName = "ghcr.io/jasonboukheir/vllm-xpu-int4-tq";
  finalImageTag = "fcc0c8365";
}
```

The matching `services.local-llm.vllm` module options (`enableXpuGraph`,
`cudagraphCaptureSizes`, `workingDir`) and reasoning-parser plumbing live in
my dotfiles repo at
[github.com/jasonboukheir/nix](https://github.com/jasonboukheir/nix) under
`modules/nixos/services/local-llm/`.

## Tweaks worth knowing

**Want lower-latency single-stream and you don't need 32k context?** Set
`--max-model-len 8192` and `--gpu-memory-utilization 0.92`. KV jumps to
~210k tokens at 8k context (25× concurrency); you can also drop
`--max-num-seqs 8` to take the eager fallback off the table since everything
fits in graph captures.

**Co-resident with an embedding model?** Drop `--gpu-memory-utilization` to
0.80 and run a separate vLLM instance on a different port for the embedding;
0.07 is plenty for `Qwen3-Embedding-0.6B`.

**Single-stream above 58 tok/s?** Hard to push past this on a B70 without
custom kernels — both this path and `llama.cpp` Q4_K_M sit at ~50% of the
B70's effective DRAM bandwidth, the former bounded by framework + Triton
kernels, the latter by hand-tuned SYCL. Going faster requires reading less
per token (e.g. a fully quantized checkpoint where attention is also int4)
or hand-fused decode kernels.

## Caveats

- The image is **not** an upstream vLLM release. It bakes in five patches on
  `vllm-project/vllm` that I plan to upstream but haven't yet:
  `XPUExpertsWNA16`, the WNA16 oracle XPU backend, `INCXPUMoEMethod`, INC's
  GPTQ-on-XPU auto-claim, and INC's GPTQModel `dynamic` field handling.
- Image is squashed (single layer, 27 GiB). Not optimized for size — most of
  the bloat is the OneAPI compiler, dev libraries, and source trees that are
  only needed at build time. A multi-stage rebuild could cut it to ~12 GiB.
- TurboQuant KV cache compression is still a TBD upstream merge for some
  combinations; this image carries a working version. The KV-side
  TurboQuant attention backend is from `vllm-project/vllm` PR #38479.
- Prebuilt for `bmg_g31`. Other Xe2 cards (B580, B770) should work but I
  haven't tested them.
- Single-card only. No tensor-parallel testing yet.

## Useful URLs

- Image: https://ghcr.io/jasonboukheir/vllm-xpu-int4-tq
- Source branch: https://github.com/jasonboukheir/vllm/tree/tq-hybrid-allow
- vLLM XPU RFC: https://github.com/vllm-project/vllm/issues/33214
- TurboQuant KV cache PR: https://github.com/vllm-project/vllm/pull/38479
- Bandwidth/latency analysis: this repo's `vllm-intel-arc/results/PERF-INVESTIGATION.md`

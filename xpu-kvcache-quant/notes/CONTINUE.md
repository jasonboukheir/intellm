# Continuation note — paused 2026-04-29

Pick up here when resuming TurboQuant + Qwen3.6 work.

## Where we are

Phases A, A.5, C complete. Patch #1 landed; patch #2 sketched but not applied. Two strategic forks ahead.

```
Phase A   ✅ TQ on Qwen3-4B (uniform attention) — works on XPU end-to-end
Phase C   ✅ KL harness — TQ k8v4 within paper bounds, 70% greedy agreement at 4k context
Phase A.5 ⏳ TQ on Qwen3.6-35B-A3B (hybrid Gated DeltaNet + full attention)
            ✅ patch #1: boundary skip uses full-attention indices (commit 1207f07d5)
            ❌ patch #2 needed: alignment helper assumes raw-BF16 per-token bytes
            ❓ patch #3 unknown: TQ Triton kernels under hybrid bucketing
```

## Artifacts

| artifact | location |
|---|---|
| forked vllm | `/home/jasonbk/Projects/vllm` (`origin: jasonboukheir/vllm`, `upstream: vllm-project/vllm`) |
| boundary-skip patch | branch `tq-hybrid-allow`, commit `1207f07d5` |
| upstream baseline image | `vllm-xpu-tq:main-6841f5dc7` (35.4 GB) |
| patched image | `vllm-xpu-tq:hybrid-1207f07d5` (35.4 GB) |
| capture script | `/home/jasonbk/Projects/intellm/xpu-kvcache-quant/harness/capture_logprobs_vllm.py` |
| phase results | `xpu-kvcache-quant/notes/phase-a-result.md`, `phase-a5-result.md`, `phase-c-result.md` |
| Qwen3.6 weights cache | `/home/jasonbk/scratch/vllm-cache/hub/models--Qwen--Qwen3.6-35B-A3B` (67 GB) |

## The exact next step (~30-60 min)

**Apply patch #2: TQ-aware `_align_hybrid_block_size`.** Sketch in `phase-a5-result.md` ("The real second patch"). File: `vllm/platforms/interface.py`, function `_align_hybrid_block_size`, lines ~545-555.

```python
# Inside _align_hybrid_block_size, replace the FullAttentionSpec construction
# with a branch that uses TQFullAttentionSpec when cache_dtype is turboquant_*.

if cache_config.cache_dtype.startswith("turboquant_"):
    from vllm.model_executor.layers.quantization.turboquant.config import (
        TurboQuantConfig,
    )
    from vllm.v1.kv_cache_interface import TQFullAttentionSpec
    tq_cfg = TurboQuantConfig.from_cache_dtype(
        cache_config.cache_dtype, model_config.get_head_size()
    )
    attn_page_size_1_token = TQFullAttentionSpec(
        block_size=1,
        num_kv_heads=model_config.get_num_kv_heads(parallel_config),
        head_size=model_config.get_head_size(),
        dtype=kv_cache_dtype,
        kv_quant_mode=kv_quant_mode,
        tq_slot_size=tq_cfg.slot_size_aligned,
    ).page_size_bytes
elif model_config.use_mla:
    # ... existing MLA branch unchanged
else:
    # ... existing plain branch unchanged
```

Then:

```bash
cd /home/jasonbk/Projects/vllm
# (apply edit on tq-hybrid-allow branch)
git add vllm/platforms/interface.py
git commit -m "turboquant: hybrid alignment helper uses TQ slot size for per-token bytes"

# rebuild — most layers cached, only vllm-install layer redoes
podman build --format=docker -f docker/Dockerfile.xpu --target vllm-openai \
  -t vllm-xpu-tq:hybrid-align-$(git rev-parse --short HEAD) .

# retry Qwen3.6 boot — same flags as last attempt
podman run -d --name vllm-tq-qwen36 \
  --device /dev/dri --group-add keep-groups --ipc=host \
  -p 8003:8000 \
  -v /home/jasonbk/scratch/vllm-cache:/root/.cache/huggingface:z \
  vllm-xpu-tq:hybrid-align-<sha> \
  Qwen/Qwen3.6-35B-A3B \
  --kv-cache-dtype turboquant_k8v4 \
  --max-model-len 1024 --gpu-memory-utilization 0.78 \
  --enforce-eager --max-num-seqs 1 --max-logprobs 50 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --cpu-offload-gb 55 \
  --host 0.0.0.0 --port 8000
```

Watch for either:
- Engine init clean → tries decode → either coherent generation (✅ patch #3 not needed) or kernel-level crash (❌ need patch #3 — TQ kernels assume contiguous attention layer indexing).
- New assertion / different error → patch #2 was right but exposed yet another layer.

Useful regex to grep logs for: `TQ:|attention backend|Available KV|init engine|Application startup|AssertionError|RuntimeError|Traceback`.

Note: a **single-GPU 32 GB B70** plus 67 GB BF16 + 14 GiB free KV pool means cpu-offload=55+ is needed, decode will be slow (~1-3 tok/s expected via PCIe weight shuffle). This is correctness-only — not a perf benchmark.

## After patch #2 lands and step 1 works: the strategic fork

To get a *fast* TurboQuant + Qwen3.6 setup on this single GPU, we need online weight quantization, which means combining TQ with `sym_int4` (Intel IPEX). Two paths:

**(a) Backport TQ → llm-scaler v0.14.0**
- Cherry-pick TQ + our patches onto `intel/llm-scaler/vllm/patches/vllm_for_multi_arc.patch`'s base (vllm v0.14.0)
- ~50-100 LOC of TQ code to forward-port, plus has to live alongside the existing 692 KB multi-arc patch
- Faster to a working demo
- Cost: every upstream TQ fix needs to be re-ported

**(b) Forward-port `sym_int4` → upstream main**
- Lift the IPEX `sym_int4` quantization path out of `intel/llm-scaler` into a new `vllm/model_executor/layers/quantization/sym_int4.py` against upstream main
- Bigger one-time effort but TurboQuant stays at HEAD and sym_int4 becomes a clean module
- Could potentially upstream

Recommended: (b) long-term, but (a) if you want a working perf demo soon.

If pursuing (b), starting points:
- `intel/llm-scaler/vllm/patches/vllm_for_multi_arc.patch` is the source of the sym_int4 logic — extract `--quantization sym_int4` handling.
- IPEX's online Q4_0 packing — the `pip install arctic-inference==0.1.1` from the llm-scaler Dockerfile may be involved.
- `~/.config/nix/docs/VLLM.md` has the user's notes on the Intel-specific pieces (`compressed_tensors_moe.py:166` Marlin fallback, AutoRound IPEX path issues, etc.)

## Open questions to revisit

- Does the alignment-helper patch (#2) make the unifier short-circuit (page sizes equal → `len(page_sizes) <= 1`), or does cdiv rounding leave a 1-byte gap that re-triggers it? If the latter, may need to round attention page exactly to mamba page (no overshoot).
- Do TQ Triton kernels (`triton_turboquant_store.py`, `triton_turboquant_decode.py`) handle a sparse layer-index pattern (only every-4th layer has TQ KV)? Likely yes since vLLM's existing `kv_cache_dtype_skip_layers` produces analogous sparseness, but needs verification at decode time.
- Compression ratio in hybrid: only ~25% of layers have TQ-quantized KV (10/40 full-attention out of 40 total). Real KV memory savings are smaller than uniform-attention numbers from Phase A/C. Worth re-running KL harness on hybrid + TQ to see actual quality at hybrid context lengths.

## Resume-from-cold checklist

If picking this up after several days:
- [ ] `cd /home/jasonbk/Projects/vllm && git status` — confirm on `tq-hybrid-allow`
- [ ] `git log --oneline upstream/main..HEAD` — what's the patch series so far?
- [ ] `podman images vllm-xpu-tq` — which images still exist?
- [ ] `git fetch upstream && git log --oneline HEAD..upstream/main -- vllm/v1/attention/backends/turboquant_attn.py vllm/engine/arg_utils.py vllm/platforms/interface.py` — has upstream changed any of the files we touched? If yes, rebase before resuming.
- [ ] Re-read `phase-a5-result.md` for the full diagnostic chain.

## Dead-ends already ruled out

- `--block-size 64`, `--block-size 128` against patched image: hits same `unify_kv_cache_spec_page_size` assertion. Helper's wrong `mamba_page_size_padded` is upstream of any user knob.
- Splicing PR files into `intel/vllm:0.17.0-xpu` or `intel/llm-scaler-vllm:0.14.0-b8.2`: structurally fails on engine/config refactors that landed between their pinned commits and PR #38479's merge. Already documented in `phase-a-result.md`.
- AutoRound on master for XPU: present (`inc.py` formats `auto_round:auto_gptq` / `auto_round:auto_awq`) but routes through `gptq_marlin` / `awq_marlin` Marlin repack which is CUDA-only. Won't work on XPU until someone adds an XPU repack op.
- MXFP4 online (BF16 → MXFP4 RTN): not in upstream. `mxfp4` quant is checkpoint-driven (`hf_quant_cfg["quant_method"] == "mxfp4"`), needs a pre-quantized weights file.

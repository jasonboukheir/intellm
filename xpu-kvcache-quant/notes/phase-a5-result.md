# Phase A.5 result — 2026-04-29

Outcome: **A.5-2 — explicit upstream reject.** TurboQuant cannot run on `Qwen/Qwen3.6-35B-A3B` (or any other hybrid attention + Mamba/linear-attention model) in upstream main. The check is structural, not just missing-kernel.

## What we tried

Boot:
```
podman run -d --name vllm-tq-qwen36 \
  --device /dev/dri --group-add keep-groups --ipc=host \
  -p 8000:8000 \
  -v /home/jasonbk/scratch/vllm-cache:/root/.cache/huggingface:z \
  vllm-xpu-tq:main-6841f5dc7 \
  Qwen/Qwen3.6-35B-A3B \
  --kv-cache-dtype turboquant_k8v4 \
  --tensor-parallel-size 2 \
  --max-model-len 8192 --gpu-memory-utilization 0.90 \
  --enforce-eager --max-num-seqs 4 --max-logprobs 50 \
  --limit-mm-per-prompt '{"image":0,"video":0}'
```

(TP=2 was speculative — host actually has 1 GPU, see "Even if not blocked" below.)

## What happened

Boot died at config validation, not weight load:

```
File "/opt/venv/lib/python3.12/site-packages/vllm/engine/arg_utils.py", line 1705, in create_engine_config
    raise NotImplementedError(
NotImplementedError: TurboQuant KV cache is not supported for hybrid
(attention + Mamba) models. Boundary layer protection requires uniform
attention layers.
```

Source (`vllm/engine/arg_utils.py:1700-1709`):

```python
if resolved_cache_dtype.startswith("turboquant_"):
    if model_config.is_hybrid:
        raise NotImplementedError(
            "TurboQuant KV cache is not supported for hybrid "
            "(attention + Mamba) models. Boundary layer protection "
            "requires uniform attention layers."
        )
    # ... (uniform path) computes boundary skip layers
```

The reason is in the uniform path: TurboQuant auto-skips layers `[0, 1, N-2, N-1]` for boundary protection. For Qwen3.6's 64-layer 3:1 linear-to-full pattern, layers 0, 1, 62 are linear-attention (no standard KV cache), so the absolute index notion of "boundaries" breaks.

`is_hybrid` is true for Qwen3.6 because vLLM's `kv_cache_interface.py` classifies linear-attention layers (Gated DeltaNet) under `MambaSpec` — the error message says "attention + Mamba" but the `is_hybrid` predicate fires on any non-uniform `KVCacheSpec` mix.

## Verified architecture

`/home/jasonbk/scratch/vllm-cache/hub/models--Qwen--Qwen3.6-35B-A3B/.../config.json`:
- `architectures: ["Qwen3_5MoeForConditionalGeneration"]` (registered in upstream main)
- `model_type: "qwen3_5_moe"`
- `full_attention_interval: 4`
- `head_dim: 256`
- `layer_types: [linear_attention, linear_attention, linear_attention, full_attention, ...]` — explicit 3:1 ratio across 64 layers (16 full-attention, 48 linear-attention).

## Patch sketch (not implemented)

`vllm/engine/arg_utils.py:1700-1709` and `vllm/model_executor/layers/quantization/turboquant/config.py:160-175`:

```python
# In arg_utils.py, replace the unconditional reject:
if resolved_cache_dtype.startswith("turboquant_"):
    from vllm.model_executor.layers.quantization.turboquant.config import (
        TurboQuantConfig,
    )
    if model_config.is_hybrid:
        layer_types = getattr(model_config.hf_text_config, "layer_types", None)
        if layer_types is None:
            raise NotImplementedError(
                "TurboQuant on hybrid model requires hf_config.layer_types"
            )
        full_attn_indices = [i for i, t in enumerate(layer_types)
                             if t == "full_attention"]
        boundary = TurboQuantConfig.get_boundary_skip_layers_from_indices(
            full_attn_indices, n=2
        )
    else:
        num_layers = model_config.hf_text_config.num_hidden_layers
        boundary = TurboQuantConfig.get_boundary_skip_layers(num_layers)
    existing = set(cache_config.kv_cache_dtype_skip_layers)
    cache_config.kv_cache_dtype_skip_layers = sorted(
        existing | set(boundary), key=lambda x: int(x)
    )
```

Plus a new helper `get_boundary_skip_layers_from_indices(indices, n)` that returns first-N + last-N of `indices` (instead of `range(num_layers)`).

## Verification work the patch leaves on the table

The patch removes the *config-time* block. Two layers below it still need verification:

1. **`triton_turboquant_store.py` and `triton_turboquant_decode.py`** — do they index the K/V cache by per-layer `layer_idx` only, or do they assume contiguous attention-layer indexing? The metadata builder in `turboquant_attn.py` operates per-layer, so per-layer indexing is the model. Likely fine, but needs a smoke test.
2. **vLLM's hybrid KV cache manager.** Hybrid models use a different cache-group routing. The TurboQuant attention backend is selected per *attention* layer; the linear-attention layers go through `MambaAttentionBackendEnum.GDN_ATTN` instead. Whether the two cache groups co-exist cleanly when one carries quantized cache and one carries SSM state is the real integration question. Best test: run with the patch and a small hybrid model end-to-end (Jamba-Mini or similar), capture logprobs, sanity-check coherence.

## Even if not blocked: weight-fit on single GPU

This box has 1 × Arc B70 (32 GB) — confirmed by `ls /dev/dri/render*` showing only `renderD128` and `renderD129`, where renderD129 is the iGPU and the discrete card is renderD128. So TP=2 isn't an option.

Single-GPU options to fit Qwen3.6-35B-A3B (35B BF16 ≈ 70 GB):

| path | available in upstream main? | XPU? | online from BF16? | Qwen3-MoE? |
|---|---|---|---|---|
| dynamic FP8 (`--quantization fp8`) | yes | yes | yes | yes — but ~35 GB doesn't fit |
| MXFP4 (`--quantization mxfp4`) | yes | yes (`Mxfp4MoeBackend.XPU` / `XPUExpertsMXFp4`) | **no** — checkpoint-driven (`hf_quant_cfg["quant_method"] == "mxfp4"`) | likely yes if checkpoint exists |
| AWQ pre-quantized (`cyankiwi/...AWQ-4bit` cached locally) | yes | **no** — routes through `gptq_marlin_repack` (CUDA-only) | n/a | n/a |
| AutoRound (Intel Neural Compressor, `inc.py`) | yes (formats `auto_round:auto_gptq`, `auto_round:auto_awq`) | **no** — routes through `awq_marlin` / `gptq_marlin` (CUDA-only) | n/a | n/a |
| `--quantization sym_int4` (IPEX) | **no** — only in `intel/llm-scaler-vllm` patch set | n/a | yes (in llm-scaler) | yes |
| BF16 + `--cpu-offload-gb 40` | yes | yes | yes | yes — slow, feasible for testing |

Net: with a single 32 GB GPU and upstream main, the only realistic online quant for Qwen3-MoE is `--quantization fp8 --cpu-offload-gb` (slow) or moving to the `intel/llm-scaler-vllm` image with `sym_int4` (no TurboQuant there).

## What we would need to actually test TQ + Qwen3.6 here

1. Patch upstream's `is_hybrid` reject (above) AND verify the kernel + manager paths.
2. Either:
   a. Wait for / add a non-Marlin XPU INT4 path for Qwen3-MoE (so the AWQ checkpoint loads), or
   b. Add `sym_int4`-style online IPEX quantization to upstream (large effort), or
   c. Use a smaller hybrid model that fits BF16 in 32 GB (e.g. AI21 Jamba-Mini-12B, Zamba2-2.7B, or a hypothetical Qwen3.6-3B-A0.3B if it existed). This validates the TQ-hybrid integration without needing weight quant.

The cleanest next step is **(c) on the patched build** — proves the kernel path on hybrid without the orthogonal weight-quant problem. If we want Qwen3.6-35B-A3B specifically, we need (a) or (b) regardless.

## Re. AutoRound on master

In `vllm/model_executor/layers/quantization/inc.py`, formats `auto_round:auto_gptq` and `auto_round:auto_awq`. Routes through `gptq_marlin` / `awq_marlin` which (per the user's `~/.config/nix/docs/VLLM.md`) hit `torch.ops._C.gptq_marlin_repack` — CUDA-only, no XPU registration. So **AutoRound is technically present on master but does not work on XPU today.** Would require an XPU repack op or an AutoRound XPU-native path.

## Status of subtasks

- [x] Static check on hybrid + TurboQuant integration
- [x] Empirical boot test (failed at config validation as expected from static check)
- [x] This writeup
- [x] Patch upstream's hybrid reject — see addendum below
- [ ] Patch hybrid KV cache manager page-size unification — out of scope for v1

Phase A.5 ✅ done (with negative result + clear remediation path).

## Addendum (later same session): patch lands, next layer surfaces

After writing the original conclusion above, applied the boundary-skip patch on a `tq-hybrid-allow` branch of the vLLM fork (commit `1207f07d5`):

- `vllm/model_executor/layers/quantization/turboquant/config.py`: added `get_boundary_skip_layers_from_indices(attention_layer_indices, n=2)`.
- `vllm/engine/arg_utils.py`: replaced the unconditional `is_hybrid` reject with a hybrid-aware branch that reads `hf_text_config.layer_types`, filters to `full_attention` indices, and computes boundary skips from those.

Rebuilt image as `vllm-xpu-tq:hybrid-1207f07d5`.

### Regression on Qwen3-4B (uniform): clean
`TQ: skipping layers ['0', '1', '34', '35'] for boundary protection (num_layers=36)` — identical KV profile (283,392 tokens), coherent decode. Uniform path unaffected.

### Qwen3.6-35B-A3B (hybrid): patched config validation passes
With aggressive cpu-offload (`--cpu-offload-gb 55 --max-model-len 1024 --gpu-memory-utilization 0.78`) on the single B70:

```
TQ: skipping layers ['3', '7', '35', '39'] for boundary protection (num_layers=40)
   ↑ first/last 2 of full_attention indices [3,7,11,...,39] in 40-layer model
Using TurboQuant attention backend.
Loading weights took 73.58 seconds   (cpu-offload IO)
Model loading took 8.79 GiB memory and 152.598857 seconds
Available KV cache memory: 14.39 GiB
```

Past config, past attention-backend selection, past weight load. **All three layers I worried about cleared.**

### Then: vLLM's hybrid KV cache manager assertion fires

```
File "vllm/v1/core/kv_cache_utils.py", line 1656, in get_kv_cache_groups
    kv_cache_spec = unify_kv_cache_spec_page_size(kv_cache_spec)
File "vllm/v1/core/kv_cache_utils.py", line 1042, in unify_kv_cache_spec_page_size
    assert new_spec.page_size_bytes == max_page_size
AssertionError
```

`unify_kv_cache_spec_page_size` tries to make all layers' `page_size_bytes` equal by scaling `block_size`:

```python
ratio = max_page_size // layer_page_size  # divisibility checked one line up
new_block_size = layer_spec.block_size * ratio
new_spec = replace(layer_spec, block_size=new_block_size)
assert new_spec.page_size_bytes == max_page_size
```

The math works *if* every spec's `page_size_bytes` is linear in `block_size`. Two failure modes I can see in `vllm/v1/kv_cache_interface.py`:

1. **`MambaSpec.page_size_bytes`** (line 530) is `sum(prod(shape) * dtype_size for shape, dtype in zip(shapes, dtypes))` — a fixed sum over Mamba state shapes, **independent of `block_size`**. So `replace(block_size=new)` doesn't change `page_size_bytes` for Mamba layers. If MambaSpec is the smaller spec being grown, the assertion fails.
2. **`AttentionSpec.page_size_bytes`** (line 138) returns `page_size_padded` if it's set — also fixed, ignoring block_size. If TQ's spec carries `page_size_padded` in some layers (e.g. for alignment), same problem.

Most likely it's the first: in Qwen3.6's hybrid layout, MambaSpec for the Gated DeltaNet states has a much larger fixed `page_size_bytes` than TQFullAttentionSpec. The unifier sees Mamba as `max_page_size`, tries to scale TQ's block_size up. Each TQ layer should scale (it's linear), but if the divisibility passes only by chance and the two distinct TQ block sizes from full-attn layers don't all reach exactly `max_page_size` after the multiply, the assertion fires.

Either way, this is **upstream-design territory** — the hybrid manager assumes a specific algebraic relation between `block_size` and `page_size_bytes` that holds for the existing hybrid models (Jamba/Zamba/Bamba where attention has FP16 KV and Mamba state is the constant) but doesn't generalize to a quant'd attention spec mixed with a constant-Mamba spec.

### Three real fixes (none small)

1. **Special-case TQ + Mamba in the hybrid manager.** Let TQ attention live in its own `kv_cache_group` and Mamba state in another, no shared page-size pool. The hybrid manager has bucketing logic; TQ should be its own bucket. Patch is in `vllm/v1/core/kv_cache_utils.py:get_kv_cache_groups`, ~30-50 LOC plus changes to `_get_kv_cache_groups_uniform_page_size` to skip TQ specs.
2. **Tune `--mamba-block-size` so MambaSpec.page_size_bytes is divisible by TQ's** — alignment hack. Worth probing as a 5-min experiment before going to (1).
3. **Make TQ's `AttentionSpec.page_size_padded` align with Mamba's page size.** Wastes memory; not a real fix.

### Recommended next step

Try (2): brute-force probe `--mamba-block-size` values to see if any combination produces clean divisibility. If a value works, that's a non-invasive workaround we can use today. If none works, proceed to (1) as an upstream patch / contribution.

**Probe outcome (later same session):** ❌ no value of `--block-size` works, because the bug is upstream of any block-size knob.

`vllm/platforms/interface.py:_align_hybrid_block_size` builds `attn_page_size_1_token` from a plain `FullAttentionSpec(...)`, not `TQFullAttentionSpec(tq_slot_size=...)`. So for Qwen3.6 + TQ k8v4 head_dim=256:

- helper computed `attn_page_size_1_token = 2 × 1 × 8 × 256 × 2 = 8192` bytes/tok (raw BF16)
- helper logged `Setting attention block size to 2176 tokens to ensure that attention page size is >= mamba page size.`
- helper set `mamba_page_size_padded = 2176 × 8192 = 17,825,792` bytes
- but the **actual** TQ k8v4 page at block=2176 is `2176 × 8 × 388 = 6,754,304` bytes (~2.64× smaller — exactly the BF16/TQ ratio)
- unifier sees max=mamba_padded (17.8 MB), TQ=6.6 MB → ratio not integer → assertion

User-set `--block-size` flag doesn't help because the helper still pads mamba to its (wrong) computed value before the unifier runs. The fix must teach the helper about TQ.

### The real second patch (needed for hybrid TQ to work)

In `vllm/platforms/interface.py:_align_hybrid_block_size`, replace:

```python
attn_page_size_1_token = FullAttentionSpec(
    block_size=1,
    num_kv_heads=...,
    head_size=...,
    dtype=kv_cache_dtype,
    kv_quant_mode=kv_quant_mode,
).page_size_bytes
```

with a TQ-aware path:

```python
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
    # ... existing MLA branch
else:
    # ... existing plain branch
```

~20 LOC. Combined with the boundary-skip patch from earlier, this should make TurboQuant + hybrid models load on a properly-sized GPU.

**Status:** the second patch is sketched but not yet applied. Probe of `--block-size 64` and `--block-size 128` both hit the same assertion (confirming user-knob doesn't help). Tagged image `vllm-xpu-tq:hybrid-1207f07d5` is the boundary-skip-only build; the next iteration would be `vllm-xpu-tq:hybrid+align-<sha>`.

Either way, **the patch on `tq-hybrid-allow` is real progress**: TurboQuant + Qwen3.6 boots through ~85% of the init pipeline now, vs failing at the very first arg_utils check before. The remaining blocker is in vLLM's hybrid memory manager, not in TQ itself.

### Image artifacts produced

- `vllm-xpu-tq:main-6841f5dc7` (35.4 GB) — upstream main, used by Phase A and Phase C.
- `vllm-xpu-tq:hybrid-1207f07d5` (35.4 GB) — patched, hybrid-aware boundary skip. Branch: `tq-hybrid-allow`.

Both retained.

## Memory-worthy facts

- Upstream main rejects TurboQuant on hybrid models at `arg_utils.py:1705`.
- `Qwen3_5MoeForConditionalGeneration` is on upstream registry; Qwen3.6 *can* load there with non-TQ KV.
- `Mxfp4MoeBackend.XPU` exists in upstream main (newer than user's docs). Checkpoint-driven, not online.
- AutoRound in `inc.py` is CUDA-only on XPU (Marlin repack op missing).
- Single B70 in this box (renderD128 = discrete, renderD129 = iGPU).

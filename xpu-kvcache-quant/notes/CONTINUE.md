# Continuation note — paused 2026-04-30

Pick up here when resuming TurboQuant + XPU INT4 work. **Strategy pivoted late this session — read this and the linked memory entries before starting.**

## Where we are

Phases A, A.5, C, D complete. TurboQuant + Qwen3.6 hybrid runs end-to-end. Strategy fork from earlier CONTINUE.md (sym_int4 forward-port) has been resolved: **Intel announced IPEX EOL March 2026 ([issue #867](https://github.com/intel/intel-extension-for-pytorch/issues/867))**, so the IPEX path is permanently stranded and we're now aligned with vLLM RFC [#33214](https://github.com/vllm-project/vllm/issues/33214) (vllm-xpu-kernels migration).

```
Phase A   ✅ TQ on Qwen3-4B (uniform attention) — works on XPU
Phase C   ✅ KL harness — TQ k8v4 within paper bounds, 70% greedy agreement at 4k
Phase A.5 ✅ TQ on Qwen3.6-35B-A3B (4-patch chain on tq-hybrid-allow)
Phase D   ✅ Hybrid TQ k8v4 vs BF16 KL — within paper envelope, +0.183 ce_q-ce_p
            ⚠️ boundary protection disabled on hybrid (patch 5 territory)
sym_int4  ❌ IPEX 2.11+xpu doesn't ship; IPEX retired 2025-08-06; abandoned
WNA16 MoE 🔨 in progress: XPUExpertsWNA16 foundation committed (38237e347)
            still TODO: WNA16 oracle XPU branch + INC MoE method + checkpoint adapter
```

## Patch series state on `jasonboukheir/vllm` `tq-hybrid-allow` (9 commits above upstream main)

```
38237e347 xpu_moe: add XPUExpertsWNA16 wired to xpu_fused_moe(is_int4=True)
db67f73b0 docker: switch sym_int4 image to bigdl-core for the .so + --no-deps IPEX  ⚰️ stranded
2756dd749 docker: sketch Dockerfile.xpu.sym_int4 for IPEX-backed weight quant       ⚰️ stranded
1db471bde sym_int4: model_loader plumbing for CPU-side quantization                 ⚰️ stranded
1aeec4306 sym_int4: forward-port IPEX-backed INT4 weight quant from llm-scaler      ⚰️ stranded
4088d3dd0 turboquant: disable boundary-skip auto-add for hybrid runs                ✅ working
9ad5531f2 kv_cache_utils: unify pads via page_size_padded                           ✅ working
658927c23 turboquant: hybrid alignment helper uses TQ slot size                     ✅ working
1207f07d5 turboquant: relax hybrid-model reject; route via full-attn indices        ✅ working
```

The 4 sym_int4 + docker commits are stranded by IPEX EOL — keep as reference; don't extend.

## The exact next step (~1-2 sessions)

**Wire `XPUExpertsWNA16` into a quant config so a 4-bit GPTQ-Int4 MoE checkpoint actually loads on XPU.** Three sub-tasks:

### 1. WNA16 oracle XPU backend (~50 LOC, `int_wna16.py`)

`vllm/model_executor/layers/fused_moe/oracle/int_wna16.py:35` registers only `WNA16MoEBackend.MARLIN` and `BATCHED_MARLIN`, both CUDA. Add:

```python
class WNA16MoEBackend(Enum):
    MARLIN = "MARLIN"
    BATCHED_MARLIN = "BATCHED_MARLIN"
    XPU = "XPU"   # new

def backend_to_kernel_cls(backend):
    ...
    elif backend == WNA16MoEBackend.XPU:
        from vllm.model_executor.layers.fused_moe.experts.xpu_moe import (
            XPUExpertsWNA16,
        )
        return [XPUExpertsWNA16]

def _get_priority_backends():
    if current_platform.is_xpu():
        return [WNA16MoEBackend.XPU]
    return [WNA16MoEBackend.MARLIN, WNA16MoEBackend.BATCHED_MARLIN]
```

Plus `convert_to_wna16_moe_kernel_format` (line 380) needs an XPU branch that **skips Marlin repack** and just transposes weights into `[E, 2*N, K]` int4-packed layout (per `xpu_fused_moe` docstring's `is_int4` shape requirement).

### 2. INC MoE method (~30 LOC, `inc.py`)

`vllm/model_executor/layers/quantization/inc.py:apply_xpu_w4a16_quant_layer:439` returns `None` for FusedMoE. Add:

```python
if isinstance(layer, FusedMoE):
    return INCXPUMoEMethod(
        weight_bits=weight_bits,
        group_size=group_size,
        sym=sym,
        moe_config=layer.moe_config,
    )
```

`INCXPUMoEMethod` is new — pattern after `Mxfp4MoEMethod` at `mxfp4.py:463`. It needs `create_weights` (allocate the int4-packed parameters), `process_weights_after_loading` (any layout massage), and `select_gemm_impl` returning `XPUExpertsWNA16(...)`.

### 3. Checkpoint adapter

Cached locally: `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` (AWQ — has zero-points, not symmetric — likely NOT compatible without zero-point folding).
HF candidate: `palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4` (24.4 GB, GPTQv2 sym int4 group=128, desc_act=false) — closer to compatible, but layout still needs verification.

Expected layout per `xpu_fused_moe(is_int4=True)`:
```
w13: [num_experts, 2*inter_size, hidden_size]   int4-packed contiguous
w13_scales: [num_experts, 2*inter_size, hidden_size // group_size]   fp16
w2:  [num_experts, hidden_size, inter_size]     int4-packed contiguous
w2_scales:  [num_experts, hidden_size, inter_size // group_size]   fp16
```

Try booting against the GPTQ-Int4 checkpoint; whatever layout error vLLM reports tells you exactly what the adapter needs to do.

## Smoke test command (when steps 1-3 land)

```bash
podman run -d --name vllm-tq-qwen36-int4 \
  --device /dev/dri --group-add keep-groups --ipc=host \
  -p 8005:8000 \
  -v /home/jasonbk/scratch/vllm-cache:/root/.cache/huggingface:z \
  vllm-xpu-tq:wna16-<sha> \
  palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4 \
  --quantization inc \
  --kv-cache-dtype turboquant_k8v4 \
  --max-model-len 4096 --gpu-memory-utilization 0.85 \
  --enforce-eager --max-num-seqs 1 --max-logprobs 50 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --host 0.0.0.0 --port 8000
```

24.4 GB int4 weights should fit on the 32 GB B70 *without* cpu-offload — expecting a major step up from BF16's ~6.8 tok/s.

## After step (1-3) lands: the lower-priority follow-ups

- **Patch 5 — reshape generalization** to re-enable boundary protection on hybrid (`gpu_model_runner.py:6644` strided-view path). Quality improvement; ~20% PPL clawback per Phase D.
- **Long-context Phase D** — 1k and 4k prompts on hybrid TQ + Qwen3.6 (Phase C only ran short on this hybrid build).
- **`_POSSIBLE_FP8_BLOCK_KERNELS` XPU entry** — would unblock `Qwen/Qwen3.6-35B-A3B-FP8` as a separate path. Out of scope unless 4-bit doesn't pan out.

## Resume-from-cold checklist

Read in this order:
- [ ] `project_xpu_int4_moe_path.md` — full state of the WNA16 MoE work
- [ ] `project_tq_hybrid_patches.md` — TQ patch chain on hybrid
- [ ] This file
- [ ] `phase-d-result.md` — KL numbers
- [ ] `cd /home/jasonbk/Projects/vllm && git log --oneline upstream/main..HEAD` — confirm patch series
- [ ] `cat /home/jasonbk/Projects/vllm/vllm/model_executor/layers/fused_moe/experts/xpu_moe.py` — see XPUExpertsWNA16 already in place
- [ ] `git fetch upstream && git log --oneline HEAD..upstream/main -- vllm/model_executor/layers/fused_moe/oracle/int_wna16.py vllm/model_executor/layers/quantization/inc.py` — has upstream landed any of the WNA16 XPU work since? (RFC #33214 milestones int4 GEMM/MoE post-0.16)

## Working baseline today (no patches needed, no IPEX)

`vllm-xpu-tq:hybrid-noskip-4088d3dd0` running Qwen3.6-35B-A3B BF16 + TQ k8v4 + `--cpu-offload-gb 55` at ~6.8 tok/s. Functionally correct, slow. Use this if you need decode RIGHT NOW.

## Dead-ends already ruled out (this session)

- IPEX 2.11+xpu — doesn't exist, won't ship, project EOL.
- IPEX 2.7.10 + torch 2.11 with `--no-deps` — installs cleanly, base oneAPI 2025.3 stays intact, but `import intel_extension_for_pytorch` hard-checks torch major.minor and bails.
- `Qwen/Qwen3.6-35B-A3B-FP8` on XPU — `_POSSIBLE_FP8_BLOCK_KERNELS` table has no XPU entry; needs upstream patch.
- AWQ / GPTQ / GGUF / BnB / moe_wna16 on XPU — all blocked on Marlin (CUDA-only) or unimplemented; documented in agents' research summary in `project_xpu_int4_moe_path.md`.
- Backporting TQ to llm-scaler v0.14.0 — formally rejected earlier (CONTINUE.md option a from previous session); fork divergence too large.

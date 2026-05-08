Tick 4 (2026-05-07 ~18:47 UTC): pipeline-verification tick (no kernel work).
- `swiglu-dev` container removed; only `vllm-dev` remains.
- `pip install -e /workspace/vllm-xpu-kernels` mid-rebuild inside `vllm-dev`
  (started ~18:00 UTC). `ninja -j 11`, current state 906/1206 .o, rate
  collapsed from ~19/min (cold non-template phase) to ~3/min (heavy
  SYCL templates: gdn_attn xe_2, grouped_gemm xe_2, attn xe_2). ETA ~60
  min from 18:47. `MAX_JOBS=2` cap (mentioned in `vllm-xpu-kernels`
  CLAUDE.md) is NOT being applied here — ninja is running -j11; that's
  why ~11 heavy template compiles are in flight at once and per-file
  rate looks low.
- Verified intact while build runs:
  - 200 captures still in container's `/tmp/spec_gdn_captures` (30
    non_spec / 90 spec_K4_min1_max1 / 80 spec_K4_min4_max4).
  - `_spec_gdn_triton_patch.{py,pth}` staged in
    `/opt/venv/lib/python3.12/site-packages/`, byte-equal to
    `/workspace/vllm/.spec-gdn-triton-autotune-patch.py` source-of-truth.
  - `/opt/venv/bin/python` reports torch 2.11.0+xpu, `xpu.is_available()`
    True.
  - One non_spec capture loads cleanly: `ssm_state_post` is fp32, shape
    `(1, 32, 128, 128)` for num_actual_tokens=7, num_prefills=1.
- **Axes-label bug found in `vllm/.spec-gdn-narrow.py`** (read-only
  observation, not yet fixed): the axis loop labels axis 2 as "k" and
  axis 3 as "v", but the SYCL kernel writes
  `ssm_state[head*Hk*Hv + v*Hk + k]` (innermost stride is k; see
  `vllm-xpu-kernels/csrc/xpu/gdn_attn/gated_delta_rule.hpp`
  lines 144-149 / 277-281 / 296-300), so the actual logical layout is
  `(slot, head, v, k)`. That means the tick-3 finding "(head=14, k=79,
  v∈{21,51,82})" should be re-read as "(head=14, v=79, k∈{21,51,82})",
  and "(head=21, k=107, v=4)" as "(head=21, v=107, k=4)". Tile-boundary
  intuition: with sub_group_size=32, v_dim_per_sg=4, head_v_dim=128,
  v=79 maps to v_bucket=2, sg_id=3, j=3 (last per-sg lane); v=107 maps
  to v_bucket=3, sg_id=2, j=3. Both divergent v values fall on the
  *last* j-lane of their sub-group's v-tile — strong hint that the
  sub-group-boundary write in the writeback unroll is the culprit, not
  random bf16 noise. Fix narrow.py labels in next tick before claiming
  more.

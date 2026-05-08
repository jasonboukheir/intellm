2026-05-07 tick 5: build cycle complete; first rung-4 spec replay landed.

**Rebuild done:** `pip install -e /workspace/vllm-xpu-kernels --no-deps
--no-build-isolation` finished after ~1h45m (full SYCL AOT rebuild,
all 600 attn_kernels_xe_2 .o + 3 concurrent libxxx_xe_2.so AOT-link
jobs). Verify in same shell: `has spec_state_indices_tensor: True`,
`loaded: /workspace/vllm-xpu-kernels/vllm_xpu_kernels/_xpu_C.abi3.so`.
Captures intact (200 tuples in container's `/tmp/spec_gdn_captures`,
distribution unchanged). The full rebuild was avoidable — likely a
prior `vllm-xpu-clean` blew the cmake cache. From now on prefer
`vllm-xpu-rebuild` (incremental ninja) for `.hpp`/`.cpp` edits and
only `vllm-xpu-clean` when there's a real reason (toolchain change,
suspected stale config).

**Rung-4 replay** (`spec_K4_min4_max4`, k=3 fully accepted), 3
tested layer_0 tuples, `VLLM_XPU_USE_SYCL_SPEC_GDN=1`:
- **3/3 FAIL** at `core_attn_out` (max abs diff 1.86, 2.41, 2.64 —
  not bf16 noise; ~2 orders of magnitude over atol=2e-2).
- `z` is **byte-exact** (max diff 0.0). RMSNorm/gating/conv path is
  fine; bug is in the gated_delta_rule core only.
- 116 / 16384 cells fail (~0.71%), spread roughly evenly across all
  4 spec tokens (~30 fails each). NOT a "later-token uninitialized"
  pattern — the kernel does write all 4 token rows of core_attn_out.
- 25 of 32 v-heads have at least one fail (max-rate 1.6%), so the
  bug is not localized to one head.
- **The failing v_dim coords are: `[25, 33, 47, 53, 55, 60, 79, 83,
  107, 124]`** — same magic numbers as the prior non_spec
  ssm_state divergence (which after the tick-4 axis-label correction
  was at **v=79** and **v=107**, not k). The worst fails are at
  `(token=*, head=14, v=79)` and `(token=*, head=15, v=79)` across
  all 4 tokens, with sycl values negated/halved vs fla (e.g.
  sycl=-0.90 vs fla=+1.74 → delta 2.64).
- |sycl| at fail cells: max 0.90, mean 0.10; |fla|: max 1.74, mean
  0.45 — SYCL is producing systematically *wrong-magnitude /
  wrong-sign* values at these specific (head, v) coords, not zeros.
- z byte-equal + core wrong at exact v=79/107/etc set ⇒ the spec
  branch's `core_attn_out` writeback (or the source ssm_state it
  reads) has the SAME sub-group-boundary bug previously seen in
  ssm_state_post, but spec mode amplifies its visibility. Tick 4's
  decode for sub_group_size=32 / v_dim_per_sg=4: v=79 → bucket 2
  sg_id 3 j 3, v=107 → bucket 3 sg_id 2 j 3 (last per-sg lane).
- The 10-element fail set in spec mode is broader than the 2-element
  set in non_spec — spec amplifies the same boundary issue across
  more tile boundaries (likely because IS_SPEC iterates the chunk
  loop differently per-token).

Narrow tooling new: `vllm/.spec-gdn-narrow-rung4.py` (loads
`tuple_..._000004_spec_K4_min4_max4.pt`, runs `_call_sycl`, dumps
per-token max/mean, fail-rate per (head, v), worst-20 (token, head,
v) with sycl-vs-fla, and z diff).

Next tick: confirm the writeback boundary hypothesis with intermediate
dumps. Either (a) inspect `ssm_state_post` for the same rung-4 capture
and check whether it diverges at the same `(head, v, k)` coords
(propagation source), or (b) read
`vllm-xpu-kernels/csrc/xpu/gdn_attn/gated_delta_rule.hpp` lines
144-149 / 277-281 / 296-300 (state write loops the prior tick
identified) and find why the `j=3` lane at `v_bucket={2,3}` boundary
is mis-written under IS_SPEC. Don't rebuild yet — narrow first.


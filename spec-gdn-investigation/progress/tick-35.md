2026-05-08 tick 35: **Rung 4 still RED on real data**. The real
bug is in `core_attn_out`, not `ssm_state` — the synthetic
load-readback probe was looking at the wrong output.

**Setup**: pristine source (revert from tick 34), JIT-only build
+ FA2 off via direct `docker exec`. Ran the actual rung 4
ladder:
```
vllm-run -e VLLM_XPU_USE_SYCL_SPEC_GDN=1 python -m pytest \
  /workspace/vllm/tests/kernels/xpu/test_spec_gdn_replay.py -v
```
Total: 400 captures (30 non_spec, 370 spec_K4_*). Result:
**171 failed (all spec), 229 passed (non_spec + xfail
mixed-batch)**. Rung 4 confirmed RED on real data — not a
synthetic-probe artifact.

**Sample failure** (`layers_0...000004_spec_K4_min4_max4`):
```
Failed: core_attn_out (vs inline FLA oracle): SYCL diverges from FLA oracle.
  max abs diff: 1.2227e+00
  mean abs diff: 1.7577e-03
  shape: (4, 32, 128), dtype: torch.bfloat16
  tolerance: atol=0.02 rtol=0.02
```
- `(4, 32, 128)` = (num_spec_tokens=4, num_v_heads=32,
  head_v_dim=128).
- mean abs diff (1.76e-3) is comfortably under the 2e-2
  tolerance — i.e. *most* cells match.
- max abs diff (1.22) blows tolerance by 60×. Localized
  divergence; a few cells are wildly off.

**This sharpens the picture**:
- Real bug is in **core_attn_out**, the per-token attention
  output (`res = sum_i(state_local * q_local)`, computed at
  `gated_delta_rule.hpp:243` with subgroup reduction at line
  248-249). NOT in ssm_state writeback.
- The synthetic load-readback probe (T17-T34) checked
  ssm_state divergence under `g=1, beta=0`. That probe was
  affected by the act_sigmoid overflow bug (T34 finding) and
  was NEVER exercising the same code path that fails in rung 4.
  Pure misdirection — the line-241 chase was on a different
  code path than the actual failure.

**The whole T17-T34 narrowing chain is uninformative for
rung 4**:
- T17-T19 set up the load-readback probe.
- T20-T26 narrowed within that probe to line 241 (state += k*delta).
- T27-T29 explored codegen workarounds for that "line 241 bug".
- T34 found that the probe's bug was act_sigmoid overflow, not
  line 241.

**But rung 4 fails on real data** — so there IS a real bug,
just not where T20+ said it was. The investigation needs to
restart with focus on core_attn_out divergence.

**Tick 36 plan — reorient to the real bug**:
1. Pick the layer-0 worst spec capture
   (`tuple_..._layers_0_..._000004_spec_K4_min4_max4.pt`).
2. Run the SYCL kernel + FLA oracle side-by-side, dump
   core_attn_out for both. Compute element-wise abs diff.
3. Find the (t, h, v) cells with max diff. Look for patterns:
   per-token, per-head, per-v_dim, per-spec-iter.
4. Inspect the kernel's res-accumulation path
   (`gated_delta_rule.hpp:233-249`):
   - res[j] init at 234
   - inner loop res[j] += state_local[j*K+i] * q_local[i]
     at 243
   - subgroup reduction `reduce_over_group(sg, res[j], plus)`
     at 248
5. Is the divergence concentrated in:
   - One spec token index t? (suggests per-iter accumulation
     bug — IS_SPEC=true rolls state forward per-token)
   - One head? (data-dependent; less suggestive of code bug)
   - One v_dim? (suggests v-stride bug)

**Re-frame for rung 4**:
- "v=79 hot lane" from T15 was an artifact of the synthetic
  probe; doesn't apply to real rung 4. Re-derive hot lanes
  from real failures.
- All prior workaround attempts (SIMD-16, struct-wrap, IGC env
  vars) were targeting the wrong bug. Don't pursue them.
- Don't trust the load-readback probe for rung 4 narrowing;
  it tests a different code path.

**Source state**: pristine. Build: pristine JIT-only.
`_xpu_C.abi3.so` mtime 1778230290 (post-tick-34 rebuild).

**Memory savings vs prior path**: instead of pursuing T29.B
(struct-wrap, ~1 day source surgery) or workarounds aimed at
the wrong bug, tick 36 onward will diff core_attn_out and
isolate where in the res-accumulation the divergence appears.

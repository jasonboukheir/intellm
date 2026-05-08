2026-05-07 tick 18: load vs decay split — bug is in LOAD, decay ruled
out.

Wrote `vllm/.spec-gdn-noopdecay-probe.py`. Set `A_log = -100` so
`exp(A_log) ≈ 0` → `g = exp(0) = 1.0` exactly (sanity-confirmed:
`g[head=14] = 1.0000000000`). Re-ran SYCL and inline FLA oracle on
the same layer-0 worst capture with this modified A_log on the real
captured init state.

  Run                       v=79 max   v=79 fail   total core fail
  baseline (real A_log)     1.223      52          77/16384
  g≡1 (A_log=-100)          **1.229**  65          90/16384

The fault is unchanged in magnitude (1.229 vs 1.223 — slight uptick
because the real `g≈0.998` mildly contracts the wrong-loaded values;
g≡1 doesn't, so error is slightly larger). All secondary v_dim
positions are also unchanged (v=60 0.600, v=33 0.537, v=55 0.363,
etc., bit-for-bit). The whole ~94% component is purely LOAD.

**Verdict**: bug is in `gated_delta_rule.hpp:144-149`:
```
state_local[j * k_bucket_size + i] =
    static_cast<float>(ssm_state_ptr
                           [num_v_heads_id * head_k_dim * head_v_dim +
                            (k_bucket_size * sg_local_id + i) +
                            (head_v_dim_id + j) * head_k_dim]);
```

The compiled SYCL kernel reads the WRONG bf16 from `ssm_state` at
some `(num_v_heads_id, head_v_dim_id + j, sg_local_id, i)` tuples,
dominantly `(*, 79, *, *)`. The address arithmetic is symmetric with
the writeback (lines 277-281), and the writeback was correct on the
zero-state run, so it's not an address-arithmetic bug shared between
load and store. It's specific to the load (likely the bf16→fp32
conversion or a vectorized-load codegen quirk at j=3).

**Plan for tick 19**: signature-injection probe.
1. Set `ssm_state_pre[load_slot, h, v, k]` to a known unique pattern
   (e.g. `value = h * 100000 + v * 1000 + k`).
2. Set `A_log = -100` (g=1) and `b = -100` (so `beta = sigmoid(-100) ≈ 0`).
   Then iter 0 reduces to `state_post = state_pre` (since
   `delta = (v_t - kv_mem) * beta ≈ 0` and `state_post_decay =
   state_pre * 1`).
3. Read back `ssm_state_post` from any spec slot and decode each
   cell — if cell at `(h=14, v=79, k=K)` decodes to `(h', v', k')`
   for `(h', v', k') ≠ (14, 79, K)`, that names the load-side
   coordinate mix.
4. The WRITE path uses identical address arithmetic, so the readback
   reflects what was LOADED into state_local at iter 0. Any disagreement
   is a LOAD bug. Caveat: if both load AND store use the same broken
   codegen, they could cancel. We've already evidence the store is fine
   (zero-state writeback was clean), so this caveat is unlikely.

5. As a back-off, if the readback is also clean (suggesting some
   numerical interaction with state magnitude), repeat with very
   large signatures (e.g. h*1e5+v*1e3+k scaled to ~1e3).

Cheap, no rebuild, single-call SYCL + decode. Should yield the exact
load-side coordinate-mix table or rule out coordinate-mix entirely
(making it a pure load-value-corruption bug, e.g. bf16 conversion).

Tools added: `vllm/.spec-gdn-noopdecay-probe.py`.


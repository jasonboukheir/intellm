2026-05-07 tick 14: NOT a coordinate-mix; (h=14, v=79) ↔ (h=15, v=79)
paired.

Wrote `vllm/.spec-gdn-iter0-vmatch.py`. For SYCL's iter-0 state at
(h=14, v=79), enumerate all (h_cand, v_cand) in the v_head × v_dim
grid and find the closest pyref iter-0 state. Result:

  pyref(h=14, v=79)  vs  sycl(h=14, v=79):  max_diff = 4.32  <-- target
  pyref(h=15, v=79)  vs  sycl(h=14, v=79):  max_diff = 4.35  paired
  pyref(h= 9, v=79)  vs  sycl(h=14, v=79):  max_diff = 11.65 (gap)
  pyref(h= 8, v=79)  vs  sycl(h=14, v=79):  max_diff = 11.94
  ... every other coord >= 12.5

Probe across the v_cand axis only:
  pyref(h=14, v=79)  max_diff = 4.32  <-- target
  pyref(h=14, v=10)  max_diff = 12.82  (gap)
  pyref(h=14, v=12)  max_diff = 12.83
  ... rest >= 12.84

**Conclusions**:
1. The bug is NOT a (head, v) coordinate-mix at the output. SYCL's
   `state_local` at (h=14, v=79) is closer to pyref's same-coord
   value than to ANY other coord (next-best gap 8 units = 2× the
   bug). If SYCL were "writing v=78's result into v=79's slot" or
   "reading state_init at v=78", we'd see a much closer match
   somewhere else. We don't.
2. The bug ISN'T "skipping iter-0 update" either — state_init at
   the same coord is at max_diff 4.64, slightly worse than pyref's
   post-update at the same coord (4.32). So SYCL DID update; it
   updated to the wrong value.
3. **Heads 14 AND 15 are paired**: their max_diff to SYCL(14, 79)
   are 4.32 and 4.35 respectively — basically identical. Every other
   head, including the 6 other v_heads sharing k_head=1
   (`kh_per_vh = vh // 8`, so heads 8..15 all use k[t][1]), has
   max_diff > 11.6.
   - This rules out "wrong k_norm at k_head=1": that would couple
     ALL 8 of heads 8..15, not just (14, 15).
   - It points to a finer-grained structural pairing — possibly a
     work-item that handles `(head_pair = head//2)` and has a bug
     at one specific (head_pair=7, …) position. With num_v_heads=32
     and group_size=256, group_size/num_v_heads=8 → each v_head
     gets 8 work-items. Those 8 might split (head, head+1) ×
     (v_dim_per_sg=4) somehow.
   - Tick 5 retroactively confirms this pairing: "the worst fails
     are at (token=*, head=14, v=79) and (token=*, head=15, v=79)
     across all 4 tokens".

Notebook for sycl iter-0 at (h=14, v=79): |state| = 12.87
(pyref 13.73), 4.32-unit max-cell error scattered across many k
positions (tick 13's k-coord breakdown showed 11+ k positions with
> 1-unit errors).

**Refined hypothesis**: a per-(h, v) value error caused by a bug
specific to certain index pairs at the SYCL work-item level. The
pairing (h=14, h=15) suggests the kernel processes two consecutive
v_heads per work-item, and a bug at one of the two halves leaks
into the other (e.g. a register/shared-memory aliasing between the
two halves at certain write offsets).

Tools added: `vllm/.spec-gdn-iter0-vmatch.py` — reusable for any
(h, v) target.

Next tick (preferred): enumerate ALL failing (h, v) pairs in
SYCL-vs-pyref at iter 0 for the layer-0 capture, and check whether
they cluster as even-odd v_head pairs (h=14↔15, h=8↔9, …) or as
v_dim pairs (v=78↔79, …) at the same head. The pairing structure
will name the work-item bug locus.

Plan:
- Modify iter0-vmatch (or a sibling) to compute the iter-0 fail mask
  across the full (H, V, K) grid, then aggregate by:
  - per even-odd v_head pair: did BOTH (h, h^1) fail? At which (v, k)?
  - per (h, v_pair) where v_pair groups (v, v^1) or (v, v^2): same.
- Print the (head_pair, v_dim) tuples where both halves fail with
  similar amplitude.

If even-odd v_head pairs co-fail almost everywhere → the work-item
processes 2 heads, and the bug is in their sharing.

Don't rebuild. Don't touch hpp.


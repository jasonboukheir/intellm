2026-05-07 tick 15: even-odd head-pair rejected; v=79 is universal hot
sub-group-boundary lane.

Wrote `vllm/.spec-gdn-iter0-pairs.py`. Enumerated SYCL-vs-pyref iter-0
delta across the full (H=32, V=128, K=128) state grid for the layer-0
rung-4 capture.

**Top-line numbers**: 2341 / 524288 cells fail (0.45%); max abs diff
4.633 (NOT 4.32 — this run's worst was at h=16, v=79); mean 1.17e-3.
Tick 13's "(h=14, v=79)" was the worst at h=14, but h=16 is even worse.

**Top 30 (h, v) by max-cell error, total 72 such cells**:
- 21 of top 30 are v=79 (across every head 0..31).
- Outside v=79, the worst are: (h=24, v=47) max 1.81; (h=25, v=25)
  max 0.93; (h=21, v=107) max 0.80; (h=28, v=124) max 0.79.

**Even-odd head-pair hypothesis (tick 14): REJECTED.**
- 13 (h, h^1) pairs co-fail at v=79, but the within-pair amplitude
  ratios are highly asymmetric (e.g. (16, 17) at v=79: lo=4.63, hi=0.61
  → ratio 7.66). When BOTH halves fail it's because v=79 is universal,
  not because the kernel pairs them.
- Solo fails (only one of h, h^1 hits at this v): 46
- Paired fails (both h and h^1 hit at same v): 26
- More fails are solo than paired ⇒ no even-odd pairing structure.
- The tick-14 "h=14 ↔ h=15 paired" reading was a misinterpretation:
  yes, both fail at v=79, but so do all 30 other heads. There's no
  pairing — there's a v=79 universal.

**v-pair (v, v^1) at same head**: only 2 such pairs (h=19 v=(32,33),
h=18 v=(80,81)), both very small. Also no structural pairing along the
v axis.

**j-lane (v % 4) breakdown**:
  j=0: 10 cells   j=1: 13 cells   j=2: 10 cells   j=3: **39 cells**
  ⇒ j=3 (last lane per v_dim_per_sg=4) is **3-4× more common** than
  other lanes. Confirms tick 9's j-lane bias hypothesis (though j=2
  is not "protected" as tick 9 said — it has 10 cells).

**v_bucket (v // 32) breakdown**:
  bucket 0 (v=0..31):   9 cells
  bucket 1 (v=32..63): 14 cells
  bucket 2 (v=64..95): **41 cells**
  bucket 3 (v=96..127): 8 cells
  ⇒ Bucket 2 has more than 4× any other bucket.

**Heads with any iter-0 fail**: ALL 32 heads (0..31). So the bug is
not localized to any subset of v_heads. It hits every head, just at
different amplitudes (depending on each head's input values at the
hot v_dim positions).

**Refined model of the bug**: a structural defect in the SYCL kernel's
iter-0 compute at specific v_dim positions, dominated by v=79 (and
secondarily v=47, 55, 60, 83, 107, 124 etc). v=79 sits at:
- bucket 2 (v=64..95): the bug's hottest bucket.
- j=3 (last lane per v_dim_per_sg=4): the bug's hottest j-lane.
- sg_id=3 within bucket 2 (sg_3 covers v=76..79 if sg covers 4 vals):
  the LAST sub-group of the first half of bucket 2.

So v=79 is the "boundary" lane: last j of last-of-first-half sub-group
within a heavily-failing bucket. This screams sub-group-boundary
write/read aliasing in `state_local[j*K + k]` accesses.

The "value-sensitive" amplitude observation is consistent: the bug
"poisons" certain v_dim lanes by mis-mixing data; the visible delta
size at any given (h, v) cell is set by what data was mis-mixed
(input values). Layer 0 has bigger state magnitudes at v=79 than
e.g. layer 32 → bigger visible delta at the same lane.

**Concrete narrowing target**: read `gated_delta_rule.hpp` lines that
index `state_local[j * K + k]` and look for any code that uses j=3 in
a way that differs from j=0/1/2 — e.g., a manual unroll that handles
j=3 separately, a shuffle/reduction with a stride that wraps at j=3,
or a write to `state_local[(j+1) * K + k]` that should be no-op for
j=3 but isn't. Tick 9 cited lines 145-149 / 277-281 / 296-300; line
156-158 had the asymmetric `state_local[i * v_dim_per_sg + j]` index
(transposed). That asymmetry is suspicious — even if the no-init
branch isn't taken, the COMPILER may have inferred a layout from it.

Tools added: `vllm/.spec-gdn-iter0-pairs.py`.

Next tick (preferred): read `gated_delta_rule.hpp:121-300` end-to-end
focused on j=3-specific code paths and the line-156 transposed index.
Identify ANY code where j=3's code path differs from j∈{0,1,2}. Don't
rebuild yet — narrow in source first.

Plan:
1. Open gated_delta_rule.hpp:121-300.
2. List every `state_local[<expr>]` access and decompose the index.
3. Any access whose stride or wrap behavior differs at j=3 → suspect.
4. Cross-check against the captured behaviour (j=3 has 39 fails vs 10
   for other j's).

If a smoking gun appears in source → tick 17 is a 1-line hpp fix +
incremental rebuild (vllm-xpu-rebuild, ~270s if cmake cache survives)
+ rerun bug-footprint to verify.

If no smoking gun in source → tick 17 is to dump SPIR-V/LLVM IR for
the specific kernel template instantiation and look at the actual
generated code. That requires kernel rebuild with -save-temps or
similar.


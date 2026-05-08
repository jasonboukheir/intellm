2026-05-07 tick 10: pyref-of-hpp ≡ FLA recurrence; SYCL bug is real but
smaller than thought; the captured "FLA oracle" is itself unreliable.

Built three implementations of the same gated_delta_rule algorithm and
ran all three on the same rung-4 capture inputs:
1. `pyref_hpp` — pure-PyTorch transcription of `gated_delta_rule.hpp`'s
   compute (`vllm/.spec-gdn-pyref.py`).
2. `FLA-recurrence` — vllm's actual Triton kernel
   `fused_sigmoid_gating_delta_rule_update`, called inline.
3. `SYCL native` — `_xpu_C.gdn_attention` via `_call_sycl`.

Cross-check (`vllm/.spec-gdn-cross-check.py`):
- **`pyref_hpp` ≡ `FLA-recurrence` byte-equal**: max diff 5.96e-8 in
  core_attn_out, 2.86e-6 in ssm_state_post. fp32 round-off only.
- **SYCL diverges from FLA-recurrence**: 77/16384 core cells fail (max
  abs 1.22), 10234/2097152 ssm cells fail (max abs 6.00). Not bf16 noise.
- pyref vs SYCL: 77/10247 fails — i.e., SYCL diverges from the
  algorithm-as-written by exactly the same margin it diverges from FLA.
- Pyref/FLA/SYCL all also disagree with FLA-CAPTURED (108-116 core fails
  vs CAPTURED, 14489-14584 ssm fails). FLA-recurrence ≠ FLA-CAPTURED by
  108 cells max 2.58 — so the captured oracle is NOT reproducible by
  re-running the same kernel on the same inputs.

Implications:
- The "algorithm bug" narrative from tick 9 is wrong. `gated_delta_rule.hpp`'s
  formulas are mathematically equivalent to FLA's `fused_sigmoid_gating`
  Triton kernel. The pure-Python transcription of hpp lands byte-equal to
  the FLA Triton output. The math is correct.
- The SYCL kernel's COMPILED OUTPUT diverges from what its own source code
  describes — that's the codegen bug tick 9 hypothesised. Real divergence
  is 77/10234 cells, not the 116/14585 we'd been quoting (that figure
  conflated two bugs).
- The captured `core_attn_out`/`ssm_state_post` (the "FLA oracle") is NOT
  the output of `_forward_core` re-run on the same inputs. There's an
  unidentified discrepancy between capture-time and replay-time. Could be
  non-determinism in the FLA Triton kernel (autotune cache differences,
  reduction ordering across launches, or — most likely — a piece of input
  state in `_forward_core` that I haven't reproduced in the replay).
- Therefore the existing replay test compares SYCL against an unreliable
  oracle. Even a perfectly-correct SYCL kernel would fail the test.

**Side-finding (asymmetric slot anchor)**: my initial pyref's conv1d
read state from `spec_state_indices_tensor[seq, n_acc-1]` — the load
slot used by `gated_delta_rule.hpp` for the SSM state. But FLA's
`causal_conv1d_update` (and the SYCL native conv) read from
`spec_state_indices_tensor[seq, 0]` — the FIRST spec slot, which is
where the rolled conv state lives. Conv anchor and SSM anchor are
DIFFERENT slots in the spec layout. Worth keeping in mind for any
future analysis (and not changing this; it's a model-pipeline contract).

Per-token fail pattern in conv with the wrong anchor was [4956, 2263,
1120, 0] — token 3 reads zero conv state, so it had zero fails;
matches "wrong conv state values" exactly.

Heads-up: the test in `tests/kernels/xpu/test_spec_gdn_replay.py` uses
the captured FLA outputs as the oracle, so it's currently checking
against a non-reproducible target. Any further "SYCL diverges from FLA"
work has to either (a) replace the oracle with an inline FLA-recurrence
call at test time, or (b) figure out why FLA-recurrence on captured
inputs ≠ FLA-CAPTURED.

Next tick (preferred): switch the replay harness to compute the oracle
inline at test time by calling `fused_sigmoid_gating_delta_rule_update`
on the captured `mixed_qkv` (after running `causal_conv1d_update`),
rather than reading `payload["core_attn_out"]` / `payload["ssm_state_post"]`.
That gives a stable oracle. Then re-run rung-4 — should drop SYCL fails
from 116→77 (core) and 14584→10234 (ssm). After that, narrow the actual
77-cell SYCL bug (likely tile-boundary codegen as tick 9 hypothesised).

Alternative tick: reproduce the FLA-recurrence-vs-FLA-CAPTURED
discrepancy in isolation — recapture the same prompt twice, diff. If
the captured outputs differ between two runs, FLA Triton has runtime
non-determinism we need to characterise (seed, autotune timing, or
similar).

Don't touch hpp's formulas — they're correct. The SIMD/codegen
investigation should focus on the COMPILED kernel's behavior at the
77-cell divergence locus.


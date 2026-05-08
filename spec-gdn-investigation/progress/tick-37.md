2026-05-08 tick 37: **HUGE pivot — bug is in
`causal_conv1d_update`, not `gated_delta_rule`.**

Extended `vllm/.spec-gdn-coreattn-diff.py` to also diff
`conv_state_post` per spec slot. Two captures probed:

**Layer 0 K=4 (`min4_max4`, the one I've been chasing)**:
| Output            | max abs   | mean abs | #cells > 2e-2 |
|-------------------|-----------|----------|---------------|
| core_attn_out     | 1.22      | 1.76e-3  | 77 / 16384    |
| conv_state slot 1 | **47.94** | 0.89     | **39700**     |
| conv_state slot 2 | 16.00     | 0.023    | 4752          |
| conv_state slot 3 | 16.00     | 0.034    | 7202          |
| conv_state slot 4 | 15.44     | 0.035    | 6976          |
| ssm_state slot 1  | 4.63      | 1.17e-3  | 2728          |
| ssm_state slot 2  | 4.04      | 1.38e-3  | 2909          |
| ssm_state slot 3  | 6.00      | 1.64e-3  | 3128          |
| ssm_state slot 4  | 4.98      | 1.38e-3  | 2834          |

**Layer 28 K=1 (`min1_max1`, originally a different rung-4
failure)**:
| Output            | max abs   | mean abs | #cells > 2e-2 |
|-------------------|-----------|----------|---------------|
| core_attn_out     | 2.4e-4    | 1.8e-7   | 0 (PASS)      |
| conv_state slot 1 | **14.33** | 0.415    | **24092**     |
| conv_state slot 2 | 14.63     | 0.161    | 13438         |
| conv_state slot 3 | 4.03      | 0.021    | 8020          |
| conv_state slot 4 | 4.03      | 0.019    | 7766          |
| ssm_state slot 1  | 2.1e-3    | 1.4e-6   | 0 (PASS)      |
| ssm_state slot 2  | 1.8e-3    | 1.4e-6   | 0 (PASS)      |
| ssm_state slot 3  | 1.9e-3    | 1.4e-6   | 0 (PASS)      |
| ssm_state slot 4  | 1.6e-3    | 1.4e-6   | 0 (PASS)      |

**Reframing**:
- **Conv state writeback is broken in BOTH captures.** The bug is
  in `causal_conv1d_update` (not gated_delta_rule).
- For K=1 (layer 28), only conv state is wrong; downstream
  (gated_delta_rule's ssm_state + core_attn_out) is fine. So
  the conv1d_update is producing **correct q/k/v outputs** for
  the spec tokens — only the **state writeback to memory** is
  wrong.
- For K=4 (layer 0), conv state is also wrong, AND downstream
  (ssm_state + core_attn_out) is wrong. **Two sub-cases**:
  - Either conv1d_update produces wrong q/k/v outputs in this
    case (different code path with K=4),
  - Or gated_delta_rule reads conv_state from memory (it
    shouldn't — q/k/v are passed through registers/buffers,
    not via the state pool), and the wrong conv state corrupts
    something.

**The K=4 vs K=1 distinguisher**:
- num_accepted_tokens: 4 vs 1
- load_idx (in both conv1d_update and gated_delta_rule): 3 vs 0
- This affects which conv-state slot the kernel LOADS as the
  starting point.

**T17-T34 was on the wrong kernel**. ALL the line-241 work
(volatile-temp, JIT vs AOT, IGC env vars, SIMD-16, struct-wrap
plans) was investigating gated_delta_rule. The actual bug is
upstream in causal_conv1d_update.

**Tick 38 plan — locate the conv1d bug**:
1. Read `csrc/xpu/gdn_attn/causal_conv1d.hpp` (or wherever the
   spec causal_conv1d_update lives in vllm-xpu-kernels).
2. Compare against FLA's
   `vllm.model_executor.layers.mamba.ops.causal_conv1d::causal_conv1d_update`
   — which is the oracle the test uses.
3. Look specifically at:
   - State load: where does it read from for IS_SPEC?
   - Per-token state writeback: where does it store, and for
     which slot?
   - Stride / layout: is `is_conv_state_dim_first()` honored?
     (Test transposes for FLA path — `dim_first` flag.)
4. Cheap probe: run a K=1 capture, dump conv_state per slot,
   look at WHICH cells of the 8192-wide conv state diverge.
   Pattern (e.g., contiguous block at one end) would point to
   a stride/bounds bug.

**Source state**: pristine. Build state: pristine JIT-only
(mtime 1778230290), with `causal_conv1d.hpp` source unchanged.

"""Verify: native kernel passes for seq_len=1 (production decode case) but
fails for seq_len=4. Run the rung-4 capture with the same inputs but
constrained to a single batch's first token."""
import sys
import torch
import vllm._xpu_ops  # noqa: F401
import vllm_xpu_kernels._xpu_C  # noqa: F401

sys.path.insert(0, "/workspace/vllm")
from tests.kernels.xpu.test_spec_gdn_replay import _build_dense_pool

PATH = "/tmp/spec_gdn_captures/tuple_language_model_model_layers_0_linear_attn_000004_spec_K4_min4_max4.pt"

p = torch.load(PATH, map_location="cpu", weights_only=False)
device = torch.device("xpu")
cfg = p["layer_config"]

qkvz = p["projected_states_qkvz"].to(device)
ba = p["projected_states_ba"].to(device)

def _diff_summary(label, sycl, fla):
    delta = (sycl - fla).abs()
    tol = 2e-2 + 2e-2 * fla.abs()
    fail = delta > tol
    print(f"  {label}: max diff={delta.max().item():.3e}  mean={delta.mean().item():.3e}  "
          f"fails={int(fail.sum())}/{fail.numel()} ({100*fail.float().mean():.2f}%)")

# === Run with seq_len=1 (just token 0) ===
print("=== Seq_len=1 (only token 0) ===")
conv_pool, ssm_pool, remap, _ = _build_dense_pool(p, device)
core = torch.empty(p["core_attn_out"].shape, dtype=p["core_attn_out"].dtype, device=device)
z = torch.empty(p["z"].shape, dtype=p["z"].dtype, device=device)

conv_w = p["conv_weight"].to(device)
if conv_w.dim() == 3:
    conv_w = conv_w.view(conv_w.size(0), conv_w.size(2))

# Single-token decode: query_start_loc=[0, 1], seq_len=1
qsl_1 = torch.tensor([0, 1], dtype=torch.int32, device=device)
load_slot = int(p["spec_state_indices_tensor"][0, int(p["num_accepted_tokens"][0]) - 1].item())
load_slot_dense = remap[load_slot].item()
non_spec_idx = torch.tensor([load_slot_dense], dtype=torch.int32, device=device)

torch.ops._xpu_C.gdn_attention(
    core[:1], z[:1],
    qkvz[:1].contiguous(), ba[:1].contiguous(),
    cfg["num_k_heads"], cfg["num_v_heads"], cfg["head_k_dim"], cfg["head_v_dim"],
    num_prefills=0, num_decodes=1,
    has_initial_state=None,
    non_spec_query_start_loc=qsl_1,
    non_spec_state_indices_tensor=non_spec_idx,
    num_actual_tokens=1,
    conv_state=conv_pool, ssm_state=ssm_pool,
    conv_weights=conv_w,
    conv_bias=p["conv_bias"].to(device) if p["conv_bias"] is not None else None,
    activation=cfg["activation"],
    A_log=p["A_log"].to(device),
    dt_bias=p["dt_bias"].to(device),
    tp_size=cfg["tp_size"],
    reorder_input=not cfg["gqa_interleaved_layout"],
)
torch.xpu.synchronize()

sycl_core = core[:1].to("cpu").float()
fla_core = p["core_attn_out"][:1].float()
_diff_summary("core_attn_out token-0 only", sycl_core, fla_core)

# Per-v-dim diff — does v=79 still fail?
delta = (sycl_core - fla_core).abs()
tol = 2e-2 + 2e-2 * fla_core.abs()
fail = delta > tol
per_v = fail.float().sum(dim=(0, 1))
v_set = sorted([int(i) for i, n in enumerate(per_v) if n > 0])
print(f"    failing v_dim positions: {v_set[:30]}")

# Note: token-0 fla output of single-batch-seq=1 vs multi-token-batch
# should be identical (same q[0], k[0], v[0] etc., same initial state).
# So fla_core[:1] is the right reference.

# Show top failing cells
fail_idx = fail.nonzero()
ranked = sorted([tuple(int(x) for x in idx.tolist()) for idx in fail_idx],
                key=lambda c: -delta[c].item())
print(f"\nworst 10 failing (token, head, v_dim):")
for c in ranked[:10]:
    s = sycl_core[c].item(); f = fla_core[c].item()
    print(f"  {c}: sycl={s:+.4e} fla={f:+.4e} delta={abs(s-f):.4e}")

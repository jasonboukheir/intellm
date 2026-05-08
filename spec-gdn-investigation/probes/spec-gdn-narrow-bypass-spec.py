"""Run the rung-4 capture two ways through the SAME native gated_delta_rule
kernel: with and without IS_SPEC. Compare core_attn_out vs FLA. If both
fail at v=79, the bug is in the shared chunk loop, not in IS_SPEC."""
import sys
import torch
import vllm._xpu_ops  # noqa: F401
import vllm_xpu_kernels._xpu_C  # noqa: F401

sys.path.insert(0, "/workspace/vllm")
from tests.kernels.xpu.test_spec_gdn_replay import _build_dense_pool, _remap_index_tensor

PATH = "/tmp/spec_gdn_captures/tuple_language_model_model_layers_0_linear_attn_000004_spec_K4_min4_max4.pt"

p = torch.load(PATH, map_location="cpu", weights_only=False)
device = torch.device("xpu")
cfg = p["layer_config"]
n_actual = int(p["num_actual_tokens"])

def build_outputs():
    return (
        torch.empty(p["core_attn_out"].shape, dtype=p["core_attn_out"].dtype, device=device),
        torch.empty(p["z"].shape, dtype=p["z"].dtype, device=device),
    )

def common_kwargs(conv_pool, ssm_pool):
    conv_w = p["conv_weight"].to(device)
    if conv_w.dim() == 3:
        conv_w = conv_w.view(conv_w.size(0), conv_w.size(2))
    return dict(
        conv_state=conv_pool,
        ssm_state=ssm_pool,
        conv_weights=conv_w,
        conv_bias=p["conv_bias"].to(device) if p["conv_bias"] is not None else None,
        activation=cfg["activation"],
        A_log=p["A_log"].to(device),
        dt_bias=p["dt_bias"].to(device),
        tp_size=cfg["tp_size"],
        reorder_input=not cfg["gqa_interleaved_layout"],
    )

qkvz = p["projected_states_qkvz"].to(device)
ba = p["projected_states_ba"].to(device)

def _diff_summary(label, sycl, fla):
    delta = (sycl - fla).abs()
    tol = 2e-2 + 2e-2 * fla.abs()
    fail = delta > tol
    n_fail = int(fail.sum())
    print(f"  {label}: max diff={delta.max().item():.3e}  mean={delta.mean().item():.3e}  "
          f"fails={n_fail}/{fail.numel()} ({100*n_fail/fail.numel():.2f}%)")
    if n_fail > 0:
        # v_dim positions of fails (axis 2 in core_attn_out)
        per_v = fail.float().sum(dim=(0, 1))
        v_set = sorted([int(i) for i, n in enumerate(per_v) if n > 0])
        print(f"    failing v_dim positions: {v_set[:20]}")

# === Run 1: IS_SPEC=true (matches existing rung-4 result) ===
print("=== Run 1: IS_SPEC=true ===")
conv_pool, ssm_pool, remap, _ = _build_dense_pool(p, device)
core, z = build_outputs()

spec_qsl = p["spec_query_start_loc"].to(device).contiguous()
spec_idx_dense = _remap_index_tensor(p["spec_state_indices_tensor"], remap).to(torch.int32).contiguous()
num_acc = p["num_accepted_tokens"].to(device).to(torch.int32).contiguous()
num_spec_decodes = int(p["num_spec_decodes"])

torch.ops._xpu_C.gdn_attention(
    core[:n_actual], z[:n_actual],
    qkvz[:n_actual].contiguous(), ba[:n_actual].contiguous(),
    cfg["num_k_heads"], cfg["num_v_heads"], cfg["head_k_dim"], cfg["head_v_dim"],
    num_prefills=0, num_decodes=num_spec_decodes,
    has_initial_state=None,
    non_spec_query_start_loc=spec_qsl[: num_spec_decodes + 1].contiguous(),
    non_spec_state_indices_tensor=spec_idx_dense[:, 0].contiguous(),
    num_actual_tokens=n_actual,
    spec_state_indices_tensor=spec_idx_dense,
    num_accepted_tokens=num_acc,
    **common_kwargs(conv_pool, ssm_pool),
)
torch.xpu.synchronize()

sycl_core = core[:n_actual].to("cpu").float()
sycl_z = z[:n_actual].to("cpu").float()
fla_core = p["core_attn_out"][:n_actual].float()
fla_z = p["z"][:n_actual].float()
_diff_summary("core_attn_out (IS_SPEC=true)", sycl_core, fla_core)
_diff_summary("z (IS_SPEC=true)", sycl_z, fla_z)

# === Run 2: IS_SPEC=false (bypass spec args, treat seq as one decode) ===
print("\n=== Run 2: IS_SPEC=false (same inputs, no spec args) ===")
conv_pool2, ssm_pool2, remap2, _ = _build_dense_pool(p, device)
core2, z2 = build_outputs()

# query_start_loc = [0, 4]; non_spec_state_indices = the load slot only
# (= spec_state_indices[batch, num_accepted-1] = slot 4 in our capture)
non_spec_qsl = spec_qsl[: num_spec_decodes + 1].contiguous()  # [0, 4]
load_slot = int(p["spec_state_indices_tensor"][0, int(p["num_accepted_tokens"][0]) - 1].item())
load_slot_dense = remap2[load_slot].item()
non_spec_idx = torch.tensor([load_slot_dense], dtype=torch.int32, device=device)

torch.ops._xpu_C.gdn_attention(
    core2[:n_actual], z2[:n_actual],
    qkvz[:n_actual].contiguous(), ba[:n_actual].contiguous(),
    cfg["num_k_heads"], cfg["num_v_heads"], cfg["head_k_dim"], cfg["head_v_dim"],
    num_prefills=0, num_decodes=1,
    has_initial_state=None,
    non_spec_query_start_loc=non_spec_qsl,
    non_spec_state_indices_tensor=non_spec_idx,
    num_actual_tokens=n_actual,
    # NO spec args -> is_spec=false
    **common_kwargs(conv_pool2, ssm_pool2),
)
torch.xpu.synchronize()

sycl_core2 = core2[:n_actual].to("cpu").float()
sycl_z2 = z2[:n_actual].to("cpu").float()
_diff_summary("core_attn_out (IS_SPEC=false)", sycl_core2, fla_core)
_diff_summary("z (IS_SPEC=false)", sycl_z2, fla_z)

# === Compare runs 1 and 2 ===
print("\n=== Run 1 vs Run 2 (does IS_SPEC=true differ from IS_SPEC=false?) ===")
delta12 = (sycl_core - sycl_core2).abs()
print(f"  core_attn_out IS_SPEC=true vs false: max diff={delta12.max().item():.3e}  "
      f"mean={delta12.mean().item():.3e}")
delta12_z = (sycl_z - sycl_z2).abs()
print(f"  z IS_SPEC=true vs false: max diff={delta12_z.max().item():.3e}  "
      f"mean={delta12_z.mean().item():.3e}")

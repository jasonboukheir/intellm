"""Sentinel test: did the SYCL kernel actually run? Pre-fill output buffers
with -999.0 before _call_sycl. After the call, count cells that retained
the sentinel — those are positions the kernel never wrote."""
import sys
import torch
import vllm._xpu_ops  # noqa: F401
import vllm_xpu_kernels._xpu_C  # noqa: F401

sys.path.insert(0, "/workspace/vllm")
from tests.kernels.xpu.test_spec_gdn_replay import _build_dense_pool

PATH = "/tmp/spec_gdn_captures/tuple_language_model_model_layers_0_linear_attn_000004_spec_K4_min4_max4.pt"
SENTINEL = -1024.0  # exactly representable in bf16

p = torch.load(PATH, map_location="cpu", weights_only=False)
p["__path__"] = "manual"
device = torch.device("xpu")
conv_pool, ssm_pool, remap, _ = _build_dense_pool(p, device)

# Replicate _call_sycl but with sentinel-filled outputs
n_actual = int(p["num_actual_tokens"])
core_attn_out = torch.full(
    p["core_attn_out"].shape, SENTINEL,
    dtype=p["core_attn_out"].dtype, device=device,
)
z = torch.full(
    p["z"].shape, SENTINEL,
    dtype=p["z"].dtype, device=device,
)

qkvz = p["projected_states_qkvz"].to(device)
ba = p["projected_states_ba"].to(device)
conv_w = p["conv_weight"].to(device)
if conv_w.dim() == 3:
    conv_w = conv_w.view(conv_w.size(0), conv_w.size(2))
conv_b = p["conv_bias"].to(device) if p["conv_bias"] is not None else None

cfg = p["layer_config"]

# Sentinel-fill the SSM pool too: where does the kernel write?
ssm_pool_sentinel = torch.full_like(ssm_pool, SENTINEL)
# but copy back the captured pre-state at the load slot so the kernel's
# initial-state read is meaningful. Otherwise it'd start from -999s.
ssm_pool_sentinel[:] = ssm_pool[:]
# Mark slots 1..4 (post-state targets) as sentinels
spec_idx_unique = torch.unique(p["spec_state_indices_tensor"])
spec_idx_unique = spec_idx_unique[spec_idx_unique > 0]
for slot in spec_idx_unique.tolist():
    dense = remap[slot].item()
    if dense > 0:
        ssm_pool_sentinel[dense] = SENTINEL

print(f"Pre-call core_attn_out value (should be -999): {core_attn_out[0,0,0].item()}")
print(f"Pre-call z value: {z[0,0,0].item()}")
print(f"Pre-call ssm_pool_sentinel at slot dense {remap[4].item()}: "
      f"{ssm_pool_sentinel[remap[4].item(),0,0,0].item()}")

spec_qsl = p["spec_query_start_loc"].to(device).contiguous()

def _remap(idx):
    flat = idx.flatten().to(remap.device)
    out = remap[flat].view_as(idx)
    return out

spec_idx_dense = _remap(p["spec_state_indices_tensor"]).to(torch.int32).contiguous()
num_acc = p["num_accepted_tokens"].to(device).to(torch.int32).contiguous()
num_spec_decodes = int(p["num_spec_decodes"])
non_spec_idx_sentinel = spec_idx_dense[:, 0].contiguous()

torch.ops._xpu_C.gdn_attention(
    core_attn_out[:n_actual],
    z[:n_actual],
    qkvz[:n_actual].contiguous(),
    ba[:n_actual].contiguous(),
    cfg["num_k_heads"],
    cfg["num_v_heads"],
    cfg["head_k_dim"],
    cfg["head_v_dim"],
    conv_state=conv_pool,
    ssm_state=ssm_pool_sentinel,
    conv_weights=conv_w,
    conv_bias=conv_b,
    activation=cfg["activation"],
    A_log=p["A_log"].to(device),
    dt_bias=p["dt_bias"].to(device),
    tp_size=cfg["tp_size"],
    reorder_input=not cfg["gqa_interleaved_layout"],
    num_prefills=0,
    num_decodes=num_spec_decodes,
    has_initial_state=None,
    non_spec_query_start_loc=spec_qsl[: num_spec_decodes + 1].contiguous(),
    non_spec_state_indices_tensor=non_spec_idx_sentinel,
    num_actual_tokens=n_actual,
    spec_state_indices_tensor=spec_idx_dense,
    num_accepted_tokens=num_acc,
)

torch.xpu.synchronize()

print("\n=== Sentinel results ===")
core_cpu = core_attn_out[:n_actual].to("cpu").float()
z_cpu = z[:n_actual].to("cpu").float()

cs = (core_cpu == SENTINEL).sum().item()
zs = (z_cpu == SENTINEL).sum().item()
print(f"core_attn_out cells still == -999: {cs}/{core_cpu.numel()} "
      f"({100*cs/core_cpu.numel():.2f}%)")
print(f"z cells still == -999: {zs}/{z_cpu.numel()} "
      f"({100*zs/z_cpu.numel():.2f}%)")

# Per-token sentinel survival
print("\nper-token sentinel-survival rate in core_attn_out:")
for t in range(n_actual):
    n = (core_cpu[t] == SENTINEL).sum().item()
    print(f"  token[{t}]: {n}/{core_cpu[t].numel()} = {100*n/core_cpu[t].numel():.2f}%")

# Where in (head, v) does the sentinel survive?
sent_mask = core_cpu == SENTINEL
if sent_mask.any():
    per_hv = sent_mask.float().sum(dim=0)  # (H, V)
    hot = (per_hv > 0).nonzero()
    head_set = sorted(set(int(x[0]) for x in hot))
    v_set = sorted(set(int(x[1]) for x in hot))
    print(f"\n(head, v) coords where sentinel survives:")
    print(f"  heads: {head_set[:30]}")
    print(f"  v_dims: {v_set[:30]}")

# SSM pool: where did the kernel NOT write?
sycl_ssm_full = ssm_pool_sentinel.to("cpu").float()
slots_captured = p["slot_indices"].to(torch.long)
slots_dense = remap[slots_captured.to(remap.device)].to("cpu")
keep = slots_dense > 0
sycl_ssm = sycl_ssm_full[slots_dense[keep]]
sent_ssm = (sycl_ssm == SENTINEL).sum().item()
print(f"\nssm_state at captured slots: {sent_ssm}/{sycl_ssm.numel()} "
      f"cells still == -999 ({100*sent_ssm/sycl_ssm.numel():.2f}%)")
for s in range(sycl_ssm.shape[0]):
    n = (sycl_ssm[s] == SENTINEL).sum().item()
    pct = 100 * n / sycl_ssm[s].numel()
    cap_slot = int(slots_captured[keep.to('cpu')][s])
    print(f"  captured_slot={cap_slot} dense={int(slots_dense[keep][s])}: "
          f"{n}/{sycl_ssm[s].numel()} ({pct:.2f}%) still sentinel")

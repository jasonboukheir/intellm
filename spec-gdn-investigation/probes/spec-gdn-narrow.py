"""One-shot narrowing script: load the smallest non_spec capture, run SYCL,
diff against FLA oracle, dump where the divergence lives."""
import sys
import torch
import vllm._xpu_ops  # noqa: F401  — triggers _xpu_C op registration
import vllm_xpu_kernels._xpu_C  # noqa: F401  — registers gdn_attention

sys.path.insert(0, "/workspace/vllm")
from tests.kernels.xpu.test_spec_gdn_replay import _build_dense_pool, _call_sycl

p = torch.load(
    "/tmp/spec_gdn_captures/tuple_language_model_model_layers_0_linear_attn_000000_non_spec.pt",
    map_location="cpu",
    weights_only=False,
)
p["__path__"] = "manual"
device = torch.device("xpu")
conv_pool, ssm_pool, remap, _ = _build_dense_pool(p, device)
core_attn_out, z, conv_post, ssm_post = _call_sycl(p, conv_pool, ssm_pool, remap, device)

slots = p["slot_indices"].to(torch.long)
slots_dense = remap[slots.to(remap.device)]
keep = slots_dense > 0
sycl = ssm_post[slots_dense[keep]].to("cpu").float()
fla = p["ssm_state_post"][keep.to("cpu")].float()

delta = (sycl - fla).abs()
print(f"shapes: sycl={tuple(sycl.shape)} fla={tuple(fla.shape)}")
print(f"max abs diff: {delta.max().item():.4e}, mean: {delta.mean().item():.4e}")
flat_argmax = int(delta.flatten().argmax().item())
multi = [int(x) for x in torch.unravel_index(delta.flatten().argmax(), delta.shape)]
print(f"max location (multi-idx): {multi}")
print(f"  sycl value: {sycl[tuple(multi)].item():.6e}")
print(f"  fla  value: {fla[tuple(multi)].item():.6e}")

tol = 2e-2 + 2e-2 * fla.abs()
fail_mask = delta > tol
n_fail = int(fail_mask.sum().item())
print(f"cells exceeding atol+rtol*|fla|: {n_fail} / {fail_mask.numel()} ({100*n_fail/fail_mask.numel():.4f}%)")
fail_idx = fail_mask.nonzero()
for row in fail_idx[:20]:
    idx = tuple(int(x) for x in row.tolist())
    s = sycl[idx].item()
    f = fla[idx].item()
    print(f"  fail @ {idx}: sycl={s:+.4e} fla={f:+.4e} delta={abs(s-f):.4e} ratio={abs(s-f)/(abs(f)+1e-12):.4f}")
print(f"NaN in sycl: {int(torch.isnan(sycl).sum().item())}, NaN in fla: {int(torch.isnan(fla).sum().item())}")
print(f"Inf in sycl: {int(torch.isinf(sycl).sum().item())}, Inf in fla: {int(torch.isinf(fla).sum().item())}")

# Per-axis fail concentration: dim names match (slot, head, k, v)
for axis_name, axis in [("slot", 0), ("head", 1), ("k", 2), ("v", 3)]:
    other = tuple(i for i in range(fail_mask.ndim) if i != axis)
    per_axis = fail_mask.float().mean(dim=other)
    nz = (per_axis > 0).nonzero().flatten().tolist()
    print(
        f"  {axis_name}: max-rate={per_axis.max().item():.3f} "
        f"nonzero indices ({len(nz)}/{per_axis.numel()}): "
        f"{nz[:20]}{'...' if len(nz) > 20 else ''}"
    )

n_actual = int(p["num_actual_tokens"])
n_pref = int(p["num_prefills"])
n_dec = int(p["num_decodes"])
slots_list = p["slot_indices"].tolist()
print(
    f"capture: num_actual={n_actual}, num_prefills={n_pref}, "
    f"num_decodes={n_dec}, slots={slots_list}"
)

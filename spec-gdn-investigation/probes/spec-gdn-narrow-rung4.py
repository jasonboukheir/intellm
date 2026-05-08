"""Rung-4 narrowing: spec_K4_min4_max4 (k=3 fully accepted) shows
core_attn_out diverges massively from FLA. Dump per-token / per-head
diff structure to localize the bug."""
import sys
import torch
import vllm._xpu_ops  # noqa: F401
import vllm_xpu_kernels._xpu_C  # noqa: F401

sys.path.insert(0, "/workspace/vllm")
from tests.kernels.xpu.test_spec_gdn_replay import _build_dense_pool, _call_sycl

PATH = "/tmp/spec_gdn_captures/tuple_language_model_model_layers_0_linear_attn_000004_spec_K4_min4_max4.pt"

p = torch.load(PATH, map_location="cpu", weights_only=False)
p["__path__"] = "manual"
device = torch.device("xpu")
conv_pool, ssm_pool, remap, _ = _build_dense_pool(p, device)
core, z, conv_post, ssm_post = _call_sycl(p, conv_pool, ssm_pool, remap, device)

# ssm_state_post diff: walk every captured slot, compare SYCL pool vs FLA capture
slots_captured = p["slot_indices"].to(torch.long)
slots_dense = remap[slots_captured.to(remap.device)]
keep = slots_dense > 0
sycl_ssm = ssm_post[slots_dense[keep]].to("cpu").float()  # shape (n_slots, H, V, K)
fla_ssm = p["ssm_state_post"][keep.to("cpu")].float()
ssm_delta = (sycl_ssm - fla_ssm).abs()
ssm_tol = 2e-2 + 2e-2 * fla_ssm.abs()
ssm_fail = ssm_delta > ssm_tol
print("\n=== ssm_state_post diff (per slot, post-axis-fix layout = (slot, head, v, k)) ===")
print(f"shape: {tuple(sycl_ssm.shape)}  slots: {slots_captured[keep.to('cpu')].tolist()}")
print(f"max abs diff: {ssm_delta.max().item():.4e}, mean: {ssm_delta.mean().item():.4e}")
print(f"cells failing: {int(ssm_fail.sum())}/{ssm_fail.numel()} "
      f"({100*ssm_fail.float().mean():.4f}%)")
# per-slot fails
print("per-slot fails:")
for s in range(sycl_ssm.shape[0]):
    n = int(ssm_fail[s].sum())
    if n:
        # per-(head, v) hot list
        per_hv = ssm_fail[s].float().sum(dim=2)  # (H, V)
        hot = (per_hv > 0).nonzero()
        head_set = sorted(set(int(x[0]) for x in hot))
        v_set = sorted(set(int(x[1]) for x in hot))
        print(f"  captured_slot={int(slots_captured[keep.to('cpu')][s])} "
              f"dense_slot={int(slots_dense[keep][s])}: {n} fails  "
              f"heads={head_set[:15]}  v_dims={v_set[:20]}")
        # show worst 5
        for idx in (ssm_delta[s] * ssm_fail[s].float()).flatten().topk(5).indices:
            triple = tuple(int(x) for x in torch.unravel_index(idx, ssm_delta[s].shape))
            sv, fv = sycl_ssm[s][triple].item(), fla_ssm[s][triple].item()
            print(f"    (head,v,k)={triple}: sycl={sv:+.4e} fla={fv:+.4e} "
                  f"delta={abs(sv-fv):.4e}")

n_actual = int(p["num_actual_tokens"])
sycl = core[:n_actual].to("cpu").float()
fla = p["core_attn_out"][:n_actual].float()
delta = (sycl - fla).abs()

print(f"flavor={p['flavor']} num_actual={n_actual}")
print(f"num_spec_decodes={int(p['num_spec_decodes'])}")
print(f"num_accepted_tokens={p['num_accepted_tokens'].tolist()}")
print(f"spec_state_indices_tensor shape: {tuple(p['spec_state_indices_tensor'].shape)}")
print(f"  values: {p['spec_state_indices_tensor'].tolist()}")
print(f"spec_query_start_loc: {p['spec_query_start_loc'].tolist()}")
print(f"shapes: sycl={tuple(sycl.shape)} fla={tuple(fla.shape)} dtype={sycl.dtype}")
print(f"max abs diff: {delta.max().item():.4e}, mean: {delta.mean().item():.4e}")
print(f"NaN sycl/fla: {int(torch.isnan(sycl).sum())}/{int(torch.isnan(fla).sum())}")

# Per-token (axis=0): which of the 4 tokens is broken?
tok_max = delta.amax(dim=(1, 2))
tok_mean = delta.mean(dim=(1, 2))
print("\nper-token max/mean diff:")
for t in range(n_actual):
    print(f"  token[{t}]: max={tok_max[t].item():.4e} mean={tok_mean[t].item():.4e} "
          f"sycl_norm={sycl[t].norm().item():.3e} fla_norm={fla[t].norm().item():.3e}")

# Tolerance mask
tol = 2e-2 + 2e-2 * fla.abs()
fail = delta > tol
print(f"\ncells exceeding atol+rtol*|fla|: {int(fail.sum())}/{fail.numel()} ({100*fail.float().mean():.2f}%)")

# Per-token failure rate
print("per-token fail-rate:")
for t in range(n_actual):
    print(f"  token[{t}]: {int(fail[t].sum())}/{fail[t].numel()} = {100*fail[t].float().mean():.2f}%")

# Per-head failure rate (any token)
per_head = fail.float().mean(dim=(0, 2))
print(f"\nper-head fail-rate: max={per_head.max().item():.3f} "
      f"nonzero heads: {(per_head > 0).sum().item()}/{per_head.numel()}")

# z output: also part of gated_delta_rule — check it
sycl_z = z[:n_actual].to("cpu").float()
fla_z = p["z"][:n_actual].float()
dz = (sycl_z - fla_z).abs()
print(f"\nz: max diff={dz.max().item():.4e} mean={dz.mean().item():.4e}")

# Dump the worst failing cells with sycl/fla side-by-side
fail_idx = fail.nonzero()
ranked = sorted(
    [tuple(int(x) for x in idx.tolist()) for idx in fail_idx],
    key=lambda c: -delta[c].item(),
)
print(f"\nworst 20 failing (token, head, v_dim) — sycl vs fla:")
for c in ranked[:20]:
    s = sycl[c].item(); f = fla[c].item()
    print(f"  {c}: sycl={s:+.4e} fla={f:+.4e} delta={abs(s-f):.4e}")

# Are SYCL values at fail cells zero (or systematically small)?
fail_sycl = sycl[fail]
fail_fla = fla[fail]
print(f"\nat {int(fail.sum())} failing cells:")
print(f"  |sycl|: max={fail_sycl.abs().max():.4e} mean={fail_sycl.abs().mean():.4e} "
      f"<1e-6 count: {int((fail_sycl.abs() < 1e-6).sum())}")
print(f"  |fla|:  max={fail_fla.abs().max():.4e} mean={fail_fla.abs().mean():.4e}")

# Per (head, v_dim) — group fails to check tile/lane structure
fail_per_hv = fail.float().sum(dim=0)  # shape (32, 128)
hot_hv = (fail_per_hv > 0).nonzero()
print(f"\n(head, v_dim) coords with any fail: {len(hot_hv)}")
print(f"v_dim distribution of fails: {sorted(set(int(x[1]) for x in hot_hv))[:30]}")
print(f"head distribution of fails: {sorted(set(int(x[0]) for x in hot_hv))[:30]}")

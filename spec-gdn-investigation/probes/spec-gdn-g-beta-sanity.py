"""Sanity check: with A_log = -100 and b = -100, what does the kernel
ACTUALLY compute for g and beta?

The kernel does:
  A_log_local = -exp(A_log)        # line 108, fp32
  g = exp(A_log_local * softplus(a + dt_bias))   # line 171
  beta = 1 / (1 + exp(-b))         # line 169

For A_log = -100 (bf16 exact):
  exp(-100) ≈ 3.7e-44 — denormal in fp32 (min normal 1.18e-38)
  -exp(-100) ≈ -3.7e-44 (denormal or 0 with FTZ)
  softplus(a + dt_bias) for a, dt_bias ~ realistic values

For b = -100: beta = sigmoid(-100) = 1/(1+exp(100)) — exp(100) ≈ 2.7e43 (fp32 overflow → +inf), 1/inf → 0.

But what if:
- FTZ is OFF on XPU → A_log_local = -3.7e-44 (denormal, not zero)
- Then product with softplus is also tiny
- exp() of tiny negative ≈ 1 - tiny, but tiny is below fp32 precision below 1
- So g = 1.0
"""
import torch

# Reproduce the kernel computation in fp32 on XPU
device = torch.device("xpu")

A_log = torch.tensor(-100.0, dtype=torch.bfloat16, device=device).float()
a = torch.tensor(0.5, dtype=torch.bfloat16, device=device).float()  # typical
dt_bias = torch.tensor(-2.0, dtype=torch.bfloat16, device=device).float()

A_log_local = -torch.exp(A_log)
print(f"A_log_local = -exp(-100) = {A_log_local.item():.4e}")

a_plus = a + dt_bias
softplus_a = torch.where(a_plus < 20.0, torch.log1p(torch.exp(a_plus)), a_plus)
print(f"softplus(a + dt_bias) for a=0.5, dt_bias=-2.0 = {softplus_a.item():.4e}")

product = A_log_local * softplus_a
print(f"A_log_local * softplus = {product.item():.4e}")

g = torch.exp(product)
print(f"g = exp(product) = {g.item():.10f}")
print(f"g is exactly 1.0? {(g == 1.0).item()}")
print(f"g - 1.0 = {(g - 1.0).item():.4e}")

b = torch.tensor(-100.0, dtype=torch.bfloat16, device=device).float()
beta = torch.sigmoid(b)
print(f"beta = sigmoid(-100) = {beta.item():.4e}")
print(f"beta is exactly 0.0? {(beta == 0.0).item()}")

# Test with much larger softplus value (e.g., a=20, edge case)
a_large = torch.tensor(20.0, dtype=torch.bfloat16, device=device).float()
sp_large = torch.where(a_large < 20.0, torch.log1p(torch.exp(a_large)), a_large)
prod_large = A_log_local * sp_large
g_large = torch.exp(prod_large)
print(f"\nedge case a=20 (max softplus path): "
      f"sp={sp_large.item():.4f} prod={prod_large.item():.4e} "
      f"g={g_large.item():.10f}")

# Test with very small a (typical inference value)
a_tiny = torch.tensor(0.01, dtype=torch.bfloat16, device=device).float()
sp_tiny = torch.where(a_tiny < 20.0, torch.log1p(torch.exp(a_tiny)), a_tiny)
prod_tiny = A_log_local * sp_tiny
g_tiny = torch.exp(prod_tiny)
print(f"a=0.01: sp={sp_tiny.item():.4f} prod={prod_tiny.item():.4e} "
      f"g={g_tiny.item():.10f}")

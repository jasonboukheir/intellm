"""Correctness tests for xpu-flash-attention Python bindings."""

import pytest
import numpy as np


def reference_attention(Q, K, V, causal=True):
    """Naive attention in numpy for correctness checking."""
    scale = 1.0 / np.sqrt(Q.shape[-1])
    S = Q @ K.transpose(0, 1, 3, 2) * scale

    if causal:
        seq_len = S.shape[-1]
        mask = np.triu(np.ones((seq_len, seq_len), dtype=np.float32), k=1)
        S = S - mask * 1e9

    S_max = S.max(axis=-1, keepdims=True)
    S_exp = np.exp(S - S_max)
    S_softmax = S_exp / S_exp.sum(axis=-1, keepdims=True)

    return S_softmax @ V


@pytest.mark.parametrize("seq_len", [64, 128, 256, 512])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [True, False])
def test_flash_attention_correctness(seq_len, head_dim, causal):
    """Compare xpu flash attention against numpy reference."""
    np.random.seed(42)
    batch, heads = 1, 4
    Q = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float16)
    K = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float16)
    V = np.random.randn(batch, heads, seq_len, head_dim).astype(np.float16)

    ref = reference_attention(Q.astype(np.float32), K.astype(np.float32), V.astype(np.float32), causal)

    # TODO: Call our kernel once bindings are built
    # import xpu_flash_attn
    # out = xpu_flash_attn.forward(Q, K, V, causal=causal)
    # np.testing.assert_allclose(out, ref, atol=1e-2, rtol=1e-2)

    pytest.skip("XPU kernel bindings not yet built")


def test_reference_sanity():
    """Verify reference implementation produces reasonable output."""
    np.random.seed(0)
    Q = np.random.randn(1, 2, 32, 64).astype(np.float32)
    K = np.random.randn(1, 2, 32, 64).astype(np.float32)
    V = np.random.randn(1, 2, 32, 64).astype(np.float32)

    out = reference_attention(Q, K, V, causal=True)

    assert out.shape == Q.shape
    assert not np.any(np.isnan(out))
    assert np.abs(out).max() < 100

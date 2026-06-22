import math

import torch

from src.attention import (
    MultiHeadSelfAttention,
    causal_mask,
    scaled_dot_product_attention,
)


torch.manual_seed(0)


def test_attention_output_shape():
    batch, seq, d_k, d_v = 2, 5, 8, 16
    q = torch.randn(batch, seq, d_k)
    k = torch.randn(batch, seq, d_k)
    v = torch.randn(batch, seq, d_v)
    out, weights = scaled_dot_product_attention(q, k, v)
    assert out.shape == (batch, seq, d_v)
    assert weights.shape == (batch, seq, seq)


def test_attention_weights_sum_to_one():
    batch, seq, d_k = 3, 7, 12
    q = torch.randn(batch, seq, d_k)
    k = torch.randn(batch, seq, d_k)
    v = torch.randn(batch, seq, d_k)
    _, weights = scaled_dot_product_attention(q, k, v)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    # Weights are probabilities, so they must be non negative.
    assert torch.all(weights >= 0)


def test_attention_scaling_factor():
    # With identical keys and zero queries, scores are all equal so the
    # softmax is uniform regardless of the scale. Check the uniform case.
    batch, seq, d_k = 1, 4, 8
    q = torch.zeros(batch, seq, d_k)
    k = torch.randn(batch, seq, d_k)
    v = torch.randn(batch, seq, d_k)
    _, weights = scaled_dot_product_attention(q, k, v)
    expected = torch.full((batch, seq, seq), 1.0 / seq)
    assert torch.allclose(weights, expected, atol=1e-6)


def test_causal_mask_zeros_future_positions():
    seq = 6
    mask = causal_mask(seq)
    q = torch.randn(1, seq, 8)
    k = torch.randn(1, seq, 8)
    v = torch.randn(1, seq, 8)
    # Mask broadcasts over the leading batch dimension.
    _, weights = scaled_dot_product_attention(q, k, v, mask=mask)
    # For every query position i, all key positions j > i must be exactly 0.
    upper = torch.triu(torch.ones(seq, seq), diagonal=1).bool()
    masked_weights = weights[0][upper]
    assert torch.all(masked_weights == 0.0)
    # And the allowed positions still sum to 1 per row.
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_causal_mask_first_row_attends_only_to_itself():
    seq = 5
    mask = causal_mask(seq)
    q = torch.randn(1, seq, 8)
    k = torch.randn(1, seq, 8)
    v = torch.randn(1, seq, 8)
    _, weights = scaled_dot_product_attention(q, k, v, mask=mask)
    # Query 0 can only see key 0, so its weight there must be 1.
    assert torch.allclose(weights[0, 0, 0], torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(weights[0, 0, 1:], torch.zeros(seq - 1), atol=1e-6)


def test_multihead_output_shape():
    batch, seq, d_model, heads = 4, 9, 32, 4
    mha = MultiHeadSelfAttention(d_model, heads)
    x = torch.randn(batch, seq, d_model)
    out = mha(x)
    assert out.shape == (batch, seq, d_model)


def test_multihead_attention_weights_sum_to_one():
    batch, seq, d_model, heads = 2, 6, 16, 2
    mha = MultiHeadSelfAttention(d_model, heads)
    x = torch.randn(batch, seq, d_model)
    out, weights = mha(x, return_attention=True)
    assert weights.shape == (batch, heads, seq, seq)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_multihead_causal_mask_zeros_future():
    batch, seq, d_model, heads = 2, 7, 16, 4
    mha = MultiHeadSelfAttention(d_model, heads)
    x = torch.randn(batch, seq, d_model)
    mask = causal_mask(seq)
    _, weights = mha(x, mask=mask, return_attention=True)
    upper = torch.triu(torch.ones(seq, seq), diagonal=1).bool()
    for b in range(batch):
        for h in range(heads):
            assert torch.all(weights[b, h][upper] == 0.0)


def test_multihead_requires_divisible_dims():
    raised = False
    try:
        MultiHeadSelfAttention(d_model=30, num_heads=4)
    except ValueError:
        raised = True
    assert raised


def test_causal_mask_changes_output():
    # A causal mask should change the result relative to full attention,
    # confirming the mask actually flows through the module.
    batch, seq, d_model, heads = 1, 8, 16, 2
    mha = MultiHeadSelfAttention(d_model, heads)
    x = torch.randn(batch, seq, d_model)
    full = mha(x)
    masked = mha(x, mask=causal_mask(seq))
    assert not torch.allclose(full, masked)

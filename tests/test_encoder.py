import torch

from src.attention import causal_mask
from src.encoder import TransformerEncoder, TransformerEncoderBlock
from src.positional import PositionalEncoding


torch.manual_seed(0)


def test_encoder_block_output_shape_matches_input():
    batch, seq, d_model, heads, d_ff = 2, 7, 32, 4, 64
    block = TransformerEncoderBlock(d_model, heads, d_ff)
    x = torch.randn(batch, seq, d_model)
    out = block(x)
    assert out.shape == x.shape


def test_encoder_block_returns_attention_weights():
    batch, seq, d_model, heads, d_ff = 2, 5, 16, 2, 32
    block = TransformerEncoderBlock(d_model, heads, d_ff)
    x = torch.randn(batch, seq, d_model)
    out, weights = block(x, return_attention=True)
    assert out.shape == x.shape
    assert weights.shape == (batch, heads, seq, seq)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_encoder_stack_output_shape():
    batch, seq, d_model, heads, d_ff = 3, 9, 32, 4, 64
    enc = TransformerEncoder(num_layers=3, d_model=d_model, num_heads=heads, d_ff=d_ff)
    x = torch.randn(batch, seq, d_model)
    out = enc(x)
    assert out.shape == x.shape


def test_encoder_layernorm_zero_mean_unit_var():
    # After the final post layernorm sublayer each token vector should be
    # roughly zero mean and unit variance (the layernorm affine params start
    # at gamma=1, beta=0).
    batch, seq, d_model, heads, d_ff = 2, 6, 32, 4, 64
    block = TransformerEncoderBlock(d_model, heads, d_ff)
    block.eval()
    x = torch.randn(batch, seq, d_model)
    out = block(x)
    mean = out.mean(dim=-1)
    var = out.var(dim=-1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-3)


def test_encoder_causal_mask_flows_through():
    batch, seq, d_model, heads, d_ff = 1, 8, 16, 2, 32
    enc = TransformerEncoder(num_layers=2, d_model=d_model, num_heads=heads, d_ff=d_ff)
    enc.eval()
    x = torch.randn(batch, seq, d_model)
    full = enc(x)
    masked = enc(x, mask=causal_mask(seq))
    assert not torch.allclose(full, masked)


def test_encoder_with_positional_encoding_end_to_end():
    batch, seq, d_model, heads, d_ff = 2, 10, 32, 4, 64
    pe = PositionalEncoding(d_model, max_len=50)
    enc = TransformerEncoder(num_layers=2, d_model=d_model, num_heads=heads, d_ff=d_ff)
    tokens = torch.randn(batch, seq, d_model)
    out = enc(pe(tokens))
    assert out.shape == (batch, seq, d_model)
    assert torch.isfinite(out).all()


def test_encoder_gradients_flow():
    # A real training signal must backpropagate to every block.
    batch, seq, d_model, heads, d_ff = 2, 5, 16, 2, 32
    enc = TransformerEncoder(num_layers=2, d_model=d_model, num_heads=heads, d_ff=d_ff)
    x = torch.randn(batch, seq, d_model, requires_grad=True)
    out = enc(x)
    loss = out.pow(2).mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for name, p in enc.named_parameters():
        assert p.grad is not None, f"no grad for {name}"

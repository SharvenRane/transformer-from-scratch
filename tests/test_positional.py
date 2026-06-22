import torch

from src.positional import PositionalEncoding


def test_positional_preserves_shape():
    batch, seq, d_model = 2, 10, 16
    pe = PositionalEncoding(d_model, max_len=50)
    x = torch.zeros(batch, seq, d_model)
    out = pe(x)
    assert out.shape == (batch, seq, d_model)


def test_positional_adds_distinct_position_signal():
    # Feeding zeros, the output is exactly the positional code. Different
    # positions must produce different vectors.
    d_model = 16
    pe = PositionalEncoding(d_model, max_len=50)
    x = torch.zeros(1, 6, d_model)
    out = pe(x)[0]
    for i in range(out.size(0)):
        for j in range(i + 1, out.size(0)):
            assert not torch.allclose(out[i], out[j])


def test_positional_values_bounded():
    # Sinusoidal codes live in [-1, 1].
    d_model = 32
    pe = PositionalEncoding(d_model, max_len=100)
    out = pe(torch.zeros(1, 100, d_model))
    assert torch.all(out <= 1.0 + 1e-6)
    assert torch.all(out >= -1.0 - 1e-6)


def test_positional_no_grad_parameters():
    pe = PositionalEncoding(16, max_len=20)
    assert len(list(pe.parameters())) == 0


def test_positional_rejects_too_long_sequence():
    pe = PositionalEncoding(16, max_len=8)
    raised = False
    try:
        pe(torch.zeros(1, 9, 16))
    except ValueError:
        raised = True
    assert raised

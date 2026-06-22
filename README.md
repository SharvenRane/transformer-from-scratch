# transformer-from-scratch

A compact, readable implementation of the Transformer encoder built directly on
PyTorch tensor operations. The goal here is clarity. Nothing in this repo uses
`torch.nn.Transformer` or the fused `scaled_dot_product_attention` helper, so the
attention math stays fully visible in plain code.

## What is implemented

- **Scaled dot product attention** (`src/attention.py`). The core operation that
  takes queries, keys, and values, scores every query against every key, scales
  by the square root of the key dimension, applies an optional mask, and turns
  the scores into a probability distribution with softmax.
- **Multi head self attention** (`src/attention.py`). Learned projections split
  the model dimension into several heads, each head runs its own attention, and
  the results are concatenated and projected back. A causal mask helper is
  included for autoregressive use.
- **Sinusoidal positional encoding** (`src/positional.py`). The fixed sine and
  cosine code from the original paper, added to the input so the model can tell
  positions apart. It holds no learnable parameters.
- **Encoder block and stack** (`src/encoder.py`). Each block runs self attention
  then a position wise feed forward network. Both sublayers use a residual
  connection followed by layer normalization, matching the post layernorm
  arrangement of the original design. `TransformerEncoder` stacks several blocks.

## Layout

```
src/
  attention.py     scaled dot product attention, multi head attention, causal mask
  positional.py    sinusoidal positional encoding
  encoder.py       feed forward, encoder block, encoder stack
tests/
  test_attention.py
  test_positional.py
  test_encoder.py
README.md
requirements.txt
```

## Quick start

```python
import torch
from src import PositionalEncoding, TransformerEncoder
from src.attention import causal_mask

batch, seq, d_model = 2, 16, 64
x = torch.randn(batch, seq, d_model)

pos = PositionalEncoding(d_model, max_len=512)
encoder = TransformerEncoder(num_layers=4, d_model=d_model, num_heads=8, d_ff=256)

out = encoder(pos(x))                 # full bidirectional attention
out_causal = encoder(pos(x), mask=causal_mask(seq))  # masked future positions
```

## Design notes

The mask convention is boolean and follows the "allowed" reading: a `True` entry
means a query position is permitted to attend to a key position, and a `False`
entry forbids it. Forbidden scores are set to negative infinity before the
softmax so they collapse to zero weight. A row that is fully masked would make
softmax produce undefined values, so those are mapped back to zero, which keeps
the operation well defined in edge cases.

Multi head attention reshapes the projected tensors into
`(batch, num_heads, seq, d_head)`, runs attention per head in a single batched
matmul, then merges the heads back. The 2D causal mask of shape `(seq, seq)` is
broadcast across the batch and head dimensions automatically.

## Tests

The test suite checks behavior rather than surface details. It verifies that

- the encoder output keeps the same shape as its input,
- attention weights are non negative and sum to one along the key axis,
- a causal mask drives every future position to exactly zero weight while the
  remaining weights still sum to one,
- the first causal query attends only to itself,
- positional codes differ across positions and stay within the sine range,
- post layernorm output has near zero mean and near unit variance,
- gradients flow back to every parameter in a stacked encoder.

Run the suite with pytest:

```
python -m pytest tests/ -q
```

On the local CPU run all 22 tests passed in about 1.4 seconds. The tests use
tiny synthetic tensors only, so there are no downloads and nothing to set up
beyond installing the requirements.

## Requirements

PyTorch and pytest. Install with:

```
pip install -r requirements.txt
```

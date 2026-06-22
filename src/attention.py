"""Scaled dot product attention and multi head self attention.

Everything here is written directly on top of torch tensor ops so the math
stays visible. No use of torch.nn.Transformer or
torch.nn.functional.scaled_dot_product_attention.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute scaled dot product attention.

    Args:
        query: tensor of shape (..., seq_q, d_k).
        key: tensor of shape (..., seq_k, d_k).
        value: tensor of shape (..., seq_k, d_v).
        mask: optional boolean tensor broadcastable to
            (..., seq_q, seq_k). Positions that are False (or 0) are
            forbidden and receive zero attention weight.

    Returns:
        A tuple of (output, weights) where output has shape
        (..., seq_q, d_v) and weights has shape (..., seq_q, seq_k).
        The weights sum to 1 along the last (key) dimension.
    """
    d_k = query.size(-1)
    # (..., seq_q, seq_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # Where the mask is False, drive the score to a very large negative
        # value so the softmax assigns it (effectively) zero probability.
        scores = scores.masked_fill(~mask.bool(), float("-inf"))

    weights = torch.softmax(scores, dim=-1)

    # A fully masked row would produce NaNs from softmax over all -inf.
    # Replace any such NaNs with zeros so the op stays well defined.
    weights = torch.nan_to_num(weights, nan=0.0)

    output = torch.matmul(weights, value)
    return output, weights


class MultiHeadSelfAttention(nn.Module):
    """Multi head self attention with learned q, k, v and output projections."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq, d_model) -> (batch, num_heads, seq, d_head)
        batch, seq, _ = x.shape
        x = x.view(batch, seq, self.num_heads, self.d_head)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, num_heads, seq, d_head) -> (batch, seq, d_model)
        batch, num_heads, seq, d_head = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq, num_heads * d_head)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Run self attention over the input sequence.

        Args:
            x: tensor of shape (batch, seq, d_model).
            mask: optional boolean mask broadcastable to
                (batch, num_heads, seq, seq). True means a key position is
                allowed for a given query position.
            return_attention: when True also return the attention weights.

        Returns:
            The attended output of shape (batch, seq, d_model). When
            return_attention is True a tuple of (output, weights) is
            returned where weights has shape (batch, num_heads, seq, seq).
        """
        q = self._split_heads(self.w_q(x))
        k = self._split_heads(self.w_k(x))
        v = self._split_heads(self.w_v(x))

        if mask is not None and mask.dim() == 2:
            # (seq, seq) -> (1, 1, seq, seq) so it broadcasts over batch and heads.
            mask = mask.unsqueeze(0).unsqueeze(0)

        attended, weights = scaled_dot_product_attention(q, k, v, mask=mask)
        attended = self._merge_heads(attended)
        output = self.w_o(attended)
        output = self.dropout(output)

        if return_attention:
            return output, weights
        return output


def causal_mask(seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Return a lower triangular boolean mask of shape (seq_len, seq_len).

    Entry [i, j] is True when query position i may attend to key position j,
    that is when j <= i. Future positions (j > i) are False.
    """
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

"""Transformer encoder block and stack.

The block uses the post layernorm arrangement from the original paper:
each sublayer output is added to its input as a residual and then layer
normalized.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .attention import MultiHeadSelfAttention


class FeedForward(nn.Module):
    """Position wise two layer feed forward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerEncoderBlock(nn.Module):
    """A single Transformer encoder block.

    Layout per block:
        attention sublayer: out = LayerNorm(x + Attention(x))
        feed forward sublayer: out = LayerNorm(out + FeedForward(out))
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        """Apply the encoder block.

        Args:
            x: tensor of shape (batch, seq, d_model).
            mask: optional attention mask, see MultiHeadSelfAttention.
            return_attention: when True also return attention weights.
        """
        attn_out, weights = self.self_attention(x, mask=mask, return_attention=True)
        x = self.norm1(x + self.dropout1(attn_out))

        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_out))

        if return_attention:
            return x, weights
        return x


class TransformerEncoder(nn.Module):
    """A stack of Transformer encoder blocks."""

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Run the input through every encoder block in order."""
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

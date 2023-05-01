"""Base Transformer.

Glossary of shapes:
- B: Batch size.
- T: Sequence length.
- D: Model embedding size = d_model.
- H: Number of attention heads.
"""

import dataclasses
from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


def layer_norm(x: jax.Array) -> jax.Array:
  """Applies a unique LayerNorm to x with default settings."""
  ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
  return ln(x)


@dataclasses.dataclass
class Transformer(hk.Module):
  """A transformer stack."""

  num_heads: int
  num_layers: int
  key_size: int
  dropout_rate: float
  widening_factor: int = 4
  name: Optional[str] = None

  def __call__(
      self,
      embeddings: jax.Array,  # [B, T, D]
      *,
      is_training: bool = True,
  ) -> jax.Array:  # [B, T, D]
    """Transforms input embedding sequences to output embedding sequences."""

    initializer = hk.initializers.VarianceScaling(2 / self.num_layers)
    dropout_rate = self.dropout_rate if is_training else 0.
    _, _, model_size = embeddings.shape

    h = embeddings
    for _ in range(self.num_layers):
      # First the attention block.
      attn_block = hk.MultiHeadAttention(
          num_heads=self.num_heads,
          key_size=self.key_size,
          model_size=model_size,
          w_init=initializer,
      )
      h_norm = layer_norm(h)
      h_attn = attn_block(h_norm, h_norm, h_norm)
      h_attn = hk.dropout(hk.next_rng_key(), dropout_rate, h_attn)
      h = h + h_attn

      # Then the dense block.
      dense_block = hk.Sequential([
          hk.Linear(self.widening_factor * model_size, w_init=initializer),
          jax.nn.gelu,
          hk.Linear(model_size, w_init=initializer),
      ])
      h_norm = layer_norm(h)
      h_dense = dense_block(h_norm)
      h_dense = hk.dropout(hk.next_rng_key(), dropout_rate, h_dense)
      h = h + h_dense

    return layer_norm(h)
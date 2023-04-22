# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Didactic example of an autoregressive Transformer-based language model.

Glossary of shapes:
- B: Batch size.
- T: Sequence length.
- D: Model embedding size.
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
#      mask: jax.Array,  # [B, T]
      *,
      is_training: bool = True,
  ) -> jax.Array:  # [B, T, D]
    """Transforms input embedding sequences to output embedding sequences."""

    initializer = hk.initializers.VarianceScaling(2 / self.num_layers)
    dropout_rate = self.dropout_rate if is_training else 0.
    _, seq_len, model_size = embeddings.shape

#     # Compute causal mask for autoregressive sequence modelling.
#     mask = mask[:, None, None, :]  # [B, H=1, T'=1, T]
#     causal_mask = np.tril(np.ones((1, 1, seq_len, seq_len)))  # [B=1, H=1, T, T]
#     mask = mask * causal_mask  # [B, H=1, T, T]

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


@dataclasses.dataclass
class Classifier(hk.Module):
  """A ViT-style classifier."""

  transformer: Transformer
  model_size: int
  num_classes: int
  name: Optional[str] = None

  def __call__(
      self,
      input_chunks: jax.Array,
      *,
      is_training: bool = True,
  ) -> jax.Array:
    """Forward pass. Returns a sequence of logits."""
    unused_batch_size, seq_len, input_chunk_size = input_chunks.shape

    # Embed the input_chunks and positions.
    embed_init = hk.initializers.TruncatedNormal(stddev=0.2)
    embedding = hk.Linear(self.model_size, w_init=embed_init)
    token_embedding = embedding(input_chunks)  # [B, T, D]

    positional_embeddings = hk.get_parameter(
        'positional_embeddings', [seq_len, self.model_size], init=embed_init)
    input_embeddings = token_embedding + positional_embeddings  # [B, T, D]

#    # Class Token TODO: ablate this
#    class_token = hk.get_parameter(
#        'class_token', [1, 1, self.model_size], init=embed_init)
#    class_token = jnp.tile(class_token, [input_chunks.shape[0], 1, 1])
#    input_embeddings = jnp.concatenate([class_token, input_embeddings], axis=1)


    # Run the transformer over the inputs.
    embeddings = self.transformer(
        input_embeddings,
#        input_mask,
        is_training=is_training,
    )  # [B, T, D]

    # Decode the embeddings (here, we use untied weights).
    return hk.Linear(self.num_classes)(embeddings)  # [B, T, V]

import dataclasses
from typing import Optional
import haiku as hk
import jax
import jax.numpy as jnp
from meta_transformer.transformer import Transformer
from meta_transformer import utils


def chunk_weights(weights: jnp.ndarray, chunk_size: int) -> jnp.ndarray:
    flat_weights = weights.flatten()
    flat_weights = utils.pad_to_chunk_size(flat_weights, chunk_size)
    weight_chunks = jnp.split(flat_weights, len(flat_weights) // chunk_size)
    return jnp.array(weight_chunks)


@dataclasses.dataclass
class WeightEmbedding(hk.Module):
    """A module that embeds an array of neural network weights."""
    chunk_size: int
    embed_dim: int

    def __call__(
        self,
        weights: jax.Array,
    ) -> jax.Array:  # [B, T, D]
        embed = hk.Linear(self.embed_dim)
        weight_chunks = jax.vmap(chunk_weights, (0, None))(weights, self.chunk_size)  # [B, T, D]
        embeddings = embed(weight_chunks)
        return embeddings


@dataclasses.dataclass
class NetEmbedding(hk.Module):
    """A module that creates embedding vectors from neural network params."""
    embed_dim: int

    def __call__(
            self,
            params: dict,
    ) -> jax.Array:
        conv_embed = WeightEmbedding(chunk_size=256, embed_dim=self.embed_dim)
        linear_embed = WeightEmbedding(chunk_size=1024, embed_dim=self.embed_dim)
        bias_embed = WeightEmbedding(chunk_size=16, embed_dim=self.embed_dim)

        params_dict = {f"{k}/{subk}": subv for k, v in params.items() 
                    for subk, subv in v.items()}

        embeddings = []
        for k, v in params_dict.items():
            if k.endswith('b'):
                embeddings.append(bias_embed(v))
            else:
                embeddings.append(conv_embed(v) if 'conv' in k else linear_embed(v))
        return jnp.concatenate(embeddings, axis=1)


@dataclasses.dataclass
class MetaModelClassifier(hk.Module):
  """A simple meta-model."""

  transformer: Transformer
  model_size: int
  num_classes: int
  name: Optional[str] = None
  chunk_size: Optional[int] = 4

  def __call__(
      self,
      params: dict,
      *,
      is_training: bool = True,
  ) -> jax.Array:
    """Forward pass. Returns a sequence of logits."""
    net_embed = NetEmbedding(embed_dim=self.model_size)
    embeddings = net_embed(params)  # [B, T, D]
    _, seq_len, _ = embeddings.shape

    # Add positional embeddings.
    positional_embeddings = hk.get_parameter(
        'positional_embeddings', [seq_len, self.model_size], init=jnp.zeros)
    input_embeddings = embeddings + positional_embeddings  # [B, T, D]

    # Run the transformer over the inputs.
    embeddings = self.transformer(
        input_embeddings,
        is_training=is_training,
    )  # [B, T, D]

    return hk.Linear(self.num_classes)(embeddings)  # [B, T, V]
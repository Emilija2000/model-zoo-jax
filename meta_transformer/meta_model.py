import dataclasses
from typing import Optional
import haiku as hk
import jax
import jax.numpy as jnp
from meta_transformer.transformer import Transformer
from meta_transformer import utils
from jax.typing import ArrayLike
import functools
import chex


def chunk_array(x: ArrayLike, chunk_size: int) -> jax.Array:
    """Split an array into chunks of size chunk_size. 
    If not divisible by chunk_size, pad with zeros."""
    x = x.flatten()
    x = utils.pad_to_chunk_size(x, chunk_size)
    return x.reshape(-1, chunk_size)


@dataclasses.dataclass
class ChunkCNN:
    linear_chunk_size: int
    conv_chunk_size: int

    def __call__(self, params: dict) -> dict:
        """Split a CNN into nice little weight chunks."""
        # First, flatten all the layers into a single vector per layer
        params = {k: jnp.concatenate([v.flatten() for v in layer.values()])
                    for k, layer in params.items()}

        # Then, split the layers into chunks
        def chunk_size(layer_name):
            return self.conv_chunk_size if 'conv' in layer_name else \
                   self.linear_chunk_size
        params = {k: chunk_array(v, chunk_size(k)) for k, v in params.items()}
        params = {f"{k}_chunk_{i}": v for k, vs in params.items() 
                    for i, v in enumerate(vs)}
        return params


# TODO: implement unchunking
# def unchunk_layers(chunked_params: dict):
#     """Unchunk a dictionary of chunked parameters."""
#     unchunked_params = {}
#     for k, v in chunked_params.items():
#         layer, _ = k.split("_chunk_")
#         if layer not in unchunked_params:
#             unchunked_params[layer] = [v]
#         else:
#             unchunked_params[layer].append(v)
#     unchunked_params = {k: jnp.concatenate(v) for k, v in unchunked_params.items()}
#     return unchunked_params
# 
# 
# @dataclasses.dataclass
# class UnChunkCNN:
#     """Inverse of ChunkCNN."""
#     linear_chunk_size: int
#     conv_chunk_size: int
#     param_shapes: dict
# 
#     def __call__(self, chunked_params: dict) -> dict:
#         """Map chunked CNN weights back to original shape."""
#         params = unchunk_layers(chunked_params)
#         params = {k: v[:np.prod(self.param_shapes[k])] for k, v in params.items()}
#         params = {k: v.reshape(self.param_shapes[k]) for k, v in params.items()}
#         return params





@dataclasses.dataclass
class NetEmbedding(hk.Module):
    """A module that creates embedding vectors from neural network params."""
    embed_dim: int
    linear_chunk_size: int = 1024
    conv_chunk_size: int = 256


    def __call__(
            self,
            input_params: dict,
    ) -> jax.Array:
        chunk = ChunkCNN(self.linear_chunk_size, self.conv_chunk_size)
        conv_embed = hk.Linear(self.embed_dim)
        linear_embed = hk.Linear(self.embed_dim)
        chunked_params = jax.vmap(chunk)(input_params)  # dict
        embeddings = [
            conv_embed(chunk) if 'conv' in k else 
            linear_embed(chunk) for k, chunk in chunked_params.items()
            ]
        embeddings = jnp.stack(embeddings, axis=1)  # [B, T, D]
        chex.assert_shape(embeddings, [None, None, self.embed_dim])
        return embeddings


# TODO: implement unembedding
# class NetUnEmbeding(hk.Module):
#     """A module that maps embedding vectors back to neural network params."""
#     param_shapes: dict  # shape of each weight in a 'flat' representation
# 
#     def __call__(
#             self,
#             embeddings: ArrayLike,
#     ) -> dict:
#         conv_unembed = hk.Linear(self.conv_chunk_size)
#         linear_unembed = hk.Linear(self.linear_chunk_size)
# 
#         embeddings_dict = {k: v for k, v in zip(self.param_shapes.keys(), embeddings)}
#         chunked_params = {k: conv_unembed(v) if 'conv' in k else
#                              linear_unembed(v) for k, v in embeddings_dict.items()}
#         params = {k}





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

    first_out = embeddings[:, 0, :]  # [B, V]
    return hk.Linear(self.num_classes, name="linear_output")(first_out)  # [B, V]

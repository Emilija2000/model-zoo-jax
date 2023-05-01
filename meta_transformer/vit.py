import dataclasses
from typing import Optional
import haiku as hk
import jax
import jax.numpy as jnp
from meta_transformer.transformer import Transformer


@dataclasses.dataclass
class Patches(hk.Module):
  """A module that extracts patches from an image and flattens them."""
  patch_size: int
  embed_dim: int

  def __call__(
      self,
      image_batch: jax.Array,  # [B, H, W, C]
  ) -> jax.Array:  # [B, T, D]
    conv = hk.Conv2D(
        output_channels=self.embed_dim,
        kernel_shape=self.patch_size,
        stride=self.patch_size // 2,
        padding='SAME'
    )
    patches = conv(image_batch)  # [B, H', W', D]
    b, h, w, d = patches.shape
    return jnp.reshape(patches, [b, h * w, d])


@dataclasses.dataclass
class VisionTransformer(hk.Module):
  """A ViT-style classifier."""

  transformer: Transformer
  model_size: int
  num_classes: int
  name: Optional[str] = None
  patch_size: int = 4

  def __call__(
      self,
      image_batch: jax.Array,
      *,
      is_training: bool = True,
  ) -> jax.Array:
    """Forward pass. Returns a sequence of logits."""
    extract_patches = Patches(patch_size=self.patch_size, embed_dim=self.model_size)
    patches = extract_patches(image_batch)  # [B, T, D]
    _, seq_len, _ = patches.shape

    # Embed the patches and positions.
    embed_init = hk.initializers.TruncatedNormal(stddev=0.2)
    embedding = hk.Linear(self.model_size, w_init=embed_init)
    patch_embedding = embedding(patches)  # [B, T, D]

    positional_embeddings = hk.get_parameter(
        'positional_embeddings', [seq_len, self.model_size], init=embed_init)
    input_embeddings = patch_embedding + positional_embeddings  # [B, T, D]

#    # Class Token TODO: ablate this
#    class_token = hk.get_parameter(
#        'class_token', [1, 1, self.model_size], init=embed_init)
#    class_token = jnp.tile(class_token, [input_chunks.shape[0], 1, 1])
#    input_embeddings = jnp.concatenate([class_token, input_embeddings], axis=1)


    # Run the transformer over the inputs.
    embeddings = self.transformer(
        input_embeddings,
        is_training=is_training,
    )  # [B, T, D]

    return hk.Linear(self.num_classes)(embeddings)  # [B, T, V]
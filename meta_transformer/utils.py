import functools
import jax
import jax.numpy as jnp
from jax import jit, vmap, random
import numpy as np
import collections
import dataclasses
import optax
from typing import Optional, Mapping, Any
import datasets
import dm_pix as pix


# Data
def load_data(dataset_name: str = "mnist"):
    """Load mnist, cifar10 or cifar100 dataset."""
    dataset = datasets.load_dataset(dataset_name)
    dataset = dataset.with_format("jax")
    train, test = dataset["train"], dataset["test"]
    train_images, train_labels = train["img"], train["label"]
    test_images, test_labels = test["img"], test["label"]
    return train_images, train_labels, test_images, test_labels


def random_rot90(rng, image):
    rot = random.bernoulli(rng, 0.5)
    return jax.lax.cond(
        rot,
        lambda img: pix.rot90(img, 1),
        lambda img: img,
        image
    )


@jit
def augment_datapoint(rng, img):
    """Apply a random augmentation to a single image. Pixel values are assumed to be in [0, 1]"""
    rng = random.split(rng, 6)
    img = pix.random_brightness(rng[0], img, 0.3)
    img = pix.random_contrast(rng[1], img, lower=0.2, upper=3)
    img = pix.random_saturation(rng[2], img, lower=0, upper=3)
    img = pix.random_flip_left_right(rng[2], img)
    img = pix.random_flip_up_down(rng[3], img)
    img = random_rot90(rng[4], img)
    return img


def get_image_patches(image: jnp.ndarray) -> jnp.ndarray:
    """Split image into 9 patches and return flattened patches. Image is 
    assumed to have shape (H, W, C). 
    Returns a jnp.array of shape (9, H*W*C/3)."""
    H, W = image.shape[:2]
    patch_size = 4
    # pad image to nearest multiple of patch_size
    try:
        image = jnp.pad(image, ((0, H % patch_size), (0, W % patch_size), (0, 0)))
    except:
        # MNIST has no channel dimension
        image = jnp.pad(image, ((0, H % patch_size), (0, W % patch_size)))

    # split image into patches
    num_patches = (H // patch_size) * (W // patch_size)
    patches = jnp.split(image, H // patch_size, axis=0)
    patches = jnp.concatenate(patches, axis=1)
    patches = jnp.array(jnp.split(patches, 
                                  num_patches,
                                  axis=1))
    patches = patches.reshape(num_patches, -1)
    return patches


def process_datapoint(rng: jnp.ndarray, 
                      img: jnp.array,
                      patch: bool = False,
                      augment: bool = True) -> jnp.array:
    img = img / 255.0
    img = jax.lax.cond(  # Augment?
            augment, 
            lambda img: augment_datapoint(rng, img),
            lambda img: img,
            img
        )
    
#     img = jax.lax.cond(  # Patches or chunks?
#             patch,
#             lambda img: get_image_patches(img).reshape(9, -1),
#             lambda img: img.reshape(9, -1),  # simple chunking for CIFAR10
#             img
#         )
    if patch:  # Patches or chunks?
        img = get_image_patches(img).reshape(16, -1)
    else:
        img = img.reshape(64, -1, order="F")
    return img


@functools.partial(jit, static_argnums=2)
def process_batch(rng, batch, patch = False, augment = True):
    """apply a random augmentation to a batch of images.
    input is assumed to be a jnp.array of shape (B, H, W, C) with 
    values in [0, 255]. Return augmented and reshaped batch."""
    rng = random.split(rng, len(batch))
    proc = functools.partial(process_datapoint, patch=patch, augment=augment)
    return vmap(proc)(rng, batch)


# Parameter chunking
def pad_to_chunk_size(arr: jnp.ndarray, chunk_size: int) -> jnp.ndarray:
    pad_size = -len(arr) % chunk_size
    padded = jnp.pad(arr, (0, pad_size))
    return padded


def chunk_layer(weights: jnp.ndarray, biases: jnp.ndarray, chunk_size: int) -> jnp.ndarray:
    flat_weights = weights.flatten()
    flat_weights = pad_to_chunk_size(flat_weights, chunk_size)
    weight_chunks = jnp.split(flat_weights, len(flat_weights) // chunk_size)
    biases = pad_to_chunk_size(biases, chunk_size)
    bias_chunks = jnp.split(biases, len(biases) // chunk_size)
    return weight_chunks, bias_chunks


def chunk_params(params: dict, chunk_size: int) -> dict:
    """
    Chunk the parameters of an MLP into chunks of size chunk_size.
    Chunks don't cross layer boundaries and are padded with zeros if necessary.
    """
    return {
        k: chunk_layer(layer["w"], layer["b"], chunk_size) for k, layer in params.items()
    }


def dict_concatenate(dict_list, np_array=False):
    """
    Arguments:
    * dict_list: a list of dictionaries with the same keys. All values must
    be numeric or a nested dict.
    Returns:
    * a dictionary with the same keys as the input dictionaries. The values
    are lists consisting of the concatenation of the values in the input dictionaries.
    """
    for d in dict_list:
        if not isinstance(d, collections.Mapping):
            raise TypeError("Input has to be a list consisting of dictionaries.")
        elif not all([dict_list[i].keys() == dict_list[i+1].keys() for i in range(len(dict_list)-1)]):
            raise ValueError("The keys of all input dictionaries need to match.")

    keys = dict_list[0].keys()
    out = {key: [d[key] for d in dict_list] for key in keys}

    if np_array:
        for k, v in out.items():
            try:
                out[k] = jnp.asarray(v)
            except TypeError:
                out[k] = dict_concatenate(v)
    else:
        for k, v in out.items():
            if isinstance(v[0], collections.Mapping):
                out[k] = dict_concatenate(v)
    return out


def count_params(params: dict) -> int:
    """Count the number of parameters in a dictionary of parameters."""
    return sum([x.size for x in jax.tree_util.tree_leaves(params)])


# TODO finish this
@dataclasses.dataclass
class Updater:
    """A stateless abstraction around an init_fn/update_fn pair.
    This extracts some common boilerplate from the training loop.
    """
    _net_init: callable
    _loss_fn: callable
    _accuracy_fn: callable
    _opt: optax.GradientTransformation

    @functools.partial(jax.jit, static_argnums=0)
    def init(self, rng, data):
        """Initializes state of the updater."""
        out_rng, init_rng = jax.random.split(rng)
        params = self._net_init(init_rng, data['text'])
        params = {k: dict(v) for k, v in params.items()}  # unfreeze
        opt_state = self._opt.init(params)
        out = dict(
            step=np.array(0),
            rng=out_rng,
            opt_state=opt_state,
            params=params,
        )
        return out

    @functools.partial(jax.jit, static_argnums=0)
    def update(self, state: Mapping[str, Any], data: Mapping[str, jnp.ndarray]):
        """Updates the state using some data and returns metrics."""
        rng, new_rng = jax.random.split(state['rng'])
        params = state['params']
        loss, g = jax.value_and_grad(self._loss_fn)(params, rng, data)

        updates, opt_state = self._opt.update(g, state['opt_state'], params)
        params = optax.apply_updates(params, updates)

        new_state = {
            'step': state['step'] + 1,
            'rng': new_rng,
            'opt_state': opt_state,
            'params': params,
        }

        metrics = {
            'step': state['step'],
            'train/loss': loss,
        }
        return new_state, metrics
    
    @functools.partial(jax.jit, static_argnums=0)
    def validate(self, state: Mapping[str, Any], val_data: Mapping[str, jnp.ndarray]):
        params = state['params']
        loss = self._loss_fn(params, None, val_data, is_training=False)
        
        val_metrics = {
            'step': state['step']-1,
            'validation/loss': loss,
            'validation/accuracy': self._accuracy_fn(params, val_data),
        }
        return val_metrics

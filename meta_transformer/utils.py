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


def process_datapoint(rng: jnp.ndarray, 
                      img: jnp.array,
                      augment: bool = True) -> jnp.array:
    img = img / 255.0
    img = jax.lax.cond(  # Random augment?
            augment, 
            lambda img: augment_datapoint(rng, img),
            lambda img: img,
            img
        )
    return img


@functools.partial(jit, static_argnums=2)
def process_batch(rng, batch, augment = True):
    """Apply a random augmentation to a batch of images.
    Input is assumed to be a jnp.array of shape (B, H, W, C) with 
    values in [0, 255]."""
    rng = random.split(rng, len(batch))
    proc = functools.partial(process_datapoint, augment=augment)
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


def dict_concatenate(dict_list, np_array=False): # TODO: make prettier
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
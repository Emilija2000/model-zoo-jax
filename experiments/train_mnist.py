from typing import Mapping, Any
from meta_transformer import transformer
import jax
from jax import random, vmap, grad, jit, value_and_grad
import jax.numpy as jnp
import haiku as hk
from jax import nn
from meta_transformer.mnist import mnist
from meta_transformer import utils
import optax
import numpy as np
import matplotlib.pyplot as plt
import time
from functools import partial
import datasets
import dataclasses
import functools


# Model
def forward(input_chunks, is_training=True):
    t = transformer.Classifier(
        transformer=transformer.Transformer(
            num_heads=3,
            num_layers=2,
            key_size=32,
            dropout_rate=0.1,
        ),
        model_size=128,
        num_classes=10,
    )
    return t(input_chunks, is_training=is_training)


model = hk.transform(forward)


# Loss
def loss(params, rng, input_chunks, targets, is_training=True):
    logits = model.apply(params, rng, input_chunks, is_training)[:, 0, :]  # [B, C]
    assert logits.shape == targets.shape
    return -jnp.sum(targets * nn.log_softmax(logits, axis=-1), axis=-1).mean()


# Data
train_data, train_labels, test_data, test_labels = mnist()


BATCH_SIZE = 32
SEQ_LEN = 14
CHUNK_SIZE = 784 // SEQ_LEN


train_data = train_data.reshape(-1, BATCH_SIZE, SEQ_LEN, CHUNK_SIZE)
train_labels = train_labels.reshape(-1, BATCH_SIZE, 10)
test_data = test_data.reshape(-1, SEQ_LEN, CHUNK_SIZE)
test_labels = test_labels.reshape(-1, 10)
# TODO normalize inputs




# Optimizer and update function
opt = optax.adam(1e-3)


@jit
def update_fn(params, opt_state, rng, batch):
    inputs, labels = batch
    l, g = value_and_grad(loss)(params, rng, inputs, labels)
    updates, opt_state = opt.update(g, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, l


@partial(jit, static_argnums=4)
def accuracy(params, rng, inputs, one_hot_targets, is_training=True):
    logits = model.apply(params, rng, inputs, is_training)[:, 0, :]
    predictions = jnp.argmax(logits, axis=-1)
    targets = jnp.argmax(one_hot_targets, axis=-1)
    return jnp.mean(predictions == targets)


@jit
def val_accuracy(params, rng):
    """Compute accuracy on test set."""
    test_acc = accuracy(params, rng, test_data, test_labels, is_training=False)
    return test_acc


def train_mnist(rng_key=random.PRNGKey(42)):
    subkey, rng_key = random.split(rng_key)
    params = model.init(subkey, train_data[0])
    opt_state = opt.init(params)
    print("Number of parameters:", sum([x.size for x in jax.tree_util.tree_leaves(params)]) / 1e6, "Million")

    info = []
    step = 0
    for epoch in range(2):
        print(f"Epoch {epoch}")
        acc = []
        losses = []
        for batch in zip(train_data, train_labels):
            subkey0, subkey1, rng_key = random.split(rng_key, 3)
            step += 1
            params, opt_state, l = update_fn(params, opt_state, subkey0, batch)
            train_acc = accuracy(params, subkey1, *batch)
            acc.append(train_acc) 
            losses.append(l)

        subkey, rng_key = random.split(rng_key)
        info.append(dict(loss=np.mean(losses), epoch=epoch, step=step, train_acc=np.mean(acc), val_acc=val_accuracy(params, subkey)))

    info = utils.dict_concatenate(info)
    return info, model, params


if __name__ == "__main__":
    info, model, params = train_mnist()
    print("Final training accuracy:", info["train_acc"][-1])
    print("Final test accuracy:", info["val_acc"][-1])


    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(5, 8))

    # acc
    ax = axs[0]
    ax.plot(info["step"], info["train_acc"], label="train")
    ax.plot(info["step"], info["val_acc"], label="val")
    ax.set_ylabel("Training Accuracy")
    ax.set_xlabel("Step")
    ax.legend()

    # loss
    ax = axs[1]
    ax.plot(info["step"], info["loss"], label="train loss")
    ax.set_ylabel("Training Loss")
    ax.set_xlabel("Step")
    ax.set_yscale("log")

    plt.show()
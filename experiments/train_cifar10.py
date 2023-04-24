from meta_transformer import transformer
import jax
from jax import random, grad, jit, value_and_grad
import jax.numpy as jnp
import haiku as hk
from jax import nn
from meta_transformer import utils
import optax
import numpy as np
import matplotlib.pyplot as plt
import functools
from typing import Mapping, Any, Tuple, List, Iterator, Optional
import chex
import matplotlib
import wandb

AUGMENT = True
USE_WANDB = True
DATA_MEAN = 0.473
DATA_STD = 0.251


# TODO don't augment at test time
# Model
def forward(image_batch, is_training=True):
    rng = hk.next_rng_key()
    image_batch = utils.process_batch(
        rng, 
        image_batch, 
        augment=AUGMENT,
    )
    image_batch = (image_batch - DATA_MEAN) / DATA_STD
    t = transformer.Classifier(
        transformer=transformer.Transformer(
            # I think we want model_size = key_size * num_heads
            num_heads=8,
            num_layers=6,
            key_size=32,
            dropout_rate=0.1,
        ),
        model_size=256,
        num_classes=10,
    )
    return t(image_batch, is_training=is_training)


model = hk.transform(forward)

def acc_from_logits(logits, targets):
    """expects index targets, not one-hot"""
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == targets)


def loss_from_logits(logits, targets):
    """targets are index labels"""
    targets = nn.one_hot(targets, 10)
    chex.assert_equal_shape([logits, targets])
    return -jnp.sum(targets * nn.log_softmax(logits, axis=-1), axis=-1).mean()


@functools.partial(jit, static_argnums=3)
def loss_fn(params, rng, data, is_training=True):
    """data is a dict with keys 'img' and 'label'."""
    images, targets = data["img"], data["label"]
    logits = model.apply(params, rng, images, is_training)[:, 0, :]  # [B, C]
    loss = loss_from_logits(logits, targets)
    acc = acc_from_logits(logits, targets)
    return loss, acc


@jit
def val_metrics(rng, params, val_data):
    """Compute acc and loss on test set."""
    loss, acc = loss_fn(params, rng, val_data, is_training=False)
    return {"val/acc": acc, "val/loss": loss}


# Optimizer and update function
@chex.dataclass
class TrainState:
    step: int
    rng: random.PRNGKey
    opt_state: optax.OptState
    params: dict


@chex.dataclass(frozen=True)  # needs to be immutable to be hashable
class Updater: # Could also make this a function of loss_fn, model.apply, etc if we want to be flexible
    """Holds training methods. All methods are jittable."""
    opt: optax.GradientTransformation

    @functools.partial(jit, static_argnums=0)
    def init_params(self, rng: jnp.ndarray, data: dict) -> dict:
        """Initializes state of the updater."""
        out_rng, k0, k1 = jax.random.split(rng, 3)
        params = model.init(k1, data["img"])
        opt_state = self.opt.init(params)
        return TrainState(
            step=0,
            rng=out_rng,
            opt_state=opt_state,
            params=params,
        )
    
    @functools.partial(jit, static_argnums=0)
    def update(self, state: TrainState, data: dict) -> Tuple[TrainState, dict]:
        state.rng, *subkeys = jax.random.split(state.rng, 3)
        (loss, acc), grads = value_and_grad(loss_fn, has_aux=True)(
                state.params, subkeys[1], data)
        updates, state.opt_state = self.opt.update(
            grads, state.opt_state, state.params)
        state.params = optax.apply_updates(state.params, updates)
        state.step += 1
        metrics = {
                "train/loss": loss,
                "train/acc": acc,
                "step": state.step,
        }
        return state, metrics


    @functools.partial(jit, static_argnums=0)
    def compute_val_metrics(self, 
                            state: TrainState, 
                            data: dict) -> Tuple[TrainState, dict]:
        state.rng, subkey = random.split(state.rng)
        return state, val_metrics(subkey, state.params, data)


@chex.dataclass
class Logger:
    # TODO: keep state for metrics instead of just pushing to wandb?
    # TODO: log mean of train_acc and train_loss
    log_interval: int = 50

    def log(self,
            state: TrainState,
            train_metrics: dict,
            val_metrics: dict = None):
        metrics = train_metrics
        if val_metrics is not None:
            metrics.update(val_metrics)
        metrics = {k: float(v) for k, v in metrics.items() if k != "step"}
        if state.step % self.log_interval == 0 or val_metrics is not None:
            print(", ".join([f"{k}: {round(v, 3)}" for k, v in metrics.items()]))
            wandb.log(metrics, step=state.step)


def shuffle_data(rng: jnp.ndarray, images: jnp.ndarray, labels: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Shuffle the data."""
    idx = jnp.arange(len(images))
    idx = random.permutation(rng, idx)
    return images[idx], labels[idx]


def data_iterator(images: jnp.ndarray, labels: jnp.ndarray, batchsize: int = 1048, skip_last: bool = False) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
    """Iterate over the data in batches."""
    for i in range(0, len(images), batchsize):
        if skip_last and i + batchsize > len(images):
            break
        yield dict(img=images[i:i + batchsize], 
                   label=labels[i:i + batchsize])


if __name__ == "__main__":
    rng = random.PRNGKey(42)
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 5e-5
    BATCH_SIZE = 128
    NUM_EPOCHS = 100

    wandb.init(
        mode="online" if USE_WANDB else "disabled",
        project="meta-models",
        config={
            "data_augmentation": AUGMENT,
            "dataset": "CIFAR-10",
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batchsize": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
        })  # TODO properly set and log config

    steps_per_epoch = 50000 // BATCH_SIZE

    # Data
    train_images, train_labels, test_images, test_labels = utils.load_data("cifar10")
    test_data = {"img": test_images, "label": test_labels}

    # Initialization
    opt = optax.adamw(LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    updater = Updater(opt=opt)
    logger = Logger(log_interval=50)
    rng, subkey = random.split(rng)
    state = updater.init_params(subkey, {
        "img": train_images[:2], 
        "label": train_labels[:2]
        })

    print("Number of parameters:", utils.count_params(state.params) / 1e6, "Million")
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    fig.tight_layout()
    lines = None

    # Training loop
    for epoch in range(NUM_EPOCHS):
        rng, subkey = random.split(rng)
        images, labels = shuffle_data(subkey, train_images, train_labels)
        batches = data_iterator(images, labels, batchsize=BATCH_SIZE, skip_last=True)

        for batch in batches:
            state, train_metrics = updater.update(state, batch)
            if state.step % 150 == 0:
                state, val_metrics = updater.compute_val_metrics(state, test_data)
                logger.log(state, train_metrics, val_metrics)
            else:
                logger.log(state, train_metrics)

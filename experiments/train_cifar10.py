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

AUGMENT = True

# Model
def forward(image_batch, is_training=True):
    rng = hk.next_rng_key()
    input_chunks = utils.process_batch(
        rng, 
        image_batch, 
        augment=AUGMENT,
    )
    t = transformer.Classifier(
        transformer=transformer.Transformer(
            num_heads=8,
            num_layers=6,
            key_size=64,
            dropout_rate=0.1,
        ),
        model_size=128,
        num_classes=10,
    )
    return t(input_chunks, is_training=is_training)


model = hk.transform(forward)


@functools.partial(jit, static_argnums=3)
def loss(params, rng, data, is_training=True):
    """data is a dict with keys 'img' and 'label'."""
    images, targets = data["img"], data["label"]
    targets = nn.one_hot(targets, 10)
    logits = model.apply(params, rng, images, is_training)[:, 0, :]  # [B, C]
    chex.assert_equal_shape([logits, targets])
    return -jnp.sum(targets * nn.log_softmax(logits, axis=-1), axis=-1).mean()


# Metrics
@functools.partial(jit, static_argnums=3)
def accuracy(rng, params, data, is_training=True):
    """data is a dict with keys 'img' and 'label'. labels are NOT one-hot."""
    targets, inputs = data["label"], data["img"]
    logits = model.apply(params, rng, inputs, is_training)[:, 0, :]
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == targets)


@jit
def val_metrics(rng, params, val_data):
    """Compute acc and loss on test set."""
    rngs = random.split(rng)
    acc = accuracy(rngs[0], params, val_data, is_training=False)
    l = loss(params, rngs[1], val_data, is_training=False)
    return {"val/acc": acc, "val/loss": l}


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
    def update(self, state: TrainState, data: dict) -> TrainState:
        state.rng, *subkeys = jax.random.split(state.rng, 3)
        grads = grad(loss)(state.params, subkeys[1], data)       
        updates, state.opt_state = self.opt.update(
            grads, state.opt_state, state.params)
        state.params = optax.apply_updates(state.params, updates)
        state.step += 1
        return state

    @functools.partial(jit, static_argnums=0)
    def compute_validation_metrics(self, 
                                   state: TrainState, 
                                   data: dict) -> Tuple[TrainState, dict]:
        state.rng, subkey = random.split(state.rng)
        return state, val_metrics(subkey, state.params, data)
    
    @functools.partial(jit, static_argnums=0)
    def compute_metrics(self, 
                        state: TrainState, 
                        train_data: dict, 
                        val_data: dict) -> Tuple[TrainState, dict]:
        state.rng, *subkeys = random.split(state.rng, 4)
        train_acc = accuracy(subkeys[1], state.params, train_data)
        train_loss = loss(state.params, subkeys[2], train_data)
        state, val_metrics = self.compute_validation_metrics(state, val_data)
        return state, {
            "train/acc": train_acc,
            "train/loss": train_loss,
            "step": state.step,
            **val_metrics
            }


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


# Plotting
def plot_metrics(metrics, axs: np.ndarray, prefix: str = "", lines: Optional[list] = None):
    if isinstance(metrics, list):
        metrics = utils.dict_concatenate(metrics)
        metrics = {k: np.array(v) for k, v in metrics.items()}

    if lines is None:  # First call - initialize plot
        # acc
        ax = axs[0]
        ta, = ax.plot(metrics["step"], metrics["train/acc"], label=prefix + "train acc")
        va, = ax.plot(metrics["step"], metrics["val/acc"], label=prefix + "val acc")
        ax.set_ylabel("Training Accuracy")
        ax.set_xlabel("Step")
        ax.legend()

        # loss
        ax = axs[1]
        tl, = ax.plot(metrics["step"], metrics["train/loss"], label=prefix + "train loss")
        vl, = ax.plot(metrics["step"], metrics["val/loss"], label=prefix + "val loss")
        ax.set_ylabel("Training Loss")
        ax.set_xlabel("Step")
        ax.set_yscale("log")
        ax.legend()
        lines = [ta, va, tl, vl]
        return lines
    else:  # Update lines
        lines[0].set_data(metrics["step"], metrics["train/acc"])
        lines[1].set_data(metrics["step"], metrics["val/acc"])
        lines[2].set_data(metrics["step"], metrics["train/loss"])
        lines[3].set_data(metrics["step"], metrics["val/loss"])
        axs[0].autoscale_view()
        axs[0].relim()
        axs[1].autoscale_view()
        axs[1].relim()
        plt.pause(.05)
        return lines


if __name__ == "__main__":
    try:
        matplotlib.use("TkAgg")
    except ImportError:
        pass
    rng = random.PRNGKey(42)
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE = 256
    NUM_EPOCHS = 100

    steps_per_epoch = 50000 // BATCH_SIZE

    # Data
    train_images, train_labels, test_images, test_labels = utils.load_data("cifar10")
    test_data = {"img": test_images, "label": test_labels}

    # Initialization
    opt = optax.adamw(LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    updater = Updater(opt=opt)
    state = updater.init_params(rng, {
        "img": train_images[:2], 
        "label": train_labels[:2]
        })
    print("Number of parameters:", sum([x.size for x in jax.tree_util.tree_leaves(state.params)]) / 1e6, "Million")
    metrics_list = []

    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    fig.tight_layout()
    lines = None

    # Training loop
    try:
        for epoch in range(NUM_EPOCHS):
            rng, subkey = random.split(rng)
            images, labels = shuffle_data(subkey, train_images, train_labels)
            batches = data_iterator(images, labels, batchsize=BATCH_SIZE, skip_last=True)
            for batch in batches:
                if state.step % 150 == 0:
                    state, metrics = updater.compute_metrics(
                        state, batch, test_data)
                    metrics_list.append(metrics)
                    print("Epoch:", epoch, "Step:", state.step, "Train acc:", metrics["train/acc"], "Val acc:", metrics["val/acc"])
                    lines = plot_metrics(metrics_list, axs, lines=lines)
                state = updater.update(state, batch)
    except KeyboardInterrupt:
        pass


    # Plot
    metrics = utils.dict_concatenate(metrics_list)
    print("Final training accuracy:", metrics["train/acc"][-1])
    print("Final validation accuracy:", metrics["val/acc"][-1])
    lines = plot_metrics(metrics, axs, lines=lines)
    plt.show()




import jax
from jax import random, jit, value_and_grad, nn
import jax.numpy as jnp
import numpy as np
import haiku as hk
import optax
import chex
import functools
from typing import Mapping, Any, Tuple, List, Iterator, Optional, Dict
from jax.typing import ArrayLike
from meta_transformer import utils, meta_model, MetaModelClassifier, Transformer
import matplotlib.pyplot as plt
import wandb
from nninn.repl.utils import load_nets, classes_per_task, random_data_view, shuffle_and_split_data
import nninn
import os
import argparse


def flatten(x):
    return jax.flatten_util.ravel_pytree(x)[0]


# TODO replace all this with huggingface datasets
def split_data(data: list, labels: list):
    split_index = int(len(data)*0.8)
    return (data[:split_index], labels[:split_index], 
            data[split_index:], labels[split_index:])


def is_fine(params: dict):
    """Return false if std or mean is too high."""
    flat = flatten(params)
    if flat.std() > 5.0 or jnp.abs(flat.mean()) > 5.0:
        return False
    else:
        return True


def filter_data(data: List[dict], labels: List[ArrayLike]):
    """Given a list of net params, filter out those
    with very large means or stds."""
    assert len(data) == len(labels)
    f_data, f_labels = zip(*[(x, y) for x, y in zip(data, labels) if is_fine(x)])
    print(f"Filtered out {len(data) - len(f_data)} nets.\
          That's {100*(len(data) - len(f_data))/len(data):.2f}%.")
    return np.array(f_data), np.array(f_labels)


# Initialize meta-model
def model_fn(params_batch: dict, 
             dropout_rate: float = 0.1, 
             is_training: bool = False):
    net = MetaModelClassifier(
        model_size=4*32, 
        num_classes=4, 
        transformer=Transformer(
            num_heads=4,
            num_layers=2,
            key_size=32,
            dropout_rate=dropout_rate,
        ))
    return net(params_batch, is_training=is_training)


model = hk.transform(model_fn)
    

def acc_from_logits(logits, targets):
    """expects index targets, not one-hot"""
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == targets)


def loss_from_logits(logits, targets):
    """targets are index labels"""
    targets = nn.one_hot(targets, logits.shape[-1])
    chex.assert_equal_shape([logits, targets])
    return -jnp.sum(targets * nn.log_softmax(logits, axis=-1), axis=-1).mean()


@functools.partial(jit, static_argnums=3)
def loss_fn(params, rng, data: Dict[str, ArrayLike], is_training: bool = True):
    """data is a dict with keys 'input' and 'label'."""
    inputs, targets = data["input"], data["label"]
    logits = model.apply(params, rng, inputs, is_training)  # [B, C]
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


@chex.dataclass(frozen=True)  # needs to be immutable to be hashable (for static_argnums)
class Updater: # Could also make this a function of loss_fn, model.apply, etc if we want to be flexible
    """Holds training methods. All methods are jittable."""
    opt: optax.GradientTransformation

    @functools.partial(jit, static_argnums=0)
    def init_params(self, rng: jnp.ndarray, data: dict) -> dict:
        """Initializes state of the updater."""
        out_rng, subkey = jax.random.split(rng)
        params = model.init(subkey, data["input"])
        opt_state = self.opt.init(params)
        return TrainState(
            step=0,
            rng=out_rng,
            opt_state=opt_state,
            params=params,
        )
    
    @functools.partial(jit, static_argnums=0)
    def update(self, state: TrainState, data: dict) -> Tuple[TrainState, dict]:
        state.rng, subkey = jax.random.split(state.rng)
        (loss, acc), grads = value_and_grad(loss_fn, has_aux=True)(
                state.params, subkey, data)
        updates, state.opt_state = self.opt.update(
            grads, state.opt_state, state.params)
        state.params = optax.apply_updates(state.params, updates)
        metrics = {
                "train/loss": loss,
                "train/acc": acc,
                "step": state.step,
        }
        state.step += 1
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
    # TODO: keep state in between log_intervals; compute mean of 
    # train_acc and train_loss, and then log that at the end 
    # of the interval. Also keep track of val_acc and val_loss.
    log_interval: int = 50
    disable_wandb: bool = False

    def log(self,
            state: TrainState,
            train_metrics: dict = None,
            val_metrics: dict = None):
        metrics = train_metrics or {}
        if val_metrics is not None:
            metrics.update(val_metrics)
        metrics = {k: float(v) for k, v in metrics.items() if k != "step"}
        if state.step % self.log_interval == 0 or val_metrics is not None:
            print(", ".join([f"{k}: {round(v, 3)}" for k, v in metrics.items()]))
            if not self.disable_wandb:
                wandb.log(metrics, step=state.step)


def shuffle_data(rng: jnp.ndarray, inputs: jnp.ndarray, labels: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Shuffle the data."""
    idx = jnp.arange(len(inputs))
    idx = random.permutation(rng, idx)
    return inputs[idx], labels[idx]


def data_iterator(inputs: jnp.ndarray, labels: jnp.ndarray, batchsize: int = 1048, skip_last: bool = False) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
    """Iterate over the data in batches."""
    for i in range(0, len(inputs), batchsize):
        if skip_last and i + batchsize > len(inputs):
            break
        yield dict(input=inputs[i:i + batchsize], 
                   label=labels[i:i + batchsize])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training run')
    parser.add_argument('--lr', type=float, help='Learning rate', default=5e-5)
    parser.add_argument('--wd', type=float, help='Weight decay', default=1e-3)
    parser.add_argument('--bs', type=int, help='Batch size', default=32)
    parser.add_argument('--epochs', type=int, help='Number of epochs', 
                        default=25)
    parser.add_argument('--task', type=str, help='Task to train on. One of \
                        "batch_size", "augmentation", "optimizer", \
                        "activation", "initialization"', default="batch_size")
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb')
    args = parser.parse_args()


    rng = random.PRNGKey(42)
    LEARNING_RATE = args.lr  # TODO don't use extra variables
    WEIGHT_DECAY = args.wd
    BATCH_SIZE = args.bs
    NUM_EPOCHS = args.epochs
    TASK = args.task
    USE_WANDB = args.use_wandb

    # Load MNIST model checkpoints
    print(f"Training task: {TASK}.")
    inputs, all_labels = load_nets(n=500, data_dir=os.path.join(nninn.data_dir, "ctc_fixed"), 
                                flatten=False, verbose=False);
    labels = all_labels[TASK]
    filtered_inputs, filtered_labels = filter_data(inputs, labels)
    train_inputs, train_labels, val_inputs, val_labels = split_data(filtered_inputs, filtered_labels)
    val_data = {"input": utils.tree_stack(val_inputs), "label": val_labels}


    wandb.init(
        mode="online" if USE_WANDB else "disabled",
        project="meta-models",
        tags=[],
        config={
            "dataset": "MNIST-meta",
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batchsize": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "target_task": TASK,
            "dropout": 0.1,
        },
        notes="First time trying out transformer meta-model.",
        )  # TODO properly set and log config

    steps_per_epoch = len(train_inputs) // BATCH_SIZE

    print()
    print(f"Number of training examples: {len(train_inputs)}.")
    print("Steps per epoch:", steps_per_epoch)
    print("Total number of steps:", steps_per_epoch * NUM_EPOCHS)
    print()

    # Initialization
    opt = optax.adamw(LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    updater = Updater(opt=opt)
    logger = Logger(log_interval=5)
    rng, subkey = random.split(rng)
    state = updater.init_params(subkey, {
        "input": utils.tree_stack(train_inputs[:2]),
        "label": train_labels[:2]
        })

    print("Number of parameters:", utils.count_params(state.params) / 1e6, "Million")

    # Training loop
    for epoch in range(NUM_EPOCHS):
        rng, subkey = random.split(rng)
        images, labels = shuffle_data(subkey, train_inputs, train_labels)
        batches = data_iterator(images, labels, batchsize=BATCH_SIZE, skip_last=True)

        # Validate every epoch
        state, val_metrics_dict = updater.compute_val_metrics(state, val_data)
        logger.log(state, val_metrics_dict)

        for batch in batches:
            batch["input"] = utils.tree_stack(batch["input"])
            state, train_metrics = updater.update(state, batch)
            logger.log(state, train_metrics)
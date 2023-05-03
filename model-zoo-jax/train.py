import chex
from dataclasses import dataclass
import functools
import jax
from jax import jit, vmap, value_and_grad
import jax.numpy as jnp
import optax
from typing import Callable, Tuple

from losses import Evaluator
from utils import TrainState

from config import Parameters
    
@dataclass(frozen=True)  # needs to be immutable to be hashable
class Updater: 
    """Holds training methods. All methods are jittable."""
    opt: optax.GradientTransformation
    evaluator: Evaluator
    model_init: Callable

    @functools.partial(jit, static_argnums=(0))
    def init_params(self, rng: jnp.ndarray, x:jnp.ndarray) -> dict:
        """Initializes state of the updater."""
        out_rng, k0 = jax.random.split(rng)
        params = self.model_init(rng=k0, x=x, is_training=True)
        opt_state = self.opt.init(params)
        
        return TrainState(
            step=0,
            rng=out_rng,
            opt_state=opt_state,
            params=params
        )
    
    @functools.partial(jit, static_argnums=0)
    def train_step(self, state: TrainState, data:dict) -> Tuple[TrainState, dict]:
        state.rng, *subkeys = jax.random.split(state.rng, 3)
        (loss, acc), grads = value_and_grad(self.evaluator.train_metrics, has_aux=True)(state.params, subkeys[1], data)
        updates, state.opt_state = self.opt.update(grads, state.opt_state, state.params)
        state.params = optax.apply_updates(state.params, updates)
        state.step += 1
        metrics = {
                "train/loss": loss,
                "train/acc": acc,
                "step": state.step,
        }
        return state, metrics

    @functools.partial(jit, static_argnums=0)
    def val_step(self, state: TrainState, data: dict) -> Tuple[TrainState, dict]:
        state.rng, subkey = jax.random.split(state.rng)
        loss, acc = self.evaluator.val_metrics(state.params, subkey, data)
        metrics = {
                "val/loss": loss,
                "val/acc": acc
        }
        return state, metrics
    
if __name__ == '__main__':
    from config import Parameters
    from datasets.dropclassdataset import load_drop_class_dataset, get_dataloaders, split_train_dataset
    from losses import CrossEntropyLoss
    from models.models import get_model
    from logger import Logger
    
    import numpy as np
    
    seed = 2
    key = jax.random.PRNGKey(seed)
    config = Parameters(seed=key, class_dropped=0)
    
    datasets = load_drop_class_dataset(config.dataset, config.class_dropped)
    datasets = split_train_dataset(datasets)
    dataloaders = get_dataloaders(datasets, config.batch_size)
    
    model = get_model(config)
    batch_apply = vmap(model.apply, in_axes=(None,None,0,None),axis_name='batch')
    
    evaluator = CrossEntropyLoss(batch_apply, config.num_classes)
    if config.optimizer == 'adamW':
        optimizer = optax.adamw(learning_rate=config.lr, weight_decay=config.weight_decay)
    else:
        optimizer = optax.chain(
            optax.add_decayed_weights(config.weight_decay), 
            optax.sgd(config.lr) 
        )
    updater = Updater(opt=optimizer, evaluator=evaluator, model_init=model.init)

    state = updater.init_params(rng=config.seed,x=datasets['train'][0][0])
    
    logger = Logger(name="trial", config=config,log_interval=500)
    logger.init()

    for epoch in range(config.num_epochs):
        for batch in dataloaders['train']:
            state, train_metrics = updater.train_step(state, batch)
            logger.log(state, train_metrics)
            
        val_acc = []
        val_loss = []
        for val_batch in dataloaders['val']:
            state, val_metrics = updater.val_step(state, val_batch)
            val_acc.append(val_metrics['val/acc'].item())
            val_loss.append(val_metrics['val/loss'].item())
        val_metrics = {'val/acc':np.mean(val_acc), 'val/loss':np.mean(val_loss)}
        logger.log(state, train_metrics, val_metrics)
                
                
    
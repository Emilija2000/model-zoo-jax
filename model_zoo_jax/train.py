import chex
from dataclasses import dataclass
import functools
import jax
from jax import jit, vmap, value_and_grad
import jax.numpy as jnp
import optax
from typing import Callable, Tuple

from model_zoo_jax.losses import Evaluator
from model_zoo_jax.utils import TrainState
    
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
        if isinstance(params, tuple) and len(params)==2:
            params,model_state=params
        else:
            model_state=None
        opt_state = self.opt.init(params)
        
        return TrainState(
            step=0,
            rng=out_rng,
            opt_state=opt_state,
            params=params,
            model_state=model_state
        )
    
    @functools.partial(jit, static_argnums=0)
    def train_step(self, state: TrainState, data:dict) -> Tuple[TrainState, dict]:
        state.rng, *subkeys = jax.random.split(state.rng, 3)
        
        (loss, (acc, new_state)), grads = value_and_grad(self.evaluator.train_metrics, has_aux=True)(state.params, subkeys[1], data,state.model_state)
    
        state.model_state=new_state
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
        loss, (acc,state.model_state) = self.evaluator.val_metrics(state.params, subkey, data,state.model_state)
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
    print("Training dataset: {}".format(len(datasets['train'])))
    print("Test dataset: {}".format(len(datasets['test'])))
    
    #for debugging
    #from torch.utils.data import Subset
    #datasets['train'] = Subset(datasets['train'],np.arange(1000))
    
    dataloaders = get_dataloaders(datasets, config.batch_size)
    
    model,is_batch = get_model(config)
    
    if not(is_batch):
        batch_apply = vmap(model.apply, in_axes=(None,None,0,None),axis_name='batch')
        init_x = datasets['train'][0][0]
    else:
        batch_apply = model.apply
        init_x = next(iter(dataloaders['train']))[0]
    
    evaluator = CrossEntropyLoss(batch_apply, config.num_classes)
    if config.optimizer == 'adamW':
        optimizer = optax.adamw(learning_rate=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == 'sgd_scheduler':
        #total_steps = config.num_epochs*(len(dataloaders['train']))
        #schedule = optax.warmup_cosine_decay_schedule(
        #    init_value=0.0,
        #    peak_value=config.lr,
        #    warmup_steps=50,
        #    decay_steps=total_steps,
        #    end_value=0.0,
        #    )
        schedule=optax.exponential_decay(init_value=config.lr,
                                transition_steps=50*len(dataloaders['train']),
                                decay_rate=0.2,
                                staircase=True)


        optimizer = optax.chain(
            optax.add_decayed_weights(config.weight_decay), 
            optax.sgd(learning_rate=schedule,momentum=0.9,nesterov=True)
        )
    else:
        optimizer = optax.chain(
            optax.add_decayed_weights(config.weight_decay), 
            optax.sgd(config.lr,momentum=0.9)
        )
    updater = Updater(opt=optimizer, evaluator=evaluator, model_init=model.init)
    
    state = updater.init_params(rng=config.seed,x=init_x)
    print("Param count:", sum(x.size for x in jax.tree_util.tree_leaves(state.params)))
    
    logger = Logger(name="trial", config=config,log_interval=500,log_wandb=True)
    logger.init()

    for epoch in range(config.num_epochs):
        train_all_acc = []
        train_all_loss = []
        for batch in dataloaders['train']:
            state, train_metrics = updater.train_step(state, batch)
            logger.log(state, train_metrics)
            train_all_acc.append(train_metrics['train/acc'].item())
            train_all_loss.append(train_metrics['train/loss'].item())
        train_metrics = {'train/acc':np.mean(train_all_acc), 'train/loss':np.mean(train_all_loss)}
            
        val_acc = []
        val_loss = []
        for val_batch in dataloaders['val']:
            state, val_metrics = updater.val_step(state, val_batch)
            val_acc.append(val_metrics['val/acc'].item())
            val_loss.append(val_metrics['val/loss'].item())
        val_metrics = {'val/acc':np.mean(val_acc), 'val/loss':np.mean(val_loss)}
        logger.log(state, train_metrics, val_metrics)
                
                
    
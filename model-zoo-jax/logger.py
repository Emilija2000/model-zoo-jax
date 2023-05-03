import chex
import jax
import numpy as np
import os
import pickle
from typing import Optional

from utils import TrainState
import wandb

def model_save(ckpt_dir: str, state) -> None:
    "credit: https://github.com/deepmind/dm-haiku/issues/18?fbclid=IwAR0aSk2OgYCIn3YKFrDoEnSYU1xRYzywuypVQlunsZHn2w5y1vpN9_b8QXM"
    with open(os.path.join(ckpt_dir, "arrays.npy"), "wb") as f:
        for x in jax.tree_util.tree_leaves(state):
            np.save(f, x, allow_pickle=False)

    tree_struct = jax.tree_map(lambda t: 0, state)
    with open(os.path.join(ckpt_dir, "tree.pkl"), "wb") as f:
        pickle.dump(tree_struct, f)

def model_restore(ckpt_dir):
    "credit: https://github.com/deepmind/dm-haiku/issues/18?fbclid=IwAR0aSk2OgYCIn3YKFrDoEnSYU1xRYzywuypVQlunsZHn2w5y1vpN9_b8QXM"
    with open(os.path.join(ckpt_dir, "tree.pkl"), "rb") as f:
        tree_struct = pickle.load(f)
 
    leaves, treedef = jax.tree_util.tree_flatten(tree_struct)
    with open(os.path.join(ckpt_dir, "arrays.npy"), "rb") as f:
        flat_state = [np.load(f) for _ in leaves]

    return jax.tree_util.tree_unflatten(treedef, flat_state)


@chex.dataclass
class Logger:
    name: str
    config: dict
    log_interval: Optional[int] = 50
    checkpoint_dir: Optional[str] = "checkpoints"
    log_wandb:Optional[bool] = True
    
    def init(self):
        wandb.init(config=self.config, project=self.name)
        self.save_config()

    def wandb_log(self,
            state: TrainState,
            train_metrics: dict,
            val_metrics: dict = None):
        metrics = train_metrics
        if val_metrics is not None:
            metrics.update(val_metrics)
        metrics = {k: float(v) for k, v in metrics.items() if k != "step"}
        wandb.log(metrics, step=state.step)
        print(", ".join([f"{k}: {round(v, 3)}" for k, v in metrics.items()]))
            
    def save_checkpoint(self, state:TrainState, train_metrics:dict, val_metrics:dict=None):
        checkpoint_name = os.path.join(self.checkpoint_dir,self.name+"_"+str(state.step))
        if not os.path.exists(checkpoint_name):
            os.makedirs(checkpoint_name)
        model_save(checkpoint_name, state.params)
        
        metrics = train_metrics
        metrics.update(val_metrics)
        with open(os.path.join(checkpoint_name, "metrics.pkl"), "wb") as f:
            pickle.dump(dict(metrics), f)
            
    def save_config(self):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        with open(os.path.join(self.checkpoint_dir, "config.pkl"), "wb") as f:
            pickle.dump(dict(self.config), f)
        
    def log(self, 
            state: TrainState,
            train_metrics: dict,
            val_metrics: dict = None):
        if self.log_wandb and (state.step % self.log_interval == 0 or val_metrics is not None):
            self.wandb_log(state,train_metrics,val_metrics)
        if val_metrics is not None:
            self.save_checkpoint(state,train_metrics,val_metrics)
        
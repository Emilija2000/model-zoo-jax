from dataclasses import dataclass
from typing import Callable
import functools

import jax
import jax.numpy as jnp
from jax import jit

@dataclass(frozen=True)
class Evaluator:
    model_apply: Callable
    num_classes: int
    
    def acc_from_logits(self, logits, targets):
        pass
    
    def loss_from_logits(self, logits, targets):
        pass
    
    @functools.partial(jit, static_argnums=(0))
    def train_metrics(self, params, rng, data,state=None):
        images = data[0]
        targets = data[1]
        if state is None:
            logits = self.model_apply(params, rng, images, True) 
        else:
            logits,state=self.model_apply(params,state,rng,images,True)
        loss = self.loss_from_logits(logits, targets)
        acc = self.acc_from_logits(logits, targets)
        
        return loss, (acc, state)
    
    @functools.partial(jit, static_argnums=(0))
    def val_metrics(self, params, rng, data,state=None):
        images = data[0]
        targets = data[1]
        if state is None:
            logits = self.model_apply(params, rng, images, False)
        else:
            logits,state = self.model_apply(params, state, rng, images, False)
        loss = self.loss_from_logits(logits, targets)
        acc = self.acc_from_logits(logits, targets)
        return loss, (acc, state)
    
class CrossEntropyLoss(Evaluator):
    def acc_from_logits(self, logits, targets):
        """expects index targets, not one-hot"""
        predictions = jnp.argmax(logits, axis=-1)
        return jnp.mean(predictions == targets)

    def loss_from_logits(self, logits, targets):
        """targets are index labels"""
        targets = jax.nn.one_hot(targets, self.num_classes)
        return -jnp.sum(targets * jax.nn.log_softmax(logits, axis=-1), axis=-1).mean()

from dataclasses import dataclass
from typing import Callable,Optional
import functools

import jax
import jax.numpy as jnp
from jax import jit

@dataclass(frozen=True)
class Evaluator:
    model_apply: Callable
    num_classes: Optional[int]=1
    
    def metric_from_logits(self, logits, targets):
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
        acc = self.metric_from_logits(logits, targets)
        
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
        acc = self.metric_from_logits(logits, targets)
        return loss, (acc, state)
    
class CrossEntropyLoss(Evaluator):
    def metric_from_logits(self, logits, targets):
        """expects index targets, not one-hot"""
        predictions = jnp.argmax(logits, axis=-1).reshape(-1,1)
        return jnp.mean(predictions == targets.reshape(-1,1))

    def loss_from_logits(self, logits, targets):
        """targets are index labels"""
        targets = jax.nn.one_hot(targets.flatten(), self.num_classes)
        return -jnp.sum(targets * jax.nn.log_softmax(logits, axis=-1), axis=-1).mean()

class MSELoss(Evaluator):
    def r_squared_score(self, predictions, targets):
        ss_res = jnp.sum(jnp.square(targets - predictions))
        ss_tot = jnp.sum(jnp.square(targets - jnp.mean(targets)))
        r_squared = 1.0 - (ss_res / ss_tot)
        return r_squared

    def loss_from_logits(self, logits, targets):
        """targets are continuous values"""
        return jnp.mean(jnp.square(logits - targets))

    def metric_from_logits(self, logits, targets):
        #return self.r_squared_score(logits, targets)
        return jnp.sum(jnp.square(targets - logits))

''' 
class MSLELoss(Evaluator):
    def r_squared_score(self, predictions, targets):
        ss_res = jnp.sum(jnp.square(targets - jnp.expm1(predictions)))
        ss_tot = jnp.sum(jnp.square(targets - jnp.mean(targets)))
        r_squared = 1.0 - (ss_res / ss_tot)
        return r_squared

    def loss_from_logits(self, logits, targets):
        """targets are continuous values"""
        #jax.debug.print('logits {}',logits)
        #jax.debug.print('targets {}',targets)
        #jax.debug.print('square shape {}',jnp.square(logits - jnp.log1p(targets)))
        return jnp.mean(jnp.square(logits - jnp.log1p(targets)))

    def metric_from_logits(self, logits, targets):
        #return self.r_squared_score(logits, targets)
        return jnp.sum(jnp.square(targets - jnp.expm1(logits)))
'''

class MSLELoss(Evaluator):
    def sigmoid_to_lr(self, x):
        """Transform to output range (0, 1)"""
        return 1 / (1 + jnp.exp(-x))

    def r_squared_score(self, predictions, targets):
        logits_transformed = self.sigmoid_to_lr(predictions)
        ss_res = jnp.sum(jnp.square(targets - logits_transformed))
        ss_tot = jnp.sum(jnp.square(targets - jnp.mean(targets)))
        r_squared = 1.0 - (ss_res / ss_tot)
        return r_squared

    def loss_from_logits(self, logits, targets):
        """targets are continuous values"""
        logits_transformed = self.sigmoid_to_lr(logits)
        #jax.debug.print('logits {}', logits_transformed)
        #jax.debug.print('targets {}', targets)
        #jax.debug.print('square shape {}', jnp.square(jnp.log1p(logits_transformed) - jnp.log1p(targets)))
        return jnp.mean(jnp.square(jnp.log1p(logits_transformed) - jnp.log1p(targets)))

    def metric_from_logits(self, logits, targets):
        logits_transformed = self.sigmoid_to_lr(logits)
        return jnp.sum(jnp.square(targets - logits_transformed))


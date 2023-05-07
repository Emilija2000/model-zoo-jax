import chex
import jax.numpy as jnp 
from jax.random import PRNGKey
from optax import OptState
from typing import Tuple,Optional

@chex.dataclass
class TrainState:
    step: int
    rng: PRNGKey
    opt_state: OptState
    params: dict
    model_state: Optional[Tuple[jnp.array]] = None
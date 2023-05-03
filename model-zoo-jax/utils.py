import chex
from jax.random import PRNGKey
from optax import OptState

@chex.dataclass
class TrainState:
    step: int
    rng: PRNGKey
    opt_state: OptState
    params: dict
from typing import Callable
from typing import Tuple

from jax import numpy as jnp
from jax import random
from jaxtyping import Array

from zephyr.project_typing import KeyArray

Initializer = Callable[[random.PRNGKey, Tuple], Array]


def initializer_base(key: random.PRNGKey, shape) -> Array:
    return random.uniform(key, shape)


def ones(key, shape):
    return jnp.ones(shape)


def uniform(key, shape):
    return random.uniform(key, shape)


def normal(key: KeyArray, shape: tuple[int, ...]) -> Array:
    return random.normal(key, shape)

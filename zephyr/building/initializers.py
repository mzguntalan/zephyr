from typing import Callable
from typing import Literal
from typing import Tuple

import numpy as np
from jax import numpy as jnp
from jax import random
from jaxtyping import Array

from zephyr.project_typing import ActivationFunctionsWithKnownGain
from zephyr.project_typing import KeyArray
from zephyr.project_typing import Shape

Initializer = Callable[[KeyArray, Tuple[int, ...]], Array]
from typing import Union
from jax.nn.initializers import truncated_normal


def initializer_base(
    key: KeyArray, shape: Shape, feature_shape: int | None = None
) -> Array:
    feature_shape = feature_shape or shape[0]  # shape[0] is assumed to be feature_shape
    standard_deviation = 1 / np.sqrt(feature_shape)

    if len(shape) > 1:
        return truncated_normal(standard_deviation)(key, shape)

    return zeros(key, shape)


def ones(key: KeyArray, shape: Shape) -> Array:
    return jnp.ones(shape)


def zeros(key: KeyArray, shape: Shape) -> Array:
    return jnp.zeros(shape)


def uniform(key: KeyArray, shape: Shape) -> Array:
    return random.uniform(key, shape)


def normal(key: KeyArray, shape: Shape) -> Array:
    return random.normal(key, shape)

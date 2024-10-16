from functools import partial
from typing import Callable
from typing import Tuple

from jax import random
from jaxtyping import Array

from zephyr.building.initializers import Initializer
from zephyr.building.initializers import initializer_base

ArrayTemplate = Callable[[random.PRNGKey], Array]


def array(shape: tuple, initializer: Initializer = initializer_base) -> ArrayTemplate:
    return partial(initializer, shape=shape)

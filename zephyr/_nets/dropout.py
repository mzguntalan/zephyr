from jax import random

from zephyr.masking import apply_mask
from zephyr.project_typing import Array
from zephyr.project_typing import KeyArray


def dropout(key: KeyArray, x: Array, drop_probability: float) -> Array:
    mask = 1.0 * (random.uniform(key, x.shape) > drop_probability)
    return apply_mask(x, mask)

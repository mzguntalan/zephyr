from jax import random

from zephyr.masking import apply_mask
from zephyr.project_typing import Array
from zephyr.project_typing import KeyArray


def dropout(key: KeyArray, x: Array, drop_probability: float) -> Array:
    mask = 1.0 * (random.uniform(key, x.shape) > drop_probability)
    keep_probability = 1 - dropout
    scale_factor_so_that_the_sums_in_the_next_layers_are_in_the_same_range = 1 / (
        keep_probability + 1e-19
    )
    return (
        apply_mask(x, mask)
        * scale_factor_so_that_the_sums_in_the_next_layers_are_in_the_same_range
    )

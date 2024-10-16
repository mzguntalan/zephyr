from jax import numpy as jnp
from jaxtyping import Array


def apply_mask(x: Array, masks: Array) -> Array:
    return x * masks


def apply_attention_mask(x: Array, masks: Array) -> Array:
    # put large negative values instead of zeros
    return apply_mask(x, masks) + apply_mask((-1e16) * jnp.ones_like(x), 1 - masks)

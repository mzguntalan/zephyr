from typing import Optional

from jax import nn
from jax import numpy as jnp
from jax.lax import rsqrt
from jaxtyping import Array
from jaxtyping import PyTree

from zephyr._nets.mlp import branch_linear
from zephyr._nets.mlp import linear
from zephyr.building import initializers
from zephyr.building import template
from zephyr.building.initializers import Initializer
from zephyr.masking import apply_attention_mask


def layer_norm(
    params: PyTree,
    x: Array,
    axis: int,
    create_scale: bool = True,
    create_offset: bool = True,
    eps: float = 1e-16,
    initializer: Initializer = initializers.initializer_base,
) -> Array:
    mean = jnp.mean(x, axis=axis, keepdims=True)
    variance = jnp.var(x, axis=axis, keepdims=True)

    axis = _relative_axis_to_absolute_axis(axis, x)
    shape = tuple(x.shape[axis] if i == axis else 1 for i, d in enumerate(x.shape))

    scale = jnp.array([1.0])
    if create_scale:
        params["scale"] == template.array(shape, initializer)
        scale = params["scale"]
    scale = jnp.broadcast_to(scale, x.shape)

    offset = jnp.array([0.0])
    if create_offset:
        params["offset"] == template.array(shape, initializer)
        offset = params["offset"]
    offset = jnp.broadcast_to(offset, x.shape)

    inversion = scale * rsqrt(variance + eps)
    normalized = inversion * (x - mean) + offset

    return normalized


def _relative_axis_to_absolute_axis(axis: int, reference_array: Array) -> int:
    return tuple(range(reference_array.ndim))[axis]

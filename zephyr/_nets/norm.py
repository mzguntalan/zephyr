from typing import Optional

from jax import nn
from jax import numpy as jnp
from jax.lax import rsqrt
from jaxtyping import Array
from jaxtyping import PyTree

from zephyr._nets.linear import branch_linear
from zephyr._nets.linear import linear
from zephyr.building import initializers
from zephyr.building import template
from zephyr.building.initializers import Initializer
from zephyr.building.template import validate
from zephyr.functools.partial import hole_aware
from zephyr.masking import apply_attention_mask


@hole_aware
def layer_norm(
    params: PyTree,
    x: Array,
    axis: int,
    create_scale: bool = True,
    create_offset: bool = True,
    eps: float = 1e-16,
    scale_initializer: Initializer = initializers.ones,
    offset_initializer: Initializer = initializers.zeros,
    activation=lambda x: x,
) -> Array:
    mean = jnp.mean(x, axis=axis, keepdims=True)
    variance = jnp.var(x, axis=axis, keepdims=True)

    axis = _relative_axis_to_absolute_axis(axis, x)
    shape = tuple(x.shape[axis] if i == axis else 1 for i, d in enumerate(x.shape))

    scale = jnp.array([1.0])
    if create_scale:
        validate(params["scale"], shape, scale_initializer)
        scale = params["scale"]
    scale = jnp.broadcast_to(scale, x.shape)

    offset = jnp.array([0.0])
    if create_offset:
        validate(params["offset"], shape, offset_initializer)
        offset = params["offset"]
    offset = jnp.broadcast_to(offset, x.shape)

    inversion = scale * rsqrt(variance + eps)
    normalized = inversion * (x - mean) + offset

    normalized = activation(normalized)

    return normalized


def _relative_axis_to_absolute_axis(axis: int, reference_array: Array) -> int:
    return tuple(range(reference_array.ndim))[axis]

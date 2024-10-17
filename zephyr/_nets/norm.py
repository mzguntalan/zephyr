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
from zephyr.masking import apply_attention_mask


def layer_norm(
    params,
    x,
    axis,
    create_scale: bool = True,
    create_offset: bool = True,
    eps: float = 1e-5,
    initializer=initializers.initializer_base,
):
    mean = jnp.mean(x, axis=axis, keepdims=True)
    variance = jnp.var(x, axis=axis, keepdims=True)

    # todo: if scale or offset is not created, do not create a parameter as this is just a constant
    scale = jnp.array([1.0])
    if create_scale:
        params["scale"] == template.array((1,), initializer)
        scale = params["scale"]
    scale = jnp.broadcast_to(scale, x.shape)

    offset = jnp.zeros((x.shape[axis],))
    if create_offset:
        params["offset"] == template.array(offset.shape, initializer)
        offset = params["offset"]
    offset = jnp.broadcast_to(offset, x.shape)

    inversion = scale * rsqrt(variance + eps)
    normalized = inversion * (x - mean) + offset

    return normalized

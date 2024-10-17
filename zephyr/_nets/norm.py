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
    params: PyTree,
    x: Array,
    axis: int,
    create_scale: bool = True,
    create_offset: bool = True,
    eps: float = 1e-5,
    initializer=initializers.initializer_base,
) -> Array:
    mean = jnp.mean(x, axis=axis, keepdims=True)
    variance = jnp.var(x, axis=axis, keepdims=True)

    # todo: if scale or offset is not created, do not create a parameter as this is just a constant
    scale = jnp.array([1.0])
    if create_scale:
        params["scale"] == template.array((1,), initializer)
        scale = params["scale"]
    # todo: known bug, jnp.broadcast_to uses jax._src.lax.lax.asarray
    # if instead, it used jnp.asarray, it would work
    # for now, as a workaround, use jnp.asarray on params
    scale = jnp.broadcast_to(jnp.asarray(scale), x.shape)

    offset = jnp.zeros((x.shape[axis],))
    if create_offset:
        params["offset"] == template.array(offset.shape, initializer)
        offset = params["offset"]
    offset = jnp.broadcast_to(jnp.asarray(offset), x.shape)

    inversion = scale * rsqrt(variance + eps)
    normalized = inversion * (x - mean) + offset

    return normalized

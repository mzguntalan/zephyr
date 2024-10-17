from jax import nn
from jax import numpy as jnp
from jax import random
from jaxtyping import Array
from jaxtyping import PyTree
from pytest import mark

from zephyr._nets.norm import layer_norm
from zephyr.building.tracing import trace
from zephyr.project_typing import Shape


@mark.parametrize(
    ["x", "axis", "expected_shape_after_apply"],
    [
        (jnp.ones([8]), 0, (8,)),
        (jnp.ones([8, 16]), 1, (8, 16)),
    ],
)
def test_layer_norm_shape(x, axis, expected_shape_after_apply: Shape) -> None:
    params = trace(layer_norm, random.PRNGKey(0), x, axis)
    z = layer_norm(params, x, axis)
    assert z.shape == expected_shape_after_apply

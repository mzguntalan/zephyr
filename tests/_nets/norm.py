import numpy as np
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
        (jnp.ones([8, 16, 32]), 1, (8, 16, 32)),
        (jnp.ones([8, 16, 32]), -1, (8, 16, 32)),
        (jnp.ones([8, 16, 32]), -2, (8, 16, 32)),
    ],
)
def test_layer_norm_shape(
    x: Array, axis: int, expected_shape_after_apply: Shape
) -> None:
    params = trace(layer_norm, random.PRNGKey(0), x, axis)
    z = layer_norm(params, x, axis)
    assert z.shape == expected_shape_after_apply


@mark.parametrize(
    ["x", "axis", "expected_shape_after_apply"],
    [
        (jnp.ones([8]), 0, (8,)),
        (jnp.ones([8, 16]), 1, (8, 16)),
        (jnp.ones([8, 16, 32]), 1, (8, 16, 32)),
        (jnp.ones([8, 16, 32]), -1, (8, 16, 32)),
        (jnp.ones([8, 16, 32]), -2, (8, 16, 32)),
    ],
)
def test_layer_norm_shape_no_offset_no_bool(
    x: Array, axis: int, expected_shape_after_apply: Shape
) -> None:
    x_that_is_not_of_the_same_values = random.normal(random.PRNGKey(0), x.shape)
    x = x_that_is_not_of_the_same_values
    params = trace(layer_norm, random.PRNGKey(0), x, axis, False, False)
    z = layer_norm(params, x, axis, False, False)

    has_params = lambda p: len(p) >= 1

    assert z.shape == expected_shape_after_apply
    assert not has_params(
        params
    )  # because offset and scale should not be trained and is therefore a constant in their formulae

    mean = jnp.mean(z, axis)
    standard_deviation = jnp.std(z, axis)

    assert jnp.allclose(mean, jnp.zeros_like(mean), 1e-5, 1e-5)
    assert jnp.allclose(standard_deviation, jnp.ones_like(standard_deviation))

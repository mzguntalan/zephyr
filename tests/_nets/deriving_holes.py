from jax import nn
from jax import numpy as jnp
from jax import random
from jaxtyping import Array
from jaxtyping import PyTree
from pytest import mark

from zephyr._nets.mlp import branch_linear
from zephyr._nets.mlp import linear
from zephyr._nets.mlp import linear_like
from zephyr._nets.mlp import mlp
from zephyr.building.tracing import trace
from zephyr.functools.partial import derivable_hole as __
from zephyr.functools.partial import placeholder_hole as _
from zephyr.project_typing import Shape


@mark.parametrize(
    ["x", "out_dims", "expected_shape_after_apply"],
    [
        # (jnp.ones([8]), [256], (256,)),
        (jnp.ones([16, 32]), [256, 512], (16, 512)),
        (jnp.ones([4, 16, 8]), [16, 4, 2], (4, 16, 2)),
    ],
)
def test_mlp_derives_holes(
    x: Array, out_dims: list[int], expected_shape_after_apply: Shape
) -> None:

    params = trace(mlp(_, _, out_dims), random.PRNGKey(0), x)
    z = mlp(params, x, out_dims)
    assert z.shape == expected_shape_after_apply

    z_by_derived_hyperparameters = mlp(params, x, __)
    assert z.shape == z_by_derived_hyperparameters.shape
    assert jnp.all(jnp.allclose(z, z_by_derived_hyperparameters))

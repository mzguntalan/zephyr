from jax import nn
from jax import numpy as jnp
from jax import random
from jaxtyping import Array
from jaxtyping import PyTree
from pytest import mark

from zephyr._nets.mlp import branch_linear
from zephyr._nets.mlp import linear
from zephyr._nets.mlp import mlp
from zephyr.building.tracing import trace
from zephyr.project_typing import Shape


@mark.parametrize(
    ["x", "target_out", "expected_shape_after_apply"],
    [
        (jnp.ones([8]), 256, (256,)),
        (jnp.ones([8, 16]), 256, (8, 256)),
        (jnp.ones([8, 16, 32]), 4, (8, 16, 4)),
    ],
)
def test_linear(x: Array, target_out: int, expected_shape_after_apply: Shape) -> None:

    params = trace(linear, random.PRNGKey(0), x, target_out)
    z = linear(params, x, target_out)
    assert z.shape == expected_shape_after_apply


@mark.parametrize(
    ["x", "num_branches", "expected_shape_after_apply"],
    [
        (jnp.ones([8]), 4, (4, 8)),
        (jnp.ones([16, 32]), 8, (16, 8, 32)),
        (jnp.ones([4, 16, 8]), 32, (4, 16, 32, 8)),
    ],
)
def test_branch_linear(
    x: Array, num_branches: int, expected_shape_after_apply: Shape
) -> None:
    params = trace(branch_linear, random.PRNGKey(0), x, num_branches)
    z = branch_linear(params, x, num_branches)
    assert z.shape == expected_shape_after_apply


@mark.parametrize(
    ["x", "out_dims", "expected_shape_after_apply"],
    [
        (jnp.ones([8]), [256], (256,)),
        (jnp.ones([16, 32]), [256, 512], (16, 512)),
        (jnp.ones([4, 16, 8]), [16, 4, 2], (4, 16, 2)),
    ],
)
def test_mlp(x: Array, out_dims: list[int], expected_shape_after_apply: Shape) -> None:
    params = trace(mlp, random.PRNGKey(0), x, out_dims)
    z = mlp(params, x, out_dims)
    assert z.shape == expected_shape_after_apply

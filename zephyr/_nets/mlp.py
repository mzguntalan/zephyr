from typing import Callable

from jax import nn
from jax import numpy as jnp
from jaxtyping import Array
from jaxtyping import PyTree

from zephyr.building import initializers
from zephyr.building import template


def linear(
    params: PyTree,
    x: Array,
    target_out: int,
    with_bias: bool = True,
    initializer_weight: initializers.Initializer = initializers.initializer_base,
    initializer_bias: initializers.Initializer = initializers.zeros,
) -> Array:
    params["weights"] == template.array((target_out, x.shape[-1]), initializer_weight)
    z = jnp.expand_dims(x, axis=-1)
    z = params["weights"] @ z
    z = jnp.squeeze(z, axis=-1)

    if with_bias:
        params["bias"] == template.array((target_out,), initializer_bias)
        z = params["bias"] + z

    return z


def branch_linear(
    params: PyTree,
    x: Array,
    num_branches: int,
    with_bias: bool = True,
    initializer: initializers.Initializer = initializers.initializer_base,
) -> Array:
    """Branches the last dimension of `x` with each branch having the same dimension as the last dimension of `x`

    Example:
        if x.shape == (..., e) then after this function

            z = branch_linear (...x, num_branches...)
            z.shape == (..., num_branches, x.shape[-1])

    """
    z = linear(params, x, x.shape[-1] * num_branches)
    z = jnp.reshape(z, z.shape[:-1] + (num_branches, x.shape[-1]))

    return z


def mlp(
    params: PyTree,
    x: Array,
    out_dims: list[int],
    activation: Callable[[Array], Array] = nn.relu,
    activate_final: bool = False,
    initializer: initializers.Initializer = initializers.initializer_base,
) -> Array:
    for i, target_out in enumerate(out_dims[:-1]):
        x = activation(linear(params[i], x, target_out, initializer_weight=initializer))

    if len(out_dims[:-1]) == 0:
        i = 0
    else:
        i += 1
    x = linear(params[i], x, out_dims[-1], initializer_weight=initializer)

    if activate_final:
        x = activation(x)

    return x


def linear_like(
    params: PyTree,
    array_to_be_projected_to_desired_shape: Array,
    reference_array_with_desired_last_dimension: Array,
    initializer: initializers.Initializer = initializers.initializer_base,
) -> Array:
    array_with_desired_shape = linear(
        params,
        array_to_be_projected_to_desired_shape,
        reference_array_with_desired_last_dimension.shape[-1],
    )

    return array_with_desired_shape

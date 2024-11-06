from typing import Callable

from jax import nn
from jax import numpy as jnp
from jaxtyping import Array
from jaxtyping import PyTree

from zephyr._nets.linear import linear
from zephyr.building import initializers
from zephyr.building import template
from zephyr.building.template import validate
from zephyr.functools.partial import flexible


@flexible
def mlp(
    params: PyTree,
    x: Array,
    out_dims: list[int],
    activation=nn.relu,
    activation_final=lambda x: x,
    use_bias: bool = True,
    weight_initializer: initializers.Initializer = initializers.initializer_base,
    bias_initializer: initializers.Initializer = initializers.initializer_base,
):
    validate(
        params,
        expression=lambda params: [params[i]["weights"].shape[-2] for i in params]
        == out_dims,
    )
    for i, target_out in enumerate(out_dims[:-1]):
        x = activation(
            linear(
                params[i], x, target_out, use_bias, weight_initializer, bias_initializer
            )
        )

    if len(out_dims[:-1]) == 0:
        i = 0
    else:
        i += 1
    x = linear(
        params[i], x, out_dims[i], use_bias, weight_initializer, bias_initializer
    )

    x = activation_final(x)

    return x

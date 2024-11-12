from jax import nn
from jax import numpy as jnp
from jaxtyping import Array
from jaxtyping import PyTree

from zephyr._nets.linear import branch_linear
from zephyr._nets.linear import linear
from zephyr._nets.linear import linear_like
from zephyr._nets.mlp import mlp
from zephyr._nets.norm import layer_norm
from zephyr.building import initializers
from zephyr.building.template import validate
from zephyr.functools.partial import flexible


def fmap(x: Array) -> Array:
    return nn.elu(x) + 1


@flexible
def linear_attention(queries: Array, keys: Array, values: Array) -> Array:
    q = fmap(queries)
    k = fmap(keys)

    keys_transposed = jnp.moveaxis(k, -2, -1)
    scores = q @ (keys_transposed @ values)
    normalizer = q @ (keys_transposed @ jnp.ones_like(values))

    answers = scores / (normalizer + 1e-17)
    return answers


@flexible
def multi_head_linear_attention(
    queries, keys, values, num_heads: int, activation=lambda x: x
) -> Array:
    new_shape = queries.shape[:-1] + (num_heads, -1)
    queries = jnp.reshape(queries, new_shape)
    keys = jnp.reshape(queries, new_shape)
    values = jnp.reshape(queries, new_shape)

    # queries, keys, values [..., s, h, h//e]
    #                       [...,-3,-2,-1]

    queries = jnp.moveaxis(queries, -2, -3)
    keys = jnp.moveaxis(keys, -2, -3)
    values = jnp.moveaxis(values, -2, -3)

    multi_head_answers = linear_attention(queries, keys, values)  # [..., h, s, e]

    multi_head_answers = jnp.moveaxis(multi_head_answers, -2, -3)  # [..., s , h, h//e]

    combined_heads = jnp.reshape(
        multi_head_answers, multi_head_answers.shape[:-2] + (-1,)
    )

    combined_heads = activation(combined_heads)

    return combined_heads


@flexible
def linear_transformer_block(
    params: PyTree,
    queries: Array,
    keys: Array,
    values: Array,
    num_heads: int,
    embed_dim: int,
    mlp_dim: int,
    with_bias: bool = True,
    weights_initializer: initializers.Initializer = initializers.initializer_base,
    bias_initializer: initializers.Initializer = initializers.initializer_base,
    activation=lambda x: x,
) -> Array:
    queries = linear(
        params["linear"]["queries"],
        queries,
        embed_dim,
        with_bias,
        weights_initializer,
        bias_initializer,
    )
    keys = linear(
        params["linear"]["keys"],
        keys,
        embed_dim,
        with_bias,
        weights_initializer,
        bias_initializer,
    )
    values = linear(
        params["linear"]["values"],
        values,
        embed_dim,
        with_bias,
        weights_initializer,
        bias_initializer,
    )

    # values don't have to be projected since its embedding dimension was the reference
    z = multi_head_linear_attention(
        queries,
        keys,
        values,
        num_heads,
    )

    z = layer_norm(params["layer_norm"][0], z + queries, -1, True, True)
    z = layer_norm(
        params["layer_norm"][1],
        z
        + mlp(
            params["mlp"],
            z,
            [mlp_dim, embed_dim],
            weight_initializer=weights_initializer,
            bias_initializer=bias_initializer,
        ),
        -1,
        True,
        True,
    )

    z = activation(z)

    return z

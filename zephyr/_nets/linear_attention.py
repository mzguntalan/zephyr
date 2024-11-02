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
from zephyr.functools.partial import deriving_holes
from zephyr.functools.partial import hole_aware


@hole_aware
@deriving_holes
def single_head_linear_attention(
    params: PyTree,
    queries: Array,
    keys: Array,
    values: Array,
    with_bias: bool = True,
    weights_initializer=initializers.initializer_base,
    bias_initializer=initializers.initializer_base,
) -> Array:
    # queries, keys, values [... s|p, k|v]
    queries = linear(
        params["linear_queries"],
        queries,
        queries.shape[-1],
        with_bias,
        weights_initializer,
        bias_initializer,
    )
    keys = linear(
        params["linear_keys"],
        keys,
        keys.shape[-1],
        with_bias,
        weights_initializer,
        bias_initializer,
    )
    values = linear(
        params["linear_values"],
        values,
        values.shape[-1],
        with_bias,
        weights_initializer,
        bias_initializer,
    )

    scores = queries @ (jnp.moveaxis(keys, -2, -1) @ values)
    normalizer = queries @ (jnp.moveaxis(keys, -2, -1) @ jnp.ones_like(values))

    answers = scores / (normalizer + 1e-17)

    return answers


@hole_aware
@deriving_holes
def multi_head_linear_attention(
    params: PyTree,
    queries,
    keys,
    values,
    num_heads: int,
    with_bias: bool = True,
    weights_initializer=initializers.initializer_base,
    bias_initializer=initializers.initializer_base,
) -> Array:
    validate(
        params,
        expression=lambda params: params["branch_linear_queries"]["weights"].shape[-2]
        // queries.shape[-1]
        == num_heads,
    )

    queries = branch_linear(
        params["branch_linear_queries"],
        queries,
        num_heads,
        with_bias,
        weights_initializer,
        bias_initializer,
    )
    keys = branch_linear(
        params["branch_linear_keys"],
        keys,
        num_heads,
        with_bias,
        weights_initializer,
        bias_initializer,
    )
    values = branch_linear(
        params["branch_linear_values"],
        values,
        num_heads,
        with_bias,
        weights_initializer,
        bias_initializer,
    )

    # queries, keys, values [..., s, h, e]
    #                       [...,-3,-2,-1]

    queries = jnp.moveaxis(queries, -2, -3)
    keys = jnp.moveaxis(keys, -2, -3)
    values = jnp.moveaxis(values, -2, -3)

    multi_head_answers = single_head_linear_attention(
        params["single_head_linear_attention"],
        queries,
        keys,
        values,
        with_bias,
        weights_initializer,
        bias_initializer,
    )  # [..., h, s, e]

    multi_head_answers = jnp.moveaxis(multi_head_answers, -2, -3)  # [..., s , h, e]

    combined_heads = jnp.reshape(
        multi_head_answers, multi_head_answers.shape[:-2] + (-1,)
    )

    combined_heads = linear(
        params["linear_combined_heads"],
        combined_heads,
        combined_heads.shape[-1] // num_heads,
        with_bias,
        weights_initializer,
        bias_initializer,
    )

    return combined_heads


@hole_aware
def linear_transformer_block(
    params: PyTree,
    queries: Array,
    keys: Array,
    values: Array,
    num_heads: int,
    mlp_dim: int,
    with_bias: bool = True,
    weights_initializer: initializers.Initializer = initializers.initializer_base,
    bias_initializer: initializers.Initializer = initializers.initializer_base,
) -> Array:
    queries = linear_like(
        params["initial_projection_queries"],
        queries,
        values,
        with_bias,
        weights_initializer,
        bias_initializer,
    )
    keys = linear_like(
        params["initial_projection_keys"],
        keys,
        values,
        with_bias,
        weights_initializer,
        bias_initializer,
    )
    # values don't have to be projected since its embedding dimension was the reference
    z = multi_head_linear_attention(
        params["multi_head_linear_attention"],
        queries,
        keys,
        values,
        num_heads,
        with_bias,
        weights_initializer,
        bias_initializer,
    )

    z = layer_norm(
        params["layer_norm"], z + queries, -1, True, True, initializer=initializer
    )
    z = z + mlp(
        params["mlp"],
        z,
        [mlp_dim, queries.shape[-1]],
        with_bias,
        weights_initializer,
        bias_initializer,
    )

    return z

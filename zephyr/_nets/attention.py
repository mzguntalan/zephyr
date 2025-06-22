from typing import Optional

import numpy as np
from jax import nn
from jax import numpy as jnp
from jaxtyping import Array
from jaxtyping import PyTree

from zephyr._nets.linear import branch_linear
from zephyr._nets.linear import linear
from zephyr.building import initializers
from zephyr.building.template import validate
from zephyr.functools.partial import flexible
from zephyr.masking import apply_attention_mask


@flexible
def single_head_attention(
    params: PyTree,
    queries: Array,
    keys: Array,
    values: Array,
    masks: Optional[Array] = None,
    with_bias: bool = True,
    weights_initializer: initializers.Initializer = initializers.initializer_base,
    bias_initializer: initializers.Initializer = initializers.initializer_base,
    activation=lambda x: x,
) -> Array:
    keys = linear(
        params["linear_keys"],
        keys,
        keys.shape[-1],
        with_bias,
        weights_initializer,
        bias_initializer,
    )
    queries = linear(
        params["linear_queries"],
        queries,
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

    # keys [... s k]
    # queries [... p k]
    # values [... s v]
    # target [... p v]

    scores = queries @ jnp.moveaxis(keys, -1, -2) / np.sqrt(keys.shape[-1])
    if masks is not None:
        scores = apply_attention_mask(scores, masks)
    attention_map = nn.softmax(scores, axis=-1)

    answers = attention_map @ values
    answers = activation(answers)
    return answers


@flexible
def multi_head_attention(
    params: PyTree,
    queries: Array,
    keys: Array,
    values: Array,
    num_heads: int,
    masks: Optional[Array] = None,
    with_bias: bool = True,
    weights_initializer: initializers.Initializer = initializers.initializer_base,
    bias_initializer: initializers.Initializer = initializers.initializer_base,
    activation=lambda x: x,
) -> Array:
    new_shape_queries = queries.shape[:-1] + (num_heads, -1)
    new_shape_keys = keys.shape[:-1] + (num_heads, -1)
    new_shape_values = values.shape[:-1] + (num_heads, -1)
    queries = jnp.reshape(queries, new_shape_queries)
    keys = jnp.reshape(keys, new_shape_keys)
    values = jnp.reshape(values, new_shape_values)

    # queries, keys, values [..., s, h, e]
    #                       [...,-3,-2,-1]

    queries = jnp.moveaxis(queries, -2, -3)
    keys = jnp.moveaxis(keys, -2, -3)
    values = jnp.moveaxis(values, -2, -3)

    multi_head_answers = single_head_attention(
        params["single_head_attention"],
        queries,
        keys,
        values,
        masks,
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
        combined_heads.shape[-1],
        with_bias,
        weights_initializer,
        bias_initializer,
    )

    combined_heads = activation(combined_heads)

    return combined_heads


@flexible
def multi_head_self_attention(
    params: PyTree,
    x: Array,
    num_heads: int,
    masks: Optional[Array] = None,
    with_bias: bool = True,
    weights_initializer: initializers.Initializer = initializers.initializer_base,
    bias_initializer: initializers.Initializer = initializers.initializer_base,
    activation=lambda x: x,
) -> Array:
    return multi_head_attention(
        params,
        x,
        x,
        x,
        num_heads,
        masks,
        with_bias,
        weights_initializer,
        bias_initializer,
        activation,
    )


# todo: refine this to align with others

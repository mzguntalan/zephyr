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
from zephyr.functools.partial import deriving_holes
from zephyr.functools.partial import hole_aware
from zephyr.masking import apply_attention_mask


@hole_aware
@deriving_holes
def single_head_attention(
    params: PyTree,
    queries: Array,
    keys: Array,
    values: Array,
    masks: Optional[Array] = None,
    with_bias: bool = True,
    weights_initializer: initializers.Initializer = initializers.initializer_base,
    bias_initializer: initializers.Initializer = initializers.initializer_base,
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
    if masks:
        scores = apply_attention_mask(scores, masks)
    attention_map = nn.softmax(scores, axis=-1)

    answers = attention_map @ values
    return answers


@hole_aware
@deriving_holes
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

    multi_head_answers = single_head_attention(
        params["single_head_attention"],
        queries,
        keys,
        values,
        masks,
        with_bias,
        initializer,
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

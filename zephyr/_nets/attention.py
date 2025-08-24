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
    queries: Array,  # [batch, seq_q, model_dim]
    keys: Array,  # [batch, seq_k, model_dim]
    values: Array,  # [batch, seq_k, model_dim]
    num_heads: int,
    masks: Optional[Array] = None,
    with_bias: bool = True,
    weights_initializer: initializers.Initializer = initializers.initializer_base,
    bias_initializer: initializers.Initializer = initializers.initializer_base,
) -> Array:
    model_dim = queries.shape[-1]
    head_dim = model_dim // num_heads
    assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"

    Q = linear(
        params["q_proj"],
        queries,
        num_heads * head_dim,
        with_bias,
        weights_initializer,
        bias_initializer,
    )
    K = linear(
        params["k_proj"],
        keys,
        num_heads * head_dim,
        with_bias,
        weights_initializer,
        bias_initializer,
    )
    V = linear(
        params["v_proj"],
        values,
        num_heads * head_dim,
        with_bias,
        weights_initializer,
        bias_initializer,
    )

    def split_heads(x):
        return jnp.reshape(
            x, x.shape[:-1] + (num_heads, head_dim)
        )  # [..., heads, head_dim]

    Q = jnp.moveaxis(split_heads(Q), -2, 1)  # [batch, heads, seq_q, head_dim]
    K = jnp.moveaxis(split_heads(K), -2, 1)  # [batch, heads, seq_k, head_dim]
    V = jnp.moveaxis(split_heads(V), -2, 1)  # [batch, heads, seq_k, head_dim]

    scores = jnp.matmul(Q, jnp.swapaxes(K, -1, -2)) / np.sqrt(
        head_dim
    )  # [batch, heads, seq_q, seq_k]

    if masks is not None:
        # Mask should broadcast to [batch, heads, seq_q, seq_k]
        scores = apply_attention_mask(scores, masks)

    attn_weights = nn.softmax(scores, axis=-1)  # [batch, heads, seq_q, seq_k]
    attn_output = jnp.matmul(attn_weights, V)  # [batch, heads, seq_q, head_dim]

    attn_output = jnp.moveaxis(attn_output, 1, -2)  # [batch, seq_q, heads, head_dim]
    attn_output = jnp.reshape(attn_output, attn_output.shape[:-2] + (model_dim,))

    output: Array = linear(
        params["out_proj"],
        attn_output,
        model_dim,
        with_bias,
        weights_initializer,
        bias_initializer,
    )

    return output


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

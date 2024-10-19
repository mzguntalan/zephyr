from jax import nn
from jax import numpy as jnp
from jax import random
from jaxtyping import Array
from jaxtyping import PyTree
from pytest import mark

from zephyr._nets.linear_attention import linear_transformer_block
from zephyr._nets.linear_attention import multi_head_linear_attention
from zephyr._nets.linear_attention import single_head_linear_attention
from zephyr.building.tracing import trace


@mark.parametrize(
    [
        "batch_dim",
        "query_sequence_length",
        "value_sequence_length",
        "key_dim",
        "value_dim",
    ],
    [
        (8, 16, 32, 64, 128),
        (64, 8, 128, 4, 2),
        (16, 128, 8, 256, 16),
    ],
)
def test_single_head_attention(
    batch_dim: int,
    query_sequence_length: int,
    value_sequence_length: int,
    key_dim: int,
    value_dim: int,
) -> None:
    queries = jnp.ones([batch_dim, query_sequence_length, key_dim])
    keys = jnp.ones([batch_dim, value_sequence_length, key_dim])
    values = jnp.ones([batch_dim, value_sequence_length, value_dim])

    params = trace(
        single_head_linear_attention, random.PRNGKey(0), queries, keys, values
    )
    answers = single_head_linear_attention(params, queries, keys, values)

    assert answers.shape == (batch_dim, query_sequence_length, value_dim)


@mark.parametrize(
    [
        "num_heads",
        "batch_dim",
        "query_sequence_length",
        "value_sequence_length",
        "key_dim",
        "value_dim",
    ],
    [
        (8, 8, 16, 32, 64, 128),
        (4, 64, 8, 128, 4, 4),
        (4, 64, 8, 128, 4, 8),
        (4, 64, 8, 128, 8, 4),
        (8, 16, 128, 8, 256, 16),
    ],
)
def test_multi_head_attention(
    num_heads: int,
    batch_dim: int,
    query_sequence_length: int,
    value_sequence_length: int,
    key_dim: int,
    value_dim: int,
) -> None:
    queries = jnp.ones([batch_dim, query_sequence_length, key_dim])
    keys = jnp.ones([batch_dim, value_sequence_length, key_dim])
    values = jnp.ones([batch_dim, value_sequence_length, value_dim])

    params = trace(
        multi_head_linear_attention,
        random.PRNGKey(0),
        queries,
        keys,
        values,
        num_heads,
    )
    answers = multi_head_linear_attention(params, queries, keys, values, num_heads)

    assert answers.shape == (batch_dim, query_sequence_length, value_dim)


@mark.parametrize(
    [
        "num_heads",
        "batch_dim",
        "query_sequence_length",
        "value_sequence_length",
        "key_dim",
        "value_dim",
        "mlp_dim",
    ],
    [
        (8, 8, 16, 32, 64, 128, 256),
        (4, 64, 8, 128, 4, 4, 16),
        (4, 64, 8, 128, 4, 8, 64),
        (4, 64, 8, 128, 8, 4, 12),
        (8, 16, 128, 8, 256, 16, 4),
    ],
)
def test_linear_transormer_block(
    num_heads: int,
    batch_dim: int,
    query_sequence_length: int,
    value_sequence_length: int,
    key_dim: int,
    value_dim: int,
    mlp_dim: int,
) -> None:
    queries = jnp.ones([batch_dim, query_sequence_length, key_dim])
    keys = jnp.ones([batch_dim, value_sequence_length, key_dim])
    values = jnp.ones([batch_dim, value_sequence_length, value_dim])

    params = trace(
        linear_transformer_block,
        random.PRNGKey(0),
        queries,
        keys,
        values,
        num_heads,
        mlp_dim,
    )
    answers = linear_transformer_block(
        params, queries, keys, values, num_heads, mlp_dim
    )

    assert answers.shape == (batch_dim, query_sequence_length, value_dim)

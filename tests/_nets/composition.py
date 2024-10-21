from jax import nn
from jax import numpy as jnp
from jax import random

from zephyr._nets.attention import multi_head_attention
from zephyr._nets.attention import single_head_attention
from zephyr._nets.composition import sequential
from zephyr._nets.embed import token_embed
from zephyr._nets.mlp import branch_linear
from zephyr._nets.mlp import mlp
from zephyr._nets.norm import layer_norm
from zephyr.building.tracing import trace
from zephyr.functools.partial import hole as _


def test_sequential():
    batch_dim = 8
    id_min = 0
    id_max = 256
    sequence_length = 10
    embed_dim = 128

    dummy_input = random.randint(
        random.PRNGKey(0), (batch_dim, sequence_length), id_min, id_max
    )

    seq = sequential(
        _,
        _,
        [
            token_embed(_, _, id_max, embed_dim),
            mlp(_, _, [64, 64, 64]),
            layer_norm(_, _, -1),
            lambda params, x: nn.relu(x),
            mlp(_, _, [embed_dim]),
        ],
    )

    params = trace(seq, random.PRNGKey(0), dummy_input)
    outputs = seq(params, dummy_input)

    assert outputs.shape == (batch_dim, sequence_length, embed_dim)

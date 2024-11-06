from typing import Optional

from jax import numpy as jnp
from jaxtyping import Array
from jaxtyping import PyTree

from zephyr.building import template
from zephyr.building._template.validation import validate
from zephyr.building.initializers import Initializer
from zephyr.building.initializers import initializer_base
from zephyr.functools.partial import deriving_holes
from zephyr.functools.partial import hole_aware


@hole_aware
@deriving_holes
def token_embed(
    params: PyTree,
    x_token_ids: Array,
    vocab_size: int,
    embed_dim: int,
    initial_embedding_matrix: Optional[Array] = None,
    initializer: Initializer = initializer_base,
    activation=lambda x: x,
) -> Array:
    if initial_embedding_matrix is not None:
        validate(
            params["token_embeddings"],
            (vocab_size, embed_dim),
            initializer=lambda key, shape: initial_embedding_matrix,
        )
        validate(
            params,
            expression=lambda params: params["token_embeddings"].shape
            == (vocab_size, embed_dim),
        )
    else:
        validate(params["token_embeddings"], (vocab_size, embed_dim), initializer)
    z = jnp.asarray(params["token_embeddings"])[(x_token_ids,)]
    z = activation(z)
    return z

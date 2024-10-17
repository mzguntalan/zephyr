from typing import Optional

from jax import numpy as jnp
from jaxtyping import Array
from jaxtyping import PyTree

from zephyr.building import template
from zephyr.building.initializers import Initializer
from zephyr.building.initializers import initializer_base


def token_embed(
    params: PyTree,
    x_token_ids: Array,
    vocab_size: int,
    embed_dim: int,
    embedding_matrix: Optional[Array] = None,
    initializer: Initializer = initializer_base,
) -> Array:
    if embedding_matrix is not None:
        params["token_embeddings"] == template.array(
            (vocab_size, embed_dim), initializer=lambda key, shape: embedding_matrix
        )
    else:
        params["token_embeddings"] == template.array(
            (vocab_size, embed_dim), initializer
        )
    return jnp.asarray(params["token_embeddings"])[(x_token_ids,)]

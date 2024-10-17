import pytest
from jax import nn
from jax import numpy as jnp
from jaxtyping import Array
from pytest import mark

from zephyr.masking import apply_attention_mask
from zephyr.masking import apply_mask


@mark.parametrize(
    ["x", "masks", "expected"],
    [
        (jnp.array([1, 2, 3, 4]), jnp.array([0, 1, 0, 1]), jnp.array([0, 2, 0, 4])),
        (
            jnp.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
            jnp.array([[0, 1], [0, 0], [1, 1], [1, 0]]),
            jnp.array([[0, 2], [0, 0], [5, 6], [7, 0]]),
        ),
    ],
)
def test_apply_mask_zeros(x: Array, masks: Array, expected: Array) -> None:
    masked_x = apply_mask(x, masks)
    assert jnp.all(jnp.equal(masked_x, expected))


@mark.parametrize(
    ["x", "masks", "expected_binary_attention"],
    [
        (jnp.array([1, 2, 3, 4]), jnp.array([0, 1, 1, 0]), jnp.array([0, 1, 1, 0])),
        (
            jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            jnp.array([[0, 0, 1], [1, 1, 1], [1, 1, 0]]),
            jnp.array([[0, 0, 1], [1, 1, 1], [1, 1, 0]]),
        ),
    ],
)
def test_apply_attention_mask(
    x: Array, masks: Array, expected_binary_attention: Array
) -> None:
    # binary attention is the same as mask if all values of x >= 1
    attention = nn.softmax(apply_attention_mask(x, masks))
    binary_attetion = 1 * (attention >= 1e-32)
    assert jnp.all(jnp.equal(binary_attetion, expected_binary_attention))

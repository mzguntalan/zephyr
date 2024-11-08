from typing import Sequence
from typing import Union

import numpy as np
from jax import lax
from jax import numpy as jnp
from jax.lax import reduce_window
from jaxtyping import Array

from zephyr.functools.partial import flexible
from zephyr.project_typing import PaddingPreset
from zephyr.project_typing import ShapeExpression


@flexible
def max_pool(
    _params,
    x: Array,
    window_shape: ShapeExpression,
    stride: ShapeExpression = 1,
    padding: PaddingPreset = "SAME",
    channel_axis: int = -1,
    activation=lambda x: x,
) -> Array:
    if channel_axis != -1 and len(x.shape) != channel_axis:
        raise NotImplementedError("only channel last implemented")

    window_shape, strides = map(
        lambda s: _to_shape_tuple(x, s, channel_axis), [window_shape, stride]
    )

    return reduce_window(x, -jnp.inf, lax.max, window_shape, strides, padding)


@flexible
def avg_pool(
    _params,
    x: Array,
    window_shape: ShapeExpression,
    stride: ShapeExpression = 1,
    padding: PaddingPreset = "SAME",
    channel_axis: int = -1,
    activation=lambda x: x,
) -> Array:
    if channel_axis != -1 and len(x.shape) != channel_axis:
        raise NotImplementedError("only channel last implemented")

    window_shape, strides = map(
        lambda s: _to_shape_tuple(x, s, channel_axis), [window_shape, stride]
    )

    sum_pool = reduce_window(x, 0.0, lax.add, window_shape, strides, padding)
    if padding == "VALID":
        return sum_pool / np.prod(window_shape)

    window_counter_shape = [
        (v if w != 1 else 1) for (v, w) in zip(x.shape, window_shape)
    ]
    window_counts = lax.reduce_window(
        jnp.ones(window_counter_shape, x.dtype),
        0.0,
        lax.add,
        window_shape,
        strides,
        padding,
    )
    return sum_pool / window_counts


def _to_shape_tuple(
    x: Array,
    s: ShapeExpression,
    channel_axis: int = -1,
) -> Sequence[int]:
    if type(s) is int:
        if channel_axis < 0:
            return (1,) + tuple(
                s if d != channel_axis else 1 for d in range(1, len(x.shape))
            )
    elif len(s) < len(x.shape):
        return (1,) * (len(x.shape) - len(s)) + tuple(s)
    if len(x.shape) == len(s):
        return tuple(s)
    raise ValueError(f"Did not understand the inputs")

"""Affine Transformations"""

from typing import Callable
from typing import Literal
from typing import Sequence
from typing import Union

import numpy as np
from jax import nn
from jax import numpy as jnp
from jax.lax import conv_general_dilated as _conv_general_dilated
from jax.lax import conv_transpose as _conv_transpose
from jax.lax import ConvDimensionNumbers
from jaxtyping import Array
from jaxtyping import PyTree

from zephyr.building import initializers
from zephyr.building import template
from zephyr.building.template import validate
from zephyr.functools.partial import flexible
from zephyr.masking import apply_mask
from zephyr.project_typing import PaddingPreset
from zephyr.project_typing import Shape
from zephyr.project_typing import ShapeExpression


@flexible
def linear(
    params: PyTree,
    x: Array,
    out_dim: int,
    use_bias: bool = True,
    weights_initializer: initializers.Initializer = initializers.initializer_base,
    bias_initializer: initializers.Initializer = initializers.initializer_base,
    activation=lambda x: x,
) -> Array:
    validate(params["weights"], (x.shape[-1], out_dim), weights_initializer)
    z = x @ params["weights"]
    if use_bias:
        validate(params["bias"], (out_dim,), bias_initializer)
        z = z + params["bias"]
    z = activation(z)
    return z


@flexible
def branch_linear(
    params: PyTree,
    x: Array,
    num_branches: int,
    with_bias: bool = True,
    weights_initializer: initializers.Initializer = initializers.initializer_base,
    bias_initializer: initializers.Initializer = initializers.initializer_base,
    activation=lambda x: x,
) -> Array:
    """Branches the last dimension of `x` with each branch having the same dimension as the last dimension of `x`

    Example:
        if x.shape == (..., e) then after this function

            z = branch_linear (...x, num_branches...)
            z.shape == (..., num_branches, x.shape[-1])

    """
    validate(
        params,
        expression=lambda params: params["weights"].shape[-1] // x.shape[-1]
        == num_branches,
    )
    z = linear(
        params,
        x,
        x.shape[-1] * num_branches,
        with_bias,
        weights_initializer,
        bias_initializer,
        activation,
    )
    z = jnp.reshape(z, z.shape[:-1] + (num_branches, x.shape[-1]))

    return z


@flexible
def linear_like(
    params: PyTree,
    array_to_be_projected_to_desired_shape: Array,
    reference_array_with_desired_last_dimension: Array,
    use_bias: bool = True,
    weights_initializer: initializers.Initializer = initializers.initializer_base,
    bias_initializer: initializers.Initializer = initializers.initializer_base,
    activation=lambda x: x,
) -> Array:
    validate(
        params,
        expression=lambda params: reference_array_with_desired_last_dimension
        == jnp.ones(
            # array_to_be_projected_to_desired_shape.shape[:-1]
            # +
            (params["weights"].shape[-2],),
        ),
    )
    array_with_desired_shape = linear(
        params,
        array_to_be_projected_to_desired_shape,
        reference_array_with_desired_last_dimension.shape[-1],
        use_bias,
        weights_initializer,
        bias_initializer,
        activation,
    )

    return array_with_desired_shape


@flexible
def conv_general(
    params: PyTree,
    x: Array,
    num_spatial_axes: int,
    out_channels: int,
    kernel_shape: ShapeExpression,
    stride: ShapeExpression = 1,
    rate: ShapeExpression = 1,
    padding: Union[PaddingPreset, Sequence[tuple[int, int]]] = "SAME",
    dilation: ShapeExpression = 1,
    with_bias: bool = True,
    mask: None | Array = None,
    weights_initializer: initializers.Initializer = initializers.initializer_base,
    bias_initializer: initializers.Initializer = initializers.initializer_base,
    feature_group_count: int = 1,
    data_format: Literal["CHANNELS_LAST"] = "CHANNELS_LAST",
    activation=lambda x: x,
) -> Array:
    if data_format != "CHANNELS_LAST":
        raise NotImplementedError(f"`data_format` not supported")
    kernel_shape = _to_shape_tuple(kernel_shape, num_spatial_axes)
    padding = padding if type(padding) is str else _to_padding_tuple(padding)

    stride, kernel_dilation, lhs_dilation = map(
        lambda x: _to_shape_tuple(x, num_spatial_axes), [stride, rate, dilation]
    )

    dimension_numbers = _to_dimenstion_numbers(
        num_spatial_axes, data_format == "CHANNELS_LAST", False
    )
    num_axes = len(x.shape)

    uses_batches = num_axes != num_spatial_axes + 1
    if not uses_batches:
        x = jnp.expand_dims(x, axis=0)

    if x.shape[-1] % feature_group_count != 0:
        raise ValueError(
            f"Inputs channels: {x.shape[-1]} should be divisible by `feature_group_count`:{feature_group_count}"
        )

    weights_shape = kernel_shape + (x.shape[-1] // feature_group_count, out_channels)
    validate(params["weights"], weights_shape, weights_initializer)

    if mask is not None:
        params["weights"] = apply_mask(params["weights"], mask)

    z = _conv_general_dilated(
        x,
        params["weights"],
        window_strides=stride,
        padding=padding,
        lhs_dilation=lhs_dilation,
        rhs_dilation=kernel_dilation,
        dimension_numbers=dimension_numbers,
        feature_group_count=feature_group_count,
    )

    if with_bias:
        validate(params["bias"], (out_channels,), bias_initializer)
        bias = jnp.broadcast_to(params["bias"], z.shape)
        z = z + bias

    if not uses_batches:
        z = jnp.squeeze(z, 0)

    z = activation(z)

    return z


@flexible
def conv_1d(
    params: PyTree,
    x: Array,
    out_channels: int,
    kernel_shape: ShapeExpression,
    stride: ShapeExpression = 1,
    rate: ShapeExpression = 1,
    padding: Union[PaddingPreset, Sequence[tuple[int, int]]] = "SAME",
    dilation: ShapeExpression = 1,
    with_bias: bool = True,
    mask: None | Array = None,
    weights_initializer: initializers.Initializer = initializers.initializer_base,
    bias_initializer: initializers.Initializer = initializers.initializer_base,
    feature_group_count: int = 1,
    data_format: Literal["CHANNELS_LAST"] = "CHANNELS_LAST",
    activation=lambda x: x,
) -> Array:
    return conv_general(
        params,
        x,
        1,
        out_channels,
        kernel_shape,
        stride,
        rate,
        padding,
        dilation,
        with_bias,
        mask,
        weights_initializer,
        bias_initializer,
        feature_group_count,
        data_format,
        activation,
    )


@flexible
def conv_2d(
    params: PyTree,
    x: Array,
    out_channels: int,
    kernel_shape: ShapeExpression,
    stride: ShapeExpression = 1,
    rate: ShapeExpression = 1,
    padding: Union[PaddingPreset, Sequence[tuple[int, int]]] = "SAME",
    dilation: ShapeExpression = 1,
    with_bias: bool = True,
    mask: None | Array = None,
    weights_initializer: initializers.Initializer = initializers.initializer_base,
    bias_initializer: initializers.Initializer = initializers.initializer_base,
    feature_group_count: int = 1,
    data_format: Literal["CHANNELS_LAST"] = "CHANNELS_LAST",
    activation=lambda x: x,
) -> Array:
    return conv_general(
        params,
        x,
        2,
        out_channels,
        kernel_shape,
        stride,
        rate,
        padding,
        dilation,
        with_bias,
        mask,
        weights_initializer,
        bias_initializer,
        feature_group_count,
        data_format,
        activation,
    )


@flexible
def conv_3d(
    params: PyTree,
    x: Array,
    out_channels: int,
    kernel_shape: ShapeExpression,
    stride: ShapeExpression = 1,
    rate: ShapeExpression = 1,
    padding: Union[PaddingPreset, Sequence[tuple[int, int]]] = "SAME",
    dilation: ShapeExpression = 1,
    with_bias: bool = True,
    mask: None | Array = None,
    weights_initializer: initializers.Initializer = initializers.initializer_base,
    bias_initializer: initializers.Initializer = initializers.initializer_base,
    feature_group_count: int = 1,
    data_format: Literal["CHANNELS_LAST"] = "CHANNELS_LAST",
    activation=lambda x: x,
) -> Array:
    return conv_general(
        params,
        x,
        3,
        out_channels,
        kernel_shape,
        stride,
        rate,
        padding,
        dilation,
        with_bias,
        mask,
        weights_initializer,
        bias_initializer,
        feature_group_count,
        data_format,
        activation,
    )


@flexible
def conv_transpose_general(
    params: PyTree,
    x: Array,
    num_spatial_axes: int,
    out_channels: int,
    kernel_shape: ShapeExpression,
    stride: ShapeExpression = 1,
    padding: PaddingPreset | Sequence[tuple[int, int]] = "SAME",
    with_bias: bool = True,
    mask: Array | None = None,
    weights_initializer: initializers.Initializer = initializers.initializer_base,
    bias_initializer: initializers.Initializer = initializers.initializer_base,
    feature_group_count: int = 1,
    data_format: Literal["CHANNELS_LAST"] = "CHANNELS_LAST",
    activation=lambda x: x,
) -> Array:
    if data_format != "CHANNELS_LAST":
        raise NotImplementedError(f"`data_format` not supported")
    kernel_shape = _to_shape_tuple(kernel_shape, num_spatial_axes)
    padding = (
        padding
        if type(padding) is str
        else _to_padding_tuple(padding, num_spatial_axes)
    )

    stride = _to_shape_tuple(stride, num_spatial_axes)

    dimension_numbers = _to_dimenstion_numbers(
        num_spatial_axes, data_format == "CHANNELS_LAST", False
    )
    num_axes = len(x.shape)

    uses_batches = num_axes != num_spatial_axes + 1
    if not uses_batches:
        x = jnp.expand_dims(x, axis=0)

    if x.shape[-1] % feature_group_count != 0:
        raise ValueError(
            f"Inputs channels: {x.shape[-1]} should be divisible by `feature_group_count`:{feature_group_count}"
        )

    weights_shape = kernel_shape + (x.shape[-1] // feature_group_count, out_channels)
    validate(params["weights"], weights_shape, weights_initializer)

    if mask is not None:
        params["weights"] = apply_mask(params["weights"], mask)

    z = _conv_transpose(
        x,
        params["weights"],
        strides=stride,
        padding=padding,
        dimension_numbers=dimension_numbers,
    )

    if with_bias:
        validate(params["bias"], (out_channels,), bias_initializer)
        bias = jnp.broadcast_to(params["bias"], z.shape)
        z = z + bias

    if not uses_batches:
        z = jnp.squeeze(z, 0)

    z = activation(z)

    return z


@flexible
def conv_transpose_1d(
    params: PyTree,
    x: Array,
    out_channels: int,
    kernel_shape: ShapeExpression,
    stride: ShapeExpression = 1,
    padding: PaddingPreset | Sequence[tuple[int, int]] = "SAME",
    with_bias: bool = True,
    mask: Array | None = None,
    weights_initializer: initializers.Initializer = initializers.initializer_base,
    bias_initializer: initializers.Initializer = initializers.initializer_base,
    feature_group_count: int = 1,
    data_format: Literal["CHANNELS_LAST"] = "CHANNELS_LAST",
    activation=lambda x: x,
) -> Array:
    return conv_transpose_general(
        params,
        x,
        1,
        out_channels,
        kernel_shape,
        stride,
        padding,
        with_bias,
        mask,
        weights_initializer,
        bias_initializer,
        feature_group_count,
        data_format,
        activation,
    )


@flexible
def conv_transpose_2d(
    params: PyTree,
    x: Array,
    out_channels: int,
    kernel_shape: ShapeExpression,
    stride: ShapeExpression = 1,
    padding: PaddingPreset | Sequence[tuple[int, int]] = "SAME",
    with_bias: bool = True,
    mask: Array | None = None,
    weights_initializer: initializers.Initializer = initializers.initializer_base,
    bias_initializer: initializers.Initializer = initializers.initializer_base,
    feature_group_count: int = 1,
    data_format: Literal["CHANNELS_LAST"] = "CHANNELS_LAST",
    activation=lambda x: x,
) -> Array:
    return conv_transpose_general(
        params,
        x,
        2,
        out_channels,
        kernel_shape,
        stride,
        padding,
        with_bias,
        mask,
        weights_initializer,
        bias_initializer,
        feature_group_count,
        data_format,
        activation,
    )


@flexible
def conv_transpose_3d(
    params: PyTree,
    x: Array,
    out_channels: int,
    kernel_shape: ShapeExpression,
    stride: ShapeExpression,
    padding: PaddingPreset | Sequence[tuple[int, int]] = "SAME",
    with_bias: bool = True,
    mask: Array | None = None,
    weights_initializer: initializers.Initializer = initializers.initializer_base,
    bias_initializer: initializers.Initializer = initializers.initializer_base,
    feature_group_count: int = 1,
    data_format: Literal["CHANNELS_LAST"] = "CHANNELS_LAST",
    activation=lambda x: x,
) -> Array:
    return conv_transpose_general(
        params,
        x,
        3,
        out_channels,
        kernel_shape,
        stride,
        padding,
        with_bias,
        mask,
        weights_initializer,
        bias_initializer,
        feature_group_count,
        data_format,
        activation,
    )


def _to_dimenstion_numbers(
    num_spatial_axes: int, channels_last: bool, transpose: bool
) -> ConvDimensionNumbers:
    num_axes = num_spatial_axes + 2
    if channels_last:
        spatial_axes = tuple(range(1, num_axes - 1))
        image_dimension_numbers = (0, num_axes - 1) + spatial_axes
    else:
        spatial_axes = tuple(range(2, num_axes))
        image_dimension_numbers = (0, 1) + spatial_axes

    if transpose:
        kernel_dimension_numbers = (num_axes - 2, num_axes - 1) + tuple(
            range(num_axes - 2)
        )
    else:
        kernel_dimension_numbers = (num_axes - 1, num_axes - 2) + tuple(
            range(num_axes - 2)
        )

    return ConvDimensionNumbers(
        lhs_spec=image_dimension_numbers,
        rhs_spec=kernel_dimension_numbers,
        out_spec=image_dimension_numbers,
    )


def _to_padding_tuple(
    padding: tuple[int, int], num_spatial_axes: int
) -> Sequence[tuple[int, int]] | tuple[tuple[int, int]]:
    if type(padding[0]) is int:
        return (padding,) * num_spatial_axes
    elif len(padding) == 1:
        return tuple(padding) * num_spatial_axes
    raise ValueError(
        f"Padding {padding} must of length {num_spatial_axes} but got {len(padding)}"
    )


def _to_shape_tuple(x: ShapeExpression, num_times: int) -> Sequence[int]:
    if type(x) is int:
        return (x,) * num_times
    if len(x) == 1:
        return tuple(x * num_times)
    if len(x) == num_times:
        return tuple(x)
    raise ValueError(f"Did not understand {x}")

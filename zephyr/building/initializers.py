from typing import Callable
from typing import Literal
from typing import Tuple

import numpy as np
from jax import numpy as jnp
from jax import random
from jaxtyping import Array

from zephyr.project_typing import ActivationFunctionsWithKnownGain
from zephyr.project_typing import KeyArray
from zephyr.project_typing import Shape

Initializer = Callable[[KeyArray, Tuple[int, ...]], Array]
from typing import Union


def initializer_base(key: KeyArray, shape: Shape) -> Array:
    if len(shape) > 1:
        return kaiming_normal(
            key, shape
        )  # todo: maybe zephyr should use leaky_relu as default activation?
    return zeros(key, shape)  # bias terms are usually rank 0


def ones(key: KeyArray, shape: Shape) -> Array:
    return jnp.ones(shape)


def zeros(key: KeyArray, shape: Shape) -> Array:
    return jnp.zeros(shape)


def uniform(key: KeyArray, shape: Shape) -> Array:
    return random.uniform(key, shape)


def normal(key: KeyArray, shape: Shape) -> Array:
    return random.normal(key, shape)


def glorot_uniform(key: Array, shape: Shape, gain: float = 1.0) -> Array:
    """From Understanding the difficulty of training deep feedforward
    neural networks by Glorot, ..., Bengio"""
    fan_average = (
        _calculate_fan(shape, mode_in=True) + _calculate_fan(shape, mode_out=True)
    ) / 2.0
    standard_deviation = gain * jnp.sqrt(jnp.reciprocal(fan_average))
    bound = jnp.sqrt(3) * standard_deviation
    return random.uniform(key, shape, minval=-bound, maxval=bound)


def glorot_normal(key: Array, shape: Shape, gain: float = 1.0) -> Array:
    """From Understanding the difficulty of training deep feedforward
    neural networks by Glorot, ..., Bengio"""
    fan_average = (
        _calculate_fan(shape, mode_in=True) + _calculate_fan(shape, mode_out=True)
    ) / 2.0
    standard_deviation = gain * jnp.sqrt(jnp.reciprocal(fan_average))
    return random.normal(key, shape) * standard_deviation


def kaiming_normal(
    key: KeyArray,
    shape: Shape,
    mode: Literal["fan_in", "fan_out"] = "fan_in",
    activation_function_name: ActivationFunctionsWithKnownGain = "leaky_relu",
) -> Array:
    # from Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification by He,..., Sun
    fan = (
        _calculate_fan(shape, mode_in=True)
        if mode == "fan_in"
        else _calculate_fan(shape, mode_out=True)
    )

    standard_deviation = _calcluate_gain(activation_function_name) / jnp.sqrt(fan)

    bound = jnp.sqrt(3) * standard_deviation
    return random.uniform(key, shape, minval=-bound, maxval=bound)


def _calculate_fan(
    shape: Shape, mode_in: bool = False, mode_out: bool = False
) -> float:
    if mode_in and mode_out:
        raise ValueError("Choose only one mode")
    if not mode_in and not mode_out:
        raise ValueError("Choose one")

    index = 1 if mode_in else 0
    num_fmaps = shape[index]
    receptive_field_size = 1.0
    if len(shape) > 2:
        receptive_field_size = float(np.prod(shape[3:]))

    return num_fmaps * receptive_field_size


def _calcluate_gain(
    activation_function: ActivationFunctionsWithKnownGain,
) -> Union[float, Array]:
    if activation_function in ["linear", "conv"] + ["sigmoid"]:
        return 1
    elif activation_function == "tanh":
        return float(5 / 3)
    elif activation_function == "relu":
        return jnp.sqrt(2)
    elif activation_function == "leaky_relu":
        return jnp.sqrt(2 / (1 + 0.01**2))
    else:
        raise ValueError("activation_function is unknown")

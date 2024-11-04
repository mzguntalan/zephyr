from typing import Callable
from typing import Union

from jaxtyping import Array

from zephyr.building._template.array import array
from zephyr.building.initializers import Initializer
from zephyr.building.initializers import initializer_base
from zephyr.building.skeleton import Skeleton
from zephyr.functools.partial import hole_aware
from zephyr.functools.partial import placeholder_hole as _
from zephyr.project_typing import Shape


def validate(
    params: Union[Array, Skeleton],
    shape: Shape = None,
    initializer: Initializer = initializer_base,
    expression: Callable = lambda params: True,
) -> Union[Array, Skeleton]:
    if type(params) is Skeleton and shape is None:
        return params
    if shape is None and expression is None:
        return params
    if shape is None and expression is not None:
        expression(params)
        return params

    if type(params) is Skeleton:
        params == array(shape, initializer)
        return params
    else:
        # print(params.shape, shape, "<---")
        if not all([params.shape == shape]):
            raise ValueError(
                f"Incompatible shapes: shape of `params`: {params.shape} should be {shape} as specified."
            )
    return params  # not reachable, just for mypy

from typing import Union

from jaxtyping import Array

from zephyr.building._template.array import array
from zephyr.building.initializers import Initializer
from zephyr.building.initializers import initializer_base
from zephyr.building.skeleton import Skeleton
from zephyr.project_typing import Shape


def validate(
    params: Union[Array, Skeleton],
    shape: Shape,
    initializer: Initializer = initializer_base,
) -> None:
    if type(params) is Skeleton:
        params == array(shape, initializer)
    else:
        if params.shape != shape:
            raise ValueError(
                f"Incompatible shapes: shape of `params`: {params.shape} should be {shape} as specified."
            )

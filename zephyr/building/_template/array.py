from functools import partial

from zephyr.building.initializers import Initializer
from zephyr.building.initializers import initializer_base
from zephyr.project_typing import ArrayTemplate
from zephyr.project_typing import Shape

ArrayTemplate_ = partial[ArrayTemplate]


def array(shape: Shape, initializer: Initializer = initializer_base) -> ArrayTemplate:
    return partial(initializer, shape=shape)


def array_equal(array_1: ArrayTemplate_, array_2: ArrayTemplate_) -> bool:
    return (
        array_1.func == array_2.func
        and array_1.keywords["shape"] == array_2.keywords["shape"]
    )

from typing import Callable
from typing import Tuple
from typing import Union

from jaxtyping import Array
from jaxtyping import PyTree

from zephyr.building.skeleton import Skeleton
from zephyr.project_typing import KeyArray


def trace(f: Callable, key: KeyArray, *args) -> PyTree:
    params = Skeleton(key)
    f(params, *args)

    return params.materialize()

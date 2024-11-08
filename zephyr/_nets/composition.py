# ?
# todo: maybe a sequential utility?
# what are good ways to compose functions relevant for DNNs?
from functools import partial
from functools import wraps
from typing import Callable
from typing import Sequence
from typing import Union

from jaxtyping import Array
from jaxtyping import PyTree

from zephyr.functools import composition
from zephyr.functools.composition import chain
from zephyr.functools.composition import thread_params
from zephyr.functools.partial import hole_aware
from zephyr.functools.partial import placeholder_hole as _

Params = PyTree
Layer = Callable[[Params, Array], Array]


@hole_aware
def sequential(
    params: PyTree, x: Array, layers: Sequence[Callable[[Params, Array], Array]]
) -> Array:
    return chain(thread_params(layers, params))(x)

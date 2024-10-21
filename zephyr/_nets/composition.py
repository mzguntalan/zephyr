# ?
# todo: maybe a sequential utility?
# what are good ways to compose functions relevant for DNNs?
from typing import Callable
from typing import Sequence
from typing import Union

from jaxtyping import Array
from jaxtyping import PyTree

from zephyr.functools.partial import hole_aware

Params = PyTree
Layer = Callable[[Params, Array], Array]


@hole_aware
def sequential(
    params: PyTree, x: Array, layers: Sequence[Callable[[Params, Array], Array]]
) -> Array:
    for i, layer in enumerate(layers):
        x = layer(params[i], x)
    return x

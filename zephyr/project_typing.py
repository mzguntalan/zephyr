from typing import Callable
from typing import Literal
from typing import NewType
from typing import Sequence
from typing import Union

from jaxtyping import Array

# KeyArray = NewType("KeyArray", Array)

KeyArray = Array
Shape = tuple[int, ...]
ActivationFunctionsWithKnownGain = Literal[
    "linear",
    "conv",  # corresponds to any conv (1d,2d, 1d transpose, etc)
    "sigmoid",
    "tanh",
    "relu",
    "leaky_relu",
]

PaddingPreset = Literal["SAME", "VALID", "CIRCULAR", "CAUSAL"]


ArrayTemplate = Callable[[KeyArray], Array]

ShapeExpression = Union[int, Sequence[int]]

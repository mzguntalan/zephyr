from functools import partial
from typing import Callable

from zephyr.project_typing import Any
from zephyr.project_typing import KeyArray

Template = Callable[[KeyArray], Any]


# example
def generic(key): ...

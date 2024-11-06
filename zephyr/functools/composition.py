"""TODO: Decide if these should go in _nets: they are increasingly useful in creation of nets"""

from functools import partial
from functools import wraps
from typing import Any
from typing import Callable
from typing import Sequence

from jax import random
from jaxtyping import Array

from zephyr.functools.partial import flexible


# @flexible
def thread(
    functions: Sequence[Callable], t: Any, split_rule: Callable = lambda x, i: x
) -> Sequence[Callable]:
    threaded_functions = []
    i = -1
    for i, fn in enumerate(functions[:-1]):
        t, t_sub = split_rule(t, i)
        threaded_functions.append(partial(fn, t_sub))

    threaded_functions.append(partial(functions[-1], t))

    return threaded_functions


@flexible
def chain(functions: Sequence[Callable]) -> Callable:
    @flexible
    def f(x):
        for fn in functions:
            x = fn(x)

        return x

    return f


@flexible
def params_split(params, i):
    return params[i], params[i + 1]


@flexible
def key_split(key, i):
    return random.split(key)


@flexible
def identity_split(x, i):
    return x


@flexible
def skip(f):
    @flexible
    @wraps(f)
    def inner(x):
        return x + f(x)

    return inner


thread_params = partial(thread, split_rule=params_split)
thread_key = partial(thread, split_rule=key_split)
thread_identity = partial(thread, split_rule=identity_split)

from functools import partial
from typing import Any
from typing import Callable
from typing import Sequence

from jax import random
from jaxtyping import Array


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


def chain(functions: Sequence[Callable]) -> Callable:
    def f(x):
        for fn in functions:
            x = fn(x)

        return x

    return f


def params_split(params, i):
    return params[i], params[i + 1]


def key_split(key, i):
    return random.split(key)


def identity_split(x, i):
    return x


thread_params = partial(thread, split_rule=params_split)
thread_key = partial(thread, split_rule=key_split)
thread_identity = partial(thread, split_rule=identity_split)

"""Module that would be possibly helpful for partial
application of functions in python


Notes:
    - experimental: might or might not be useful for the 
    rest of the lib
    - hopefully: it helps create more readable code
"""

from itertools import chain
from typing import Any
from typing import Callable
from typing import ParamSpec
from typing import Sequence
from typing import TypeVar
from typing import Union


class Hole:
    def __init__(self, name: str = ""):
        self._name = name


hole = Hole()


Parameters = ParamSpec("Parameters")
ReducedParameters = ParamSpec("ReducedParameters")
MissingParameters = ParamSpec("MissingParameters")
Return = TypeVar("Return")
FunctionToBeWrapped = Callable[Parameters, Return]
InnerFunction = Callable[
    Parameters, Callable[Parameters, Union[Return, Callable[MissingParameters, Return]]]
]


def hole_aware(f: FunctionToBeWrapped) -> InnerFunction:
    def inner(
        *args_possibly_with_placeholders: Parameters,
        **kwargs_possibly_with_placeholders: Parameters,
    ) -> Union[Return, Callable[MissingParameters, Return]]:
        """Example Usage

        @placeholder_aware
        def g(x,y,z): return x+y+z

        placeholder = Placeholder()

        x,y,z = 1,2,3
        g(x,y,z) # 6
        g(placeholder, y, z)(x) # 6
        g(x, placeholder, placeholder)(y,z) # 6
        g(placeholder, y, placeholder)(x,y,z) # 6
        """
        is_with_placeholder = False
        for arg in chain(
            args_possibly_with_placeholders, kwargs_possibly_with_placeholders.values()
        ):
            if type(arg) is Hole:
                is_with_placeholder = True
                break

        if not is_with_placeholder:
            complete_args = args_possibly_with_placeholders
            complete_kwargs = kwargs_possibly_with_placeholders
            return f(*complete_args, **kwargs_possibly_with_placeholders)
        if is_with_placeholder:

            @hole_aware
            def almost_f(
                *missing_args: MissingParameters,
                **missing_kwargs_or_overwrites: MissingParameters,
            ) -> Return:

                missing_args_supply = iter(missing_args)
                complete_args = []
                for arg in args_possibly_with_placeholders:
                    if type(arg) is Hole:
                        complete_args.append(next(missing_args_supply))
                    else:
                        complete_args.append(arg)

                complete_kwargs = (
                    kwargs_possibly_with_placeholders | missing_kwargs_or_overwrites
                )

                if _contains_hole(complete_kwargs.values()):
                    return almost_f(*complete_args, **complete_kwargs)

                return f(*complete_args, **complete_kwargs)

            return almost_f

    return inner


# aliases
Placeholder = Hole
placeholder = hole
placeholder_aware = hole_aware


def _contains_hole(seq: Sequence):
    for item in seq:
        if type(item) is Hole:
            return True
    return False

"""Module that would be possibly helpful for partial
application of functions in python


Notes:
    - experimental: might or might not be useful for the 
    rest of the lib
    - hopefully: it helps create more readable code
"""

from functools import wraps
from itertools import chain
from itertools import repeat
from typing import Any
from typing import Callable
from typing import ParamSpec
from typing import Sequence
from typing import TypeVar
from typing import Union


class Hole:
    def __init__(self, name: str = ""):
        self._name = name
        self._value = Underived()

    def __eq__(self, anything_is_equal_to_this):
        if self.is_unset:
            self._value = anything_is_equal_to_this
            return True
        return self._value == anything_is_equal_to_this

    @property
    def value(self):
        return self._value

    @property
    def is_set(self):
        return not self.is_unset

    @property
    def is_unset(self):
        return type(self.value) is Underived

    def __req__(self, anything_is_equal_to_this):
        if type(self.value) is Underived:
            self._value = anything_is_equal_to_this
            return True
        return self._value == anything_is_equal_to_this

    def __iter__(self):
        return repeat(Hole())

    def __add__(self, x):
        return Hole()

    def __radd__(self, x):
        return Hole()

    def __sub__(self, x):
        return Hole()

    def __rsub__(self, x):
        return Hole()

    def __mul__(self, x):
        return Hole()

    def __rmul__(self, x):
        return Hole()

    def __truediv__(self, x):
        return Hole()

    def __rtruediv__(self, x):
        return Hole()

    def __str__(self):
        return str(self.value)

    def __getitem__(self, key):
        return Hole(lambda x: self.unpack(x)["key"])


class PlaceholderHole(Hole): ...


class DerivableHole(Hole):
    def __init__(self, noisy=False):
        Hole.__init__(self)
        self._noisy = noisy  # todo: this is for debug and understanding only

    def __eq__(self, x):
        if self._noisy:
            print("Derived: ", x)
        return Hole.__eq__(self, x)

    def __req__(self, x):
        if self._noisy:
            print("Derived: ", x)
        return Hole.__eq__(self, x)


class Underived:
    def __init__(self):
        pass

    def __str__(self):
        return "UNDERIVED"


class IncompleteDerivationError(ValueError): ...


placeholder_hole = PlaceholderHole()
derivable_hole = DerivableHole()
noisy_derivable_hole = DerivableHole(noisy=True)

Parameters = ParamSpec("Parameters")
ReducedParameters = ParamSpec("ReducedParameters")
MissingParameters = ParamSpec("MissingParameters")
Return = TypeVar("Return")
FunctionToBeWrapped = Callable[Parameters, Return]
InnerFunction = Callable[
    Parameters, Callable[Parameters, Union[Return, Callable[MissingParameters, Return]]]
]


def infinite_generator(value):
    while True:
        yield value


class Unspecified:
    def __init__(self):
        pass

    def __str__(self):
        return "UNSPECIFIED"


unspecified_parameter = Unspecified()


# doing: replace REPLACEMENT OF HOLES with CALLING OF HOLES
def hole_aware(f: FunctionToBeWrapped) -> InnerFunction:
    @wraps(f)
    def inner(
        *args_possibly_with_placeholders: Parameters,
        **kwargs_possibly_with_placeholders: Parameters,
    ) -> Union[Return, Callable[MissingParameters, Return]]:
        is_with_placeholder = False
        for arg in chain(
            args_possibly_with_placeholders, kwargs_possibly_with_placeholders.values()
        ):
            if isinstance(arg, PlaceholderHole):
                is_with_placeholder = True
                break

        if not is_with_placeholder:
            return f(
                *args_possibly_with_placeholders, **kwargs_possibly_with_placeholders
            )

        @hole_aware
        def almost_f(
            *missing_args: MissingParameters,
            **missing_kwargs_or_overwrites: MissingParameters,
        ) -> Return:
            missing_args_supply = chain(iter(missing_args))
            complete_args = []
            for arg in args_possibly_with_placeholders:
                if isinstance(arg, PlaceholderHole):
                    candidate_arg_for_hole = next(missing_args_supply)

                    supplied_arg = candidate_arg_for_hole

                    complete_args.append(supplied_arg)
                else:
                    complete_args.append(arg)

            complete_kwargs = (
                kwargs_possibly_with_placeholders | missing_kwargs_or_overwrites
            )

            if _contains_placeholder_hole(complete_kwargs.values()):
                return almost_f(*complete_args, **complete_kwargs)

            return hole_aware(f)(*complete_args, **complete_kwargs)

        return almost_f

    inner._original_function = f
    return inner


def deriving_holes(f: FunctionToBeWrapped) -> InnerFunction:
    @wraps(f)
    def inner(*args, **kwargs):
        new_args = []
        new_kwargs = {}
        all_holes = []

        for arg in args:
            if type(arg) is DerivableHole:
                v = DerivableHole(noisy=arg._noisy)
                new_args.append(v)
                all_holes.append(v)
            else:
                new_args.append(arg)

        for k in kwargs:
            v = kwargs[k]
            if type(v) is DerivableHole:
                v = DerivableHole(noisy=v._noisy)
                new_kwargs[k] = v
                all_holes.append(v)
            else:
                new_kwargs[k] = v

        try:
            return f(*new_args, **new_kwargs)
        except:

            filled_args = []
            filled_kwargs = {}

            report_args = []
            report_kwargs = {}

            report_terms = {True: "Derived", False: "Underived", -1: "Given"}
            is_complete = True

            for arg in new_args:
                if type(arg) is DerivableHole:
                    report_args.append(report_terms[arg.is_set])
                    filled_args.append(arg.value)
                else:
                    report_args.append(report_terms[-1])
                    filled_args.append(arg)

            for k in new_kwargs:
                v = new_kwargs[k]
                if type(v) is DerivableHole:
                    report_kwargs[k] = report_terms[v.is_set]
                    filled_kwargs[k] = v.value
                else:
                    report_kwargs[k] = report_terms[-1]
                    filled_kwargs[k] = v

            if not all([h.is_set for h in all_holes]):
                raise IncompleteDerivationError(
                    f"Cannot fully derive values for {f.__name__} args:{str(report_args)} kwargs: {str(report_kwargs)}"
                )

            return f(*filled_args, **filled_kwargs)

    return inner


def _contains_placeholder_hole(seq: Sequence[Any]) -> bool:
    for item in seq:
        if type(item) is PlaceholderHole:
            return True
    return False


def flexible(f):
    # deriving_holes is postponed as it is very tedious to write
    @hole_aware
    @wraps(f)
    def inner(*args, **kwargs):
        return f(*args, **kwargs)

    return inner

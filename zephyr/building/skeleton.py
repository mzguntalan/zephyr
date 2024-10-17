from jax import numpy as jnp
from jax import random

from zephyr.building.template import ArrayTemplate
from zephyr.project_typing import KeyArray


class Skeleton:
    def __init__(self, key: KeyArray = random.PRNGKey(0)):
        self._contents = {}
        self._key = random.PRNGKey(key)
        self.dtype = (
            jnp.float32
        )  # some jax operations don't work if the object does not have a valid jax dtype;
        # if it does then jax calls __jax_array__

    def materialize(self):
        """This initializes the arrays at the leaves using the appropriate initializers"""
        if type(self._contents) is dict:
            d = {}
            for k in self._contents:
                r = self._contents[k]
                if callable(r):
                    self._key, key = random.split(self._key)
                    r = r(key)
                else:  # is an skeletal params
                    r = r.materialize()
                d[k] = r
            return d
        else:  # array_template
            self._key, key = random.split(self._key)
            return self._contents(key)  # array

    def __jax_array__(self):
        return self._contents(self._key)

    def __getitem__(self, key):
        if key in self._contents:
            return self._contents[key]
        else:
            new_params = Skeleton()
            self._contents[key] = new_params
            return self._contents[key]

    def __eq__(self, array_template: ArrayTemplate):
        self._contents = array_template
        return True

    def __add__(self, x):
        return self._contents(self._key) + x

    def __radd__(self, x):
        return x + self._contents(self._key)

    def __sub__(self, x):
        return self._contents(self._key) - x

    def __rsub__(self, x):
        return x - self._contents(self._key)

    def __mul__(self, x):
        return self._contents(self._key) * x

    def __rmul__(self, x):
        return x * self._contents(self._key)

    def __matmul__(self, x):
        return self._contents(self._key) @ x

    def __rmatmul__(self, x):
        return x @ self._contents(self._key)

    def __truediv__(self, x):
        return self._contents(self._key) / x

    def __rtruediv__(self, x):
        return x / self._contents(self._key)

    def __pow__(self, n):
        return self._contents(self._key) ** n

    def __neg__(self):
        return -self._contents(self._key)

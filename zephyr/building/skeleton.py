from collections.abc import Sequence
from typing import Any

import jax
import numpy as np
from jax import Array
from jax import numpy as jnp
from jax import random

from zephyr.project_typing import ArrayTemplate
from zephyr.project_typing import KeyArray
from zephyr.project_typing import Shape

# match jax._src.basearray typing
Shard = Any
Sharding = Any
Device = Any


class Skeleton(Array):
    def __init__(self, key: KeyArray = random.PRNGKey(0)):
        self._contents = {}
        self._key = key

    def materialize(self):
        """This initializes the arrays at the leaves using the appropriate initializers"""
        if type(self._contents) is dict:
            d = {}
            for k in self._contents:
                r = self._contents[k]
                if callable(r):
                    self._key, key = random.split(self._key)
                    r = r(key)
                else:  # is an skeletal
                    r = r.materialize()
                d[k] = r
            return d
        else:  # array_template
            self._key, key = random.split(self._key)
            return self._contents(key)  # array

    def __jax_array__(self):
        return self.materialize()

    def __array__(self):
        return self.materialize()

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
        return self.materialize() + x

    def __radd__(self, x):
        return x + self.materialize()

    def __sub__(self, x):
        return self.materialize() - x

    def __rsub__(self, x):
        return x - self.materialize()

    def __mul__(self, x):
        return self.materialize() * x

    def __rmul__(self, x):
        return x * self.materialize()

    def __matmul__(self, x):
        return self.materialize() @ x

    def __rmatmul__(self, x):
        return x @ self.materialize()

    def __truediv__(self, x):
        return self.materialize() / x

    def __rtruediv__(self, x):
        return x / self.materialize()

    def __pow__(self, n):
        return self.materialize() ** n

    def __neg__(self):
        return -self.materialize()

    @property
    def dtype(self) -> np.dtype:
        return np.float32

    @property
    def ndim(self) -> int:
        return len(self.materialize().shape)

    @property
    def size(self) -> int:
        return np.prod(self.materialize().shape)

    @property
    def shape(self) -> Shape:
        return self.materialize().shape

    def addressable_data(self, index: int) -> Array:
        return self.materialize().addressable_data(index)

    @property
    def addressable_shards(self) -> Sequence[Shard]:
        return self.materialize().addressable_shards

    @property
    def global_shards(self) -> Sequence[Shard]:
        return self.materialize().global_shards

    @property
    def is_fully_addressable(self) -> bool:
        return self.materialize().is_fully_addressable

    @property
    def is_fully_replicated(self) -> bool:
        return self.materialize().is_fully_replicated

    @property
    def sharding(self) -> Sharding:
        return self.materialize().sharding

    @property
    def commited(self) -> bool:
        return (
            self.materialize().commited
        )  # this might return an AttributeError, but that comes from jax initializing an array without this attribute <- not zephyr problem (so far)

    @property
    def device(self) -> Device | Sharding:
        return jax.default_device()

    def copy_to_host_async(self) -> None:
        return

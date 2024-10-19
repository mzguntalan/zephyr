from jax import numpy as jnp
from jax import random
from jaxtyping import Array
from pytest import mark

from zephyr.building.initializers import initializer_base
from zephyr.building.skeleton import Skeleton
from zephyr.building.template import array
from zephyr.building.template import validate
from zephyr.project_typing import Shape


@mark.parametrize(["shape"], [((16, 16),), ((8,),), ((8, 16, 32),)])
def test_array(shape: Shape) -> None:
    f = array(shape, initializer_base)
    assert callable(f)

    z = f(random.PRNGKey(0))
    assert isinstance(z, Array)


@mark.parametrize(["shape"], [((16, 16),), ((8,),), ((8, 16, 32),)])
def test_validate(shape: Shape) -> None:
    params_skeleton = Skeleton()
    validate(params_skeleton, shape, initializer_base)
    assert params_skeleton.materialize().shape == shape

    params_with_correct_shape = jnp.ones(shape)
    try:
        validate(params_with_correct_shape, shape, initializer_base)
    except:
        raise AssertionError("Validate should not throw an error")

    params_with_wrong_shape = jnp.ones(tuple(i + 1 for i in shape))
    try:
        validate(params_with_wrong_shape, shape, initializer_base)
        raise AssertionError(
            "Validate should throw an error when shapes do not match. It did not throw an error"
        )
    except:
        pass


@mark.parametrize(["shape"], [((16, 16),), ((8,),), ((8, 16, 32),)])
def test_validate_multiple_validations(shape) -> None:
    params_skeleton = Skeleton()
    validate(params_skeleton, shape, initializer_base)
    validate(params_skeleton, shape, initializer_base)
    pass

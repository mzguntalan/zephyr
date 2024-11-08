from typing import Any
from typing import Callable

from jaxtyping import Array

from zephyr.functools.partial import flexible

Leaf = Any
Tree = Any
Kwargs = Any


@flexible
def apply_updates(
    update: Callable[[Leaf, ..., Kwargs], Leaf], *trees, **kwargs_for_update
) -> Tree:
    example_tree = trees[0]
    if not isinstance(example_tree, dict):
        return update(*trees, **kwargs_for_update)
    return {
        k: apply_updates(update, *[tree[k] for tree in trees], **kwargs_for_update)
        for k in example_tree
    }  # trees should have the same structure

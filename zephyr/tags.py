from typing import Any

from jaxtyping import Array
from jaxtyping import PyTree

from zephyr.functools.partial import flexible

Leaf = Any
Tag = Any


@flexible
def get_last_n_tags(
    p: dict[Tag, Leaf], n: int, context: list[Tag] = []
) -> dict[Tag, list[Tag]]:
    if not isinstance(p, dict):  # it is a leaf
        return context[-n - 1 : -1]
    return {k: get_last_n_tags(v, n, context=context + [k]) for k, v in p.items()}


@flexible
def get_immediate_tags(
    p: dict[Tag, Leaf], context: list[Tag] = []
) -> dict[Tag, list[Tag]]:
    return get_last_n_tags(p, 1)


@flexible
def get_lineage_tags(
    p: dict[Tag, Leaf], context: list[Tag] = []
) -> dict[Tag, list[Tag]]:
    return get_last_n_tags(p, -1)

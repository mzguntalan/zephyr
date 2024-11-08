from typing import Any

from jaxtyping import Array
from jaxtyping import PyTree

from zephyr.functools.partial import flexible

Leaf = Any
Tag = Any


@flexible
def get_immediate_tags(params: dict, context=[]):
    if isinstance(params, Array):
        return context
    return {k: get_immediate_tags(v, context=[k]) for k, v in params.items()}


@flexible
def get_lineage_tags(params: dict, context=[]):
    if isinstance(params, Array):
        return context
    return {k: get_lineage_tags(v, context=context + [k]) for k, v in params.items()}

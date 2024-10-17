from zephyr.building.skeleton import Skeleton


def trace(f, *args):
    params = Skeleton()
    f(params, *args)

    return params.materialize()

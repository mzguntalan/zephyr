from zephyr.building.skeleton import SkeletalParams


def trace(f, *args):
    params = SkeletalParams()
    f(params, *args)

    return params.materialize()

from zephyr.functools.partial import Hole
from zephyr.functools.partial import hole_aware


def test_make_aware_of_placeholders():
    @hole_aware
    def g(a, b, c, d):
        return a + b + c + d

    a, b, c, d = 1, -1, 2, 3
    total = a + b + c + d

    _ = Hole()

    assert g(a, b, c, d) == total
    assert g(_, b, c, d)(a) == total
    assert g(_, b, _, d)(a, c) == total
    assert g(_, b, _, _)(a, c, d) == total
    assert g(_, _, _, _)(a, b, c, d) == total


def test_make_aware_of_placeholders_nested():
    @hole_aware
    def g(a, b, c, d, e, f):
        return a + b + c + d + e + f

    a, b, c, d, e, f = 1, 2, 4, 8, 16, 32

    total = a + b + c + d + e + f
    _ = Hole()

    assert g(a, b, c, d, e, f) == total
    assert g(_, _, _, _, _, _)(a, b, c, d, e, f) == total
    assert g(_, _, _, _, _, f)(a, b, c, d, e) == total
    assert g(_, _, _, _, _, f)(_, b, _, _, _)(a, c, d, e) == total
    assert g(_, _, _, _, _, f)(_, b, _, _, _)(a, _, _, e)(c, d) == total
    assert g(_, _, _, _, _, f)(_, b, _, _, _)(a, _, _, e)(c, _)(d) == total


def test_hole_aware_args_and_kwargs():
    @hole_aware
    def g(a, b, c, d, e, f, offset=0, scale=1):
        return scale * (a + b + c + d + e + f) + offset

    a, b, c, d, e, f = 1, 2, 4, 8, 16, 32
    offset = 1
    scale = 2

    total = a + b + c + d + e + f
    _ = Hole()

    assert g(a, b, c, d, e, f) == total
    assert g(a, b, c, d, e, f, offset=_, scale=_)(offset=0, scale=1) == total
    assert g(a, b, c, d, e, f, offset=_, scale=_)(offset=0)(scale=1) == total
    assert g(a, b, c, d, e, _)(f, offset=_)(offset=1, scale=_)(scale=2) == 2 * total + 1

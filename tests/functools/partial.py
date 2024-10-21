from zephyr.functools.partial import make_aware_of_placeholders
from zephyr.functools.partial import Placeholder


def test_make_aware_of_placeholders():
    @make_aware_of_placeholders
    def g(a, b, c, d):
        return a + b + c + d

    a, b, c, d = 1, -1, 2, 3
    total = a + b + c + d

    _ = Placeholder()

    assert g(a, b, c, d) == total
    assert g(_, b, c, d)(a) == total
    assert g(_, b, _, d)(a, c) == total
    assert g(_, b, _, _)(a, c, d) == total
    assert g(_, _, _, _)(a, b, c, d) == total


def test_make_aware_of_placeholders_nested():
    @make_aware_of_placeholders
    def g(a, b, c, d, e, f):
        return a + b + c + d + e + f

    a, b, c, d, e, f = 1, 2, 4, 8, 16, 32

    total = a + b + c + d + e + f
    _ = Placeholder()

    assert g(a, b, c, d, e, f) == total
    assert g(_, _, _, _, _, _)(a, b, c, d, e, f) == total
    assert g(_, _, _, _, _, f)(a, b, c, d, e) == total
    assert g(_, _, _, _, _, f)(_, b, _, _, _)(a, c, d, e) == total
    assert g(_, _, _, _, _, f)(_, b, _, _, _)(a, _, _, e)(c, d) == total
    assert g(_, _, _, _, _, f)(_, b, _, _, _)(a, _, _, e)(c, _)(d) == total

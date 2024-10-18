from zephyr.functools.partial import make_aware_of_placeholders
from zephyr.functools.partial import Placeholder


def test_make_aware_of_placeholders():
    @make_aware_of_placeholders
    def g(a, b, c, d):
        return a + b + c + d

    a, b, c, d = 1, -1, 2, 3
    total = a + b + c + d

    p = Placeholder()

    assert g(a, b, c, d) == total
    assert g(p, b, c, d)(a) == total
    assert g(p, b, p, d)(a, c) == total
    assert g(p, b, p, p)(a, c, d) == total
    assert g(p, p, p, p)(a, b, c, d) == total

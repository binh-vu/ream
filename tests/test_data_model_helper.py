from typing import get_type_hints


def test_get_type_hints():
    class A:
        def __init__(self, a: int, b: int):
            self.a: int = a
            self.b: int = b

    assert get_type_hints(A) == {}

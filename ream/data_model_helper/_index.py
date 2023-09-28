from __future__ import annotations

import pickle
from typing import Generic, TypeVar

T = TypeVar("T")


class Index(Generic[T]):
    __slots__ = ["index"]

    def __init__(self, index: T):
        self.index = index

    def to_bytes(self) -> bytes:
        return pickle.dumps(self)

    @staticmethod
    def from_bytes(obj):
        return pickle.loads(obj)


class OffsetIndex(Index[T], Generic[T]):
    __slots__ = ["offset"]

    def __init__(self, index: T, offset: int):
        super().__init__(index)
        self.offset = offset


SupportedIndex = (dict, list, Index)

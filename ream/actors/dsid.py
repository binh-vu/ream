from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Generator, Generic, Optional, Sequence

from ream.actors.base import BaseActor, P
from ream.dataset_helper import DatasetList, E


class IDDSActor(ABC, Generic[E, P], BaseActor[P]):
    """A actor that is responsible for querying the dataset."""

    def __init__(self, params: P, dep_actor: Optional[Sequence[BaseActor]] = None):
        super().__init__(params, dep_actor)
        self.store: dict[str, list[E]] = {}

    def __call__(self, dsquery: str) -> DatasetList[E]:
        if dsquery in self.store:
            return DatasetList(dsquery, self.store[dsquery])
        return self.load_dataset(dsquery)

    @abstractmethod
    def load_dataset(self, dsquery: str) -> DatasetList[E]:
        raise NotImplementedError()

    @contextmanager
    def use_example(
        self, id: str, example: E, prefix: str = "ex:"
    ) -> Generator[str, None, None]:
        """Temporary add examples to the store, so other actors can use it via querying"""
        key = prefix + id
        assert key not in self.store
        self.store[key] = [example]
        try:
            yield key
        finally:
            del self.store[key]

    @contextmanager
    def use_examples(
        self, id: str, examples: list[E], prefix: str = "exs:"
    ) -> Generator[str, None, None]:
        """Temporary add examples to the store, so other actors can use it via querying"""
        key = prefix + id
        assert key not in self.store
        self.store[key] = examples
        try:
            yield key
        finally:
            del self.store[key]

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Generator, Generic, Optional

from ream.actors.base import BaseActor, P
from ream.dataset_helper import DatasetList, E


class IDDSActor(ABC, Generic[E, P], BaseActor[P]):
    """A actor that is responsible for querying the dataset."""

    def __init__(self, params: P, dep_actor: Optional[list[BaseActor]] = None):
        super().__init__(params, dep_actor)
        self.store = {}

    def __call__(self, dsquery: str) -> DatasetList[E]:
        if dsquery in self.store:
            return DatasetList(dsquery, [self.store[dsquery]])
        return self.load_dataset(dsquery)

    @abstractmethod
    def load_dataset(self, dsquery: str) -> DatasetList[E]:
        raise NotImplementedError()

    @contextmanager
    def use_example(self, id: str, example: E) -> Generator[str, None, None]:
        """Temporary add an example to the store, so other actors can use it via querying"""
        key = f"ex:{id}"
        assert key not in self.store
        self.store[key] = example
        yield key
        del self.store[key]

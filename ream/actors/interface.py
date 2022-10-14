from __future__ import annotations
from abc import ABC
from typing import (
    List,
    TypeVar,
    Generic,
)

E = TypeVar("E")


class Actor(ABC, Generic[E]):
    """A foundation unit to define a computational graph, which is your program.

    It can be:
        * your entire method
        * a component in your pipeline such as preprocessing
        * a wrapper, wrapping a library, or a step in your algorithm
            that you want to try different method, and see their performance

    Therefore, it should have two basic methods:
        * run: to run the actor with a given input
        * evaluate: to evaluate the current actor, and optionally store some debug information.
            It is recommended to not pass the evaluating datasets to the run function, but rather
            the examples in the datasets.

    ## How to configure this actor?

    By design, it can be configured via dataclasses containing parameters. But if necessary,
    you can also add more parameters to `run` or `batch_run` functions (not recommended).
    """

    def batch_run(self, examples: List[E]):
        """Run the actor with a list of examples"""
        raise NotImplementedError()

    def run(self, example: E):
        """Run the actor with a single example"""
        raise NotImplementedError()

    def run_on_dataset(self, dataset: str):
        """Run the actor on examples from a dataset."""
        raise NotImplementedError()

    def evaluate(self, *args: str):
        """Evaluate the actor. The evaluation metrics can be printed to the console,
        or stored in a temporary variable of this class to access it later."""
        raise NotImplementedError()

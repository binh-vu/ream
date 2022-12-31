from inspect import Parameter
from typing import List, Optional

import pytest
from ream.cache_helper import CacheArgsHelper
from ream.prelude import BaseActor, NoParams


class CandidateGeneration(BaseActor[List[str], NoParams]):
    VERSION = 100

    def __init__(self, params: NoParams):
        super().__init__(params)

    def algo1(self, dataset: str, topk: int, seed: Optional[int]):
        raise NotImplementedError()

    def algo2_noann(self, examples):
        raise NotImplementedError()

    def algo3_dyn1(self, *examples):
        raise NotImplementedError()

    def algo3_dyn2(self, name: str, **kwargs):
        raise NotImplementedError()


def func_with_complex_type(self, actor: CandidateGeneration):
    raise NotImplementedError()


# with __future__, all annotations become str: https://peps.python.org/pep-0563/
# so the function below is a testcase for that
def func_with_complex_type_postpone(self, actor: "CandidateGeneration"):
    raise NotImplementedError()


class TestCacheArgsHelper:
    def test_ensure_auto_cache_key_friendly(self):
        helper = CacheArgsHelper(CandidateGeneration.algo1)
        helper.ensure_auto_cache_key_friendly()
        for fn in [
            CandidateGeneration.algo2_noann,
            CandidateGeneration.algo3_dyn1,
            CandidateGeneration.algo3_dyn2,
        ]:
            helper = CacheArgsHelper(fn)
            with pytest.raises(TypeError):
                helper.ensure_auto_cache_key_friendly()

    def test_argtypes(self):
        helper = CacheArgsHelper(CandidateGeneration.algo2_noann)
        assert helper.get_cache_argtypes() == {
            "examples": None,
        }
        helper = CacheArgsHelper(func_with_complex_type)
        assert helper.get_cache_argtypes() == {
            "actor": CandidateGeneration,
        }
        helper = CacheArgsHelper(func_with_complex_type_postpone)
        assert helper.get_cache_argtypes() == {
            "actor": CandidateGeneration,
        }

    def test_keep_args(self):
        helper = CacheArgsHelper(CandidateGeneration.algo1)
        assert helper.get_cache_argtypes() == {
            "dataset": str,
            "topk": int,
            "seed": Optional[int],
        }
        helper.keep_args(["dataset", "seed"])
        assert helper.get_cache_argtypes() == {
            "dataset": str,
            "seed": Optional[int],
        }

    def test_get_args(self):
        obj = CandidateGeneration(NoParams())
        helper = CacheArgsHelper(CandidateGeneration.algo1)

        assert helper.get_args(obj, "test", 10) == {
            "dataset": "test",
            "topk": 10,
            "seed": Parameter.empty,
        }
        assert helper.get_args(obj, "test", seed=10) == {
            "dataset": "test",
            "topk": Parameter.empty,
            "seed": 10,
        }
        assert helper.get_args(obj, dataset="test", seed=10) == {
            "dataset": "test",
            "topk": Parameter.empty,
            "seed": 10,
        }
        assert helper.get_args(obj, seed=10, dataset="test") == {
            "dataset": "test",
            "topk": Parameter.empty,
            "seed": 10,
        }

        helper.keep_args(["dataset", "seed"])
        assert helper.get_args(obj, "test", 10) == {
            "dataset": "test",
            "seed": Parameter.empty,
        }
        assert helper.get_args(obj, "test", 4, 10) == {
            "dataset": "test",
            "seed": 10,
        }

    def test_get_args_as_tuple(self):
        obj = CandidateGeneration(NoParams())
        helper = CacheArgsHelper(CandidateGeneration.algo1)
        assert helper.get_args_as_tuple(obj, "test", 10) == (
            "test",
            10,
            Parameter.empty,
        )
        assert helper.get_args_as_tuple(obj, "test", seed=10) == (
            "test",
            Parameter.empty,
            10,
        )
        assert helper.get_args_as_tuple(obj, dataset="test", seed=10) == (
            "test",
            Parameter.empty,
            10,
        )
        assert helper.get_args_as_tuple(obj, seed=10, dataset="test") == (
            "test",
            Parameter.empty,
            10,
        )

        helper.keep_args(["dataset", "seed"])
        assert helper.get_args_as_tuple(obj, "test", 10) == ("test", Parameter.empty)

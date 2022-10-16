from dataclasses import dataclass
from typing import List, Literal, Optional

from ream.prelude import NoParams, BaseActor
from ream.cache_helper import CacheArgsHelper
from inspect import Parameter, signature, _empty
import pytest


class CandidateGeneration(BaseActor[List[str], NoParams]):
    VERSION = 100

    def algo1(self, dataset: str, topk: int, seed: Optional[int]):
        raise NotImplementedError()

    def algo2_noann(self, examples):
        raise NotImplementedError()

    def algo3_dyn1(self, *examples):
        raise NotImplementedError()

    def algo3_dyn2(self, name: str, **kwargs):
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

    def test_keep_args(self):
        helper = CacheArgsHelper(CandidateGeneration.algo1)
        assert helper.get_arg_type() == {
            "dataset": str,
            "topk": int,
            "seed": Optional[int],
        }
        helper.keep_args(["dataset", "seed"])
        assert helper.get_arg_type() == {
            "dataset": str,
            "seed": Optional[int],
        }

    def test_get_args(self):
        helper = CacheArgsHelper(CandidateGeneration.algo1)
        assert helper.get_args("test", 10) == {
            "dataset": "test",
            "topk": 10,
            "seed": Parameter.empty,
        }
        assert helper.get_args("test", seed=10) == {
            "dataset": "test",
            "topk": Parameter.empty,
            "seed": 10,
        }
        assert helper.get_args(dataset="test", seed=10) == {
            "dataset": "test",
            "topk": Parameter.empty,
            "seed": 10,
        }
        assert helper.get_args(seed=10, dataset="test") == {
            "dataset": "test",
            "topk": Parameter.empty,
            "seed": 10,
        }

        helper.keep_args(["dataset", "seed"])
        assert helper.get_args("test", 10) == {
            "dataset": "test",
            "seed": Parameter.empty,
        }
        assert helper.get_args("test", 4, 10) == {
            "dataset": "test",
            "seed": 10,
        }

    def test_get_args_as_tuple(self):
        helper = CacheArgsHelper(CandidateGeneration.algo1)
        assert helper.get_args_as_tuple("test", 10) == ("test", 10, Parameter.empty)
        assert helper.get_args_as_tuple("test", seed=10) == (
            "test",
            Parameter.empty,
            10,
        )
        assert helper.get_args_as_tuple(dataset="test", seed=10) == (
            "test",
            Parameter.empty,
            10,
        )
        assert helper.get_args_as_tuple(seed=10, dataset="test") == (
            "test",
            Parameter.empty,
            10,
        )

        helper.keep_args(["dataset", "seed"])
        assert helper.get_args_as_tuple("test", 10) == ("test", Parameter.empty)

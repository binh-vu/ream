from __future__ import annotations

import pickle
from inspect import Parameter
from pathlib import Path
from typing import List, Optional

import pytest

from ream.cache_helper import Backend, Cache, Cacheable, CacheArgsHelper, FileBackend
from ream.fs import FS
from ream.prelude import BaseActor, NoParams


class CandidateGeneration(BaseActor[NoParams]):
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
        helper = CacheArgsHelper.from_actor_func(CandidateGeneration.algo1)
        helper.ensure_auto_cache_key_friendly()
        for fn in [
            CandidateGeneration.algo2_noann,
            CandidateGeneration.algo3_dyn1,
            CandidateGeneration.algo3_dyn2,
        ]:
            helper = CacheArgsHelper.from_actor_func(fn)
            with pytest.raises(TypeError):
                helper.ensure_auto_cache_key_friendly()

    def test_argtypes(self):
        helper = CacheArgsHelper.from_actor_func(CandidateGeneration.algo2_noann)
        assert helper.get_cache_argtypes() == {
            "examples": None,
        }
        helper = CacheArgsHelper.from_actor_func(func_with_complex_type)
        assert helper.get_cache_argtypes() == {
            "actor": CandidateGeneration,
        }
        helper = CacheArgsHelper.from_actor_func(func_with_complex_type_postpone)
        assert helper.get_cache_argtypes() == {
            "actor": CandidateGeneration,
        }

    def test_keep_args(self):
        helper = CacheArgsHelper.from_actor_func(CandidateGeneration.algo1)
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
        helper = CacheArgsHelper.from_actor_func(CandidateGeneration.algo1)

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
        helper = CacheArgsHelper.from_actor_func(CandidateGeneration.algo1)
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


class TestCacheHelper:
    def make_cache_fn(self, backend: Backend, **kwargs):
        class MyCacheFn(Cacheable):
            def __init__(self, workdir: FS | Path, disable: bool = False):
                super().__init__(workdir, disable)
                self.coefficient = 1

            @Cache.cache(
                backend=backend,
                **kwargs,
            )
            def __call__(self, val: int) -> int:
                return (val**2) * self.coefficient

        return MyCacheFn

    def make_flat_cache_fn(self, backend: Backend, **kwargs):
        class MyCacheFn(Cacheable):
            def __init__(self, workdir: FS | Path, disable: bool = False):
                super().__init__(workdir, disable)
                self.coefficient = 1

            @Cache.flat_cache(
                backend=backend,
                **kwargs,
            )
            def __call__(self, val: list[int]) -> list[int]:
                return [(v**2) * self.coefficient for v in val]

        return MyCacheFn

    @pytest.fixture(scope="class", params=["file", "dir", "sqlite"][:1])
    def backend(self, request) -> Backend:
        if request.param == "file":
            return FileBackend(
                ser=pickle.dumps,
                deser=pickle.loads,
                fileext="pkl",
            )

        raise NotImplementedError(request.param)

    def test_cache(self, tmp_path: Path, backend: Backend):
        fn = self.make_cache_fn(backend)(tmp_path)

        assert fn(10) == 100
        fn.coefficient = 2
        assert fn(10) == 100
        fn.coefficient = 1

        assert fn(5) == 25
        fn.coefficient = 2
        assert fn(5) == 25

        assert fn(2) == 8
        fn.coefficient = 1
        assert fn(2) == 8

    def test_flat_cache(self, tmp_path: Path, backend: Backend):
        fn = self.make_flat_cache_fn(backend)(tmp_path)

        assert fn([10, 5, 2]) == [100, 25, 4]
        fn.coefficient = 2
        assert fn([10, 5, 2]) == [100, 25, 4]

        assert fn([3, 4, 7]) == [18, 32, 98]
        fn.coefficient = 1
        assert fn([3, 4, 7]) == [18, 32, 98]

    def test_cache_disable(self, tmp_path: Path, backend: Backend):
        for disable in ["disable", True]:
            fn = self.make_cache_fn(backend, disable=disable)(tmp_path, True)

            # disable cache should always return the result
            assert fn(10) == 100
            fn.coefficient = 2
            assert fn(10) == 200

        # disable attribute should only has lower priority
        fn = self.make_cache_fn(backend, disable=False)(tmp_path, True)
        assert fn(10) == 100
        fn.coefficient = 2
        assert fn(10) == 100

    def test_flat_cache_disable(self, tmp_path: Path, backend: Backend):
        for disable in ["disable", True]:
            fn = self.make_flat_cache_fn(backend, disable=disable)(tmp_path, True)

            assert fn([10, 5, 2]) == [100, 25, 4]
            fn.coefficient = 2
            assert fn([10, 5, 2]) == [200, 50, 8]

        fn = self.make_flat_cache_fn(backend, disable=False)(tmp_path, True)
        assert fn([10, 5, 2]) == [100, 25, 4]
        fn.coefficient = 2
        assert fn([10, 5, 2]) == [100, 25, 4]

from __future__ import annotations

import pickle
from dataclasses import dataclass
from inspect import Parameter
from pathlib import Path
from typing import List, Optional

import pytest
import serde.pickle

from ream.cache_helper import (
    Backend,
    Cache,
    Cacheable,
    CacheArgsHelper,
    DirBackend,
    FileBackend,
    SqliteBackend,
)
from ream.fs import FS
from ream.prelude import BaseActor, NoParams
from tests.test_helper import has_lz4


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
    @dataclass
    class Integer:
        value: int

    def make_cache_fn(self, backend: Backend, **kwargs):
        # special hack to overcome postinit limitation
        if hasattr(backend, "_is_postinited"):
            delattr(backend, "_is_postinited")

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

    def make_complex_cache_fn(self, backend: Backend, **kwargs):
        if hasattr(backend, "_is_postinited"):
            delattr(backend, "_is_postinited")

        class MyCacheFn(Cacheable):
            def __init__(self, workdir: FS | Path, disable: bool = False):
                super().__init__(workdir, disable)
                self.coefficient = 1

            @Cache.cache(
                backend=backend,
                **kwargs,
            )
            def __call__(self, val: TestCacheHelper.Integer) -> int:
                return (val.value**2) * self.coefficient

        return MyCacheFn

    def make_flat_cache_fn(self, backend: Backend, **kwargs):
        if hasattr(backend, "_is_postinited"):
            delattr(backend, "_is_postinited")

        class MyCacheFn(Cacheable):
            def __init__(self, workdir: FS | Path, disable: bool = False):
                super().__init__(workdir, disable)
                self.coefficient = 1

            @Cache.flat_cache(
                backend=backend,
                **kwargs,
            )
            def __call__(self, vals: list[int]) -> list[int]:
                return [(v**2) * self.coefficient for v in vals]

        return MyCacheFn

    def make_complex_flat_cache_fn(self, backend: Backend, **kwargs):
        if hasattr(backend, "_is_postinited"):
            delattr(backend, "_is_postinited")

        class MyCacheFn(Cacheable):
            def __init__(self, workdir: FS | Path, disable: bool = False):
                super().__init__(workdir, disable)
                self.coefficient = 1

            @Cache.flat_cache(
                backend=backend,
                **kwargs,
            )
            def __call__(self, vals: list[TestCacheHelper.Integer]) -> list[int]:
                return [(v.value**2) * self.coefficient for v in vals]

        return MyCacheFn

    @pytest.fixture(params=[None, "gz", "bz2"] + (["lz4"] if has_lz4() else []))
    def compression(self, request):
        return request.param

    @pytest.fixture(params=["file", "dir", "sqlite"])
    def backend(self, request, compression) -> Backend:
        if request.param == "file":
            return FileBackend(
                ser=pickle.dumps,
                deser=pickle.loads,
                fileext="pkl",
                compression=compression,
            )
        if request.param == "dir":
            return DirBackend(
                ser=lambda value, path, compression: serde.pickle.ser(
                    value,
                    path / f"data.pkl.{compression}"
                    if compression is not None
                    else path / "data.pkl",
                ),
                deser=lambda path, compression: serde.pickle.deser(
                    path / f"data.pkl.{compression}"
                    if compression is not None
                    else path / "data.pkl",
                ),
                compression=compression,
            )
        if request.param == "sqlite":
            return SqliteBackend(
                ser=pickle.dumps,
                deser=pickle.loads,
                compression=compression,
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

    def test_cache_args(self, tmp_path: Path, backend: Backend):
        # if we specify no argument will be used during caching
        # calling with different arguments will always return the same result
        fn = self.make_cache_fn(backend, cache_args=[])(tmp_path)
        assert fn(10) == 100
        assert fn(4) == 100

        # we also can specify the argument from self such as coefficient
        # doing so, the result will be different when the coefficient change
        for cache_self_args in [
            "coefficient",
            CacheArgsHelper.gen_cache_self_args("coefficient"),
        ]:
            fn = self.make_cache_fn(
                backend,
                cache_self_args=cache_self_args,
            )(tmp_path)
            assert fn(5) == 25
            fn.coefficient = 2
            assert fn(5) == 50

        # we can customize how to serialize each argument to
        # make it friendly with cache key. if we don't,
        # complex type will raise error
        with pytest.raises(TypeError):
            fn = self.make_complex_cache_fn(
                backend,
            )(tmp_path)

        fn = self.make_complex_cache_fn(
            backend,
            cache_key=lambda self, val: val.value.to_bytes(),
        )(tmp_path)
        assert fn(self.Integer(15)) == 225

        fn = self.make_complex_cache_fn(
            backend,
            cache_ser_args={"val": lambda val: val.value},
        )(tmp_path)
        assert fn(self.Integer(25)) == 625

    def test_flat_cache(self, tmp_path: Path, backend: Backend):
        fn = self.make_flat_cache_fn(backend)(tmp_path)

        assert fn([10, 5, 2]) == [100, 25, 4]
        fn.coefficient = 2
        assert fn([10, 5, 2]) == [100, 25, 4]

        assert fn([3, 4, 7]) == [18, 32, 98]
        fn.coefficient = 1
        assert fn([3, 4, 7]) == [18, 32, 98]

    def test_flat_cache_args(self, tmp_path: Path, backend: Backend):
        # if we specify no argument will be used during caching
        # calling with different arguments will always return the same result
        fn = self.make_flat_cache_fn(backend, cache_args=[])(tmp_path)
        assert fn([2, 4]) == [4, 4]
        assert fn([3, 5]) == [4, 4]

        # we also can specify the argument from self such as coefficient
        # doing so, the result will be different when the coefficient change
        for cache_self_args in [
            "coefficient",
            CacheArgsHelper.gen_cache_self_args("coefficient"),
        ]:
            fn = self.make_flat_cache_fn(
                backend,
                cache_self_args=cache_self_args,
            )(tmp_path)
            assert fn([7, 9]) == [49, 81]
            fn.coefficient = 2
            assert fn([7, 9]) == [98, 162]

        # we can customize how to serialize each argument to
        # make it friendly with cache key. if we don't,
        # complex type will raise error
        with pytest.raises(TypeError):
            fn = self.make_complex_flat_cache_fn(
                backend,
            )(tmp_path)

        fn = self.make_complex_flat_cache_fn(
            backend,
            cache_key=lambda self, vals: vals.value.to_bytes(),
        )(tmp_path)
        assert fn([self.Integer(11), self.Integer(13)]) == [121, 169]
        assert fn(vals=[self.Integer(15), self.Integer(17)]) == [15**2, 17**2]

        fn = self.make_complex_flat_cache_fn(
            backend,
            cache_ser_args={"vals": lambda val: val.value},
        )(tmp_path)
        assert fn([self.Integer(21), self.Integer(23)]) == [441, 529]
        assert fn(vals=[self.Integer(25), self.Integer(27)]) == [25**2, 27**2]

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

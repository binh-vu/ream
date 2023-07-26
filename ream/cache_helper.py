from __future__ import annotations

import bz2
import gzip
import pickle
from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import contextmanager
from functools import lru_cache, partial, wraps
from inspect import Parameter, signature
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import orjson
import serde.prelude as serde
from hugedict.misc import Chain2, identity
from hugedict.sqlitedict import SqliteDict, SqliteDictFieldType
from loguru import logger
from ream.fs import FS
from ream.helper import orjson_dumps
from serde.helper import AVAILABLE_COMPRESSIONS, JsonSerde
from timer import Timer
from typing_extensions import Self

try:
    import lz4.frame as lz4_frame  # type: ignore
except ImportError:
    lz4_frame = None

NoneType = type(None)
# arguments are (self, *args, **kwargs)
CacheKeyFn = Callable[..., bytes]

F = TypeVar("F")

if TYPE_CHECKING:
    from loguru import Logger


class JLSerdeCache:
    @staticmethod
    def file(
        cache_args: Optional[list[str]] = None,
        cache_self_args: Optional[Callable[..., dict]] = None,
        cache_key: Optional[CacheKeyFn] = None,
        filename: Optional[Union[str, Callable[..., str]]] = None,
        compression: Optional[Literal["gz", "bz2", "lz4"]] = None,
        mem_persist: bool = False,
        cache_attr: str = "_cache",
        cls: Optional[Type[JsonSerde]] = None,
        log_serde_time: bool = False,
        disable: bool | str | Callable[[Any], bool] = False,
    ):
        return Cache.file(
            ser=serde.jl.ser,
            deser=partial(serde.jl.deser, cls=cls),  # type: ignore
            cache_args=cache_args,
            cache_self_args=cache_self_args,
            cache_key=cache_key,
            filename=filename,
            compression=compression,
            mem_persist=mem_persist,
            cache_attr=cache_attr,
            fileext="jl",
            log_serde_time=log_serde_time,
            disable=disable,
        )


class PickleSerdeCache:
    @staticmethod
    def file(
        cache_args: Optional[list[str]] = None,
        cache_self_args: Optional[Callable[..., dict]] = None,
        cache_key: Optional[CacheKeyFn] = None,
        filename: Optional[Union[str, Callable[..., str]]] = None,
        compression: Optional[Literal["gz", "bz2", "lz4"]] = None,
        mem_persist: bool = False,
        cache_attr: str = "_cache",
        log_serde_time: bool = False,
        disable: bool | str | Callable[[Any], bool] = False,
    ):
        return Cache.file(
            ser=serde.pickle.ser,
            deser=serde.pickle.deser,
            cache_args=cache_args,
            cache_self_args=cache_self_args,
            cache_key=cache_key,
            filename=filename,
            compression=compression,
            mem_persist=mem_persist,
            cache_attr=cache_attr,
            fileext="pkl",
            log_serde_time=log_serde_time,
            disable=disable,
        )

    @staticmethod
    def sqlite(
        cache_args: Optional[list[str]] = None,
        cache_self_args: Optional[Callable[..., dict]] = None,
        cache_key: Optional[CacheKeyFn] = None,
        compression: Optional[Literal["gz", "bz2", "lz4"]] = None,
        mem_persist: bool = False,
        cache_attr: str = "_cache",
        log_serde_time: bool = False,
        disable: bool | str | Callable[[Any], bool] = False,
    ):
        return Cache.sqlite(
            ser=pickle.dumps,
            deser=pickle.loads,
            cache_args=cache_args,
            cache_self_args=cache_self_args,
            cache_key=cache_key,
            compression=compression,
            mem_persist=mem_persist,
            cache_attr=cache_attr,
            log_serde_time=log_serde_time,
            disable=disable,
        )


class ClsSerdeCache:
    """A cache that uses the save/load methods of a class as deser/ser functions."""

    @staticmethod
    def file(
        cls: type[SaveLoadProtocol] | Sequence[type[SaveLoadProtocol]],  # type: ignore
        cache_args: Optional[list[str]] = None,
        cache_self_args: Optional[Callable[..., dict]] = None,
        cache_key: Optional[CacheKeyFn] = None,
        filename: Optional[Union[str, Callable[..., str]]] = None,
        compression: Optional[Literal["gz", "bz2", "lz4"]] = None,
        mem_persist: bool = False,
        cache_attr: str = "_cache",
        fileext: Optional[str | list[str]] = None,
        log_serde_time: bool = False,
        disable: bool | str | Callable[[Any], bool] = False,
    ):
        if isinstance(cls, Sequence):
            if fileext is None:
                exts = None
            else:
                if isinstance(fileext, list):
                    exts = fileext
                else:
                    exts = [fileext] * len(cls)
                fileext = "tuple"
            obj = ClsSerdeCache.get_tuple_serde(cls, exts)
            ser = obj["ser"]
            deser = obj["deser"]
        else:
            assert fileext is None or isinstance(fileext, str)
            obj = ClsSerdeCache.get_serde(cls)
            ser = obj["ser"]
            deser = obj["deser"]

        return Cache.file(
            ser=ser,
            deser=deser,
            cache_args=cache_args,
            cache_self_args=cache_self_args,
            cache_key=cache_key,
            filename=filename,
            compression=compression,
            mem_persist=mem_persist,
            cache_attr=cache_attr,
            fileext=fileext,
            log_serde_time=log_serde_time,
            disable=disable,
        )

    @staticmethod
    def dir(
        cls: type[SaveLoadDirProtocol] | Sequence[type[SaveLoadDirProtocol]],  # type: ignore
        cache_args: Optional[list[str]] = None,
        cache_self_args: Optional[Callable[..., dict]] = None,
        cache_key: Optional[CacheKeyFn] = None,
        dirname: Optional[Union[str, Callable[..., str]]] = None,
        compression: Optional[Literal["gz", "bz2", "lz4"]] = None,
        mem_persist: bool = False,
        cache_attr: str = "_cache",
        log_serde_time: bool = False,
        disable: bool | str | Callable[[Any], bool] = False,
    ):
        if isinstance(cls, Sequence):
            obj = ClsSerdeCache.get_tuple_serde(cls, None)
            ser = obj["ser"]
            deser = obj["deser"]
        else:
            obj = ClsSerdeCache.get_serde(cls)
            ser = obj["ser"]
            deser = obj["deser"]

        return Cache.dir(
            ser=ser,
            deser=deser,
            cache_args=cache_args,
            cache_self_args=cache_self_args,
            cache_key=cache_key,
            dirname=dirname,
            compression=compression,
            mem_persist=mem_persist,
            cache_attr=cache_attr,
            log_serde_time=log_serde_time,
            disable=disable,
        )

    @staticmethod
    def get_serde(
        klass: Union[
            type[SaveLoadProtocol],
            type[SaveLoadDirProtocol],
        ]
    ):
        def ser(
            item: Union[SaveLoadProtocol, SaveLoadDirProtocol],
            file: Path,
            *args,
        ):
            return item.save(file, *args)

        def deser(file: Path, *args):
            return klass.load(file, *args)

        return {"ser": ser, "deser": deser}

    @staticmethod
    def get_tuple_serde(
        classes: Sequence[
            Union[
                type[SaveLoadProtocol],
                type[SaveLoadDirProtocol],
            ]
        ],
        exts: Optional[list[str]] = None,
    ):
        def ser(
            items: Sequence[Optional[SaveLoadProtocol | SaveLoadDirProtocol]],
            file: Path,
            *args,
        ):
            for i, item in enumerate(items):
                if item is not None:
                    ifile = file / (f"_{i}.{exts[i]}" if exts is not None else f"_{i}")
                    item.save(ifile, *args)
            file.touch()

        def deser(file: Path, *args):
            output = []
            for i, cls in enumerate(classes):
                ifile = file / (f"_{i}.{exts[i]}" if exts is not None else f"_{i}")
                if ifile.exists():
                    output.append(cls.load(ifile, *args))
                else:
                    output.append(None)
            return tuple(output)

        return {
            "ser": ser,
            "deser": deser,
        }


class Cache:
    jl = JLSerdeCache
    pickle = PickleSerdeCache
    cls = ClsSerdeCache

    @staticmethod
    @contextmanager
    def autoclear_mem_cache(
        objs: Union[object, list[object]], cache_attr: str = "_cache"
    ):
        yield None
        for obj in objs if isinstance(objs, list) else [objs]:
            if hasattr(obj, cache_attr):
                getattr(obj, cache_attr).clear()

    @staticmethod
    def mem(
        cache_args: Optional[list[str]] = None,
        cache_self_args: Optional[Callable[..., dict]] = None,
        cache_key: Optional[CacheKeyFn | Callable[..., tuple]] = None,
        cache_attr: str = "_cache",
        disable: bool | str | Callable[[Any], bool] = False,
    ) -> Callable[[F], F]:
        """Decorator to cache the result of a function to an attribute in the instance in memory.

        Note: It does not support function with variable number of arguments.

        Args:
            cache_args: list of arguments to use for the default cache key function. If None, all arguments are used.
            cache_self_args: extra arguments that are derived from the instance to use for the default cache key
                function. If cache_key is provided this argument is ignored.
            cache_key: Function to use to generate the cache key. If None, the default is used. The default function
                only support arguments of types str, int, bool, and None.
            cache_attr: Name of the attribute to use to store the cache in the instance.
            disable: if True (either bool or an attribute (bool type) or a function called with self returning bool), the cache is disabled.
        """
        if isinstance(disable, bool) and disable:
            return identity

        def wrapper_fn(func):
            func_name = func.__name__
            cache_args_helper = CacheArgsHelper(func)
            if cache_args is not None:
                cache_args_helper.keep_args(cache_args)
            if cache_self_args is not None:
                cache_args_helper.set_self_args(cache_self_args)

            keyfn = cache_key
            if keyfn is None:
                cache_args_helper.ensure_auto_cache_key_friendly()
                keyfn = (
                    lambda self, *args, **kwargs: cache_args_helper.get_args_as_tuple(
                        self, *args, **kwargs
                    )
                )

            @wraps(func)
            def fn(self, *args, **kwargs):
                if not isinstance(disable, bool):
                    is_disable = (
                        getattr(self, disable)
                        if isinstance(disable, str)
                        else disable(self)
                    )
                    if is_disable:
                        return func(self, *args, **kwargs)

                if not hasattr(self, cache_attr):
                    setattr(self, cache_attr, {})
                cache = getattr(self, cache_attr)
                key = (func_name, keyfn(self, *args, **kwargs))
                if key not in cache:
                    cache[key] = func(self, *args, **kwargs)
                return cache[key]

            return fn

        return wrapper_fn  # type: ignore

    @staticmethod
    def file(
        ser: Callable[[Any, Path], None],
        deser: Callable[[Path], Any],
        cache_args: Optional[list[str]] = None,
        cache_self_args: Optional[Callable[..., dict]] = None,
        cache_key: Optional[CacheKeyFn] = None,
        filename: Optional[Union[str, Callable[..., str]]] = None,
        compression: Optional[Literal["gz", "bz2", "lz4"]] = None,
        mem_persist: bool = False,
        cache_attr: str = "_cache",
        fileext: Optional[str] = None,
        log_serde_time: bool = False,
        disable: bool | str | Callable[[Any], bool] = False,
    ) -> Callable[[F], F]:
        """Decorator to cache the result of a function to a file. The function must
        be a method of a class that has trait `HasWorkingFsTrait` so that we can determine
        the directory to store the cached file.

        Note: It does not support function with variable number of arguments.

        Args:
            ser: A function to serialize the output of the function to a file.
            deser: A function to deserialize the output of the function from a file.
            cache_args: list of arguments to use for the default cache key function. If None, all arguments are used. If cache_key is provided
                this argument is ignored.
            cache_self_args: extra arguments that are derived from the instance to use for the default cache key
                function. If cache_key is provided this argument is ignored.
            cache_key: Function to use to generate the cache key. If None, the default is used. The default function
                only support arguments of types str, int, bool, and None.
            filename: Filename to use for the cache file. If None, the name of the function is used. If it is a function,
                it will be called with the arguments of the function to generate the filename.
            compression: whether to compress the cache file, the compression is detected via the file extension. Therefore,
                this option has no effect if filename is provided. Moreover, when filename is a function, it cannot check
                if the filename has the correct extension.
            mem_persist: If True, the cache will also be stored in memory. This is a combination of mem and file cache.
            cache_attr: Name of the attribute to use to store the cache in the instance.
            fileext: Extension of the file to use if the filename is None (the function name is used as the filename).
            log_serde_time: if True, will log the time it takes to deserialize the cache file.
            disable: if True (either bool or an attribute (bool type) or a function called with self returning bool), the cache is disabled.
        """
        if isinstance(disable, bool) and disable:
            return identity

        def wrapper_fn(func):
            if filename is None:
                filename2 = func.__name__
                if fileext is not None:
                    filename2 += f".{fileext}"
                if compression is not None:
                    filename2 += f".{compression}"
            else:
                filename2 = filename
                if isinstance(filename2, str) and compression is not None:
                    assert filename2.endswith(compression)

            cache_args_helper = CacheArgsHelper(func)
            if cache_args is not None:
                cache_args_helper.keep_args(cache_args)

            if cache_self_args is not None:
                cache_args_helper.set_self_args(cache_self_args)

            keyfn = cache_key
            if keyfn is None:
                cache_args_helper.ensure_auto_cache_key_friendly()
                keyfn = lambda self, *args, **kwargs: orjson_dumps(
                    cache_args_helper.get_args(self, *args, **kwargs)
                )

            @wraps(func)
            def fn(self: HasWorkingFsTrait, *args, **kwargs):
                if not isinstance(disable, bool):
                    is_disable = (
                        getattr(self, disable)
                        if isinstance(disable, str)
                        else disable(self)
                    )
                    if is_disable:
                        return func(self, *args, **kwargs)

                fs = self.get_working_fs()

                if isinstance(filename2, str):
                    cache_filename = filename2
                else:
                    cache_filename = filename2(self, *args, **kwargs)

                cache_file = fs.get(
                    cache_filename, key=keyfn(self, *args, **kwargs), save_key=True
                )

                if not cache_file.exists():
                    output = func(self, *args, **kwargs)
                    with fs.acquire_write_lock(), cache_file.reserve_and_track() as fpath:
                        if log_serde_time:
                            with Timer().watch_and_report(
                                f"serialize file {cache_file._realdiskpath}",
                                self.logger.debug,
                            ):
                                ser(output, fpath)
                        else:
                            ser(output, fpath)
                else:
                    if log_serde_time:
                        with Timer().watch_and_report(
                            f"deserialize file {cache_file._realdiskpath}",
                            self.logger.debug,
                        ):
                            output = deser(cache_file.get())
                    else:
                        output = deser(cache_file.get())
                return output

            if mem_persist:
                return Cache.mem(
                    cache_args=cache_args,
                    cache_self_args=cache_self_args,
                    cache_key=cache_key,
                    cache_attr=cache_attr,
                )(fn)
            return fn

        return wrapper_fn  # type: ignore

    @staticmethod
    def dir(
        ser: Callable[[Any, Path, Optional[Literal["gz", "bz2", "lz4"]]], None],
        deser: Callable[[Path, Optional[Literal["gz", "bz2", "lz4"]]], Any],
        cache_args: Optional[list[str]] = None,
        cache_self_args: Optional[Callable[..., dict]] = None,
        cache_key: Optional[CacheKeyFn] = None,
        dirname: Optional[Union[str, Callable[..., str]]] = None,
        compression: Optional[Literal["gz", "bz2", "lz4"]] = None,
        mem_persist: bool = False,
        cache_attr: str = "_cache",
        log_serde_time: bool = False,
        disable: bool | str | Callable[[Any], bool] = False,
    ) -> Callable[[F], F]:
        """Decorator to cache the result of a function to files in a directory (each cache key use different directory). The function must
        be a method of a class that has trait `HasWorkingFsTrait` so that we can determine
        the directory to store the cached directory. This is useful when the result of the function is serialized to multiple files and need to be
        put under the same directory.

        Note: It does not support function with variable number of arguments.

        Args:
            ser: A function to serialize the output of the function to a file.
            deser: A function to deserialize the output of the function from a file.
            cache_args: list of arguments to use for the default cache key function. If None, all arguments are used. If cache_key is provided
                this argument is ignored.
            cache_self_args: extra arguments that are derived from the instance to use for the default cache key
                function. If cache_key is provided this argument is ignored.
            cache_key: Function to use to generate the cache key. If None, the default is used. The default function
                only support arguments of types str, int, bool, and None.
            dirname: Directory name to use for the cache file. If None, the name of the function is used. If it is a function,
                it will be called with the arguments of the function to generate the filename.
            compression: whether to compress the cache file, the compression is detected via the file extension. Therefore,
                this option has no effect if filename is provided. Moreover, when filename is a function, it cannot check
                if the filename has the correct extension.
            mem_persist: If True, the cache will also be stored in memory. This is a combination of mem and file cache.
            cache_attr: Name of the attribute to use to store the cache in the instance.
            fileext: Extension of the file to use if the filename is None (the function name is used as the filename).
            log_serde_time: if True, will log the time it takes to deserialize the cache file.
            disable: if True (either bool or an attribute (bool type) or a function called with self returning bool), the cache is disabled.
        """
        if isinstance(disable, bool) and disable:
            return identity

        def wrapper_fn(func):
            if dirname is None:
                dirname2 = func.__name__
            else:
                dirname2 = dirname

            cache_args_helper = CacheArgsHelper(func)
            if cache_args is not None:
                cache_args_helper.keep_args(cache_args)

            if cache_self_args is not None:
                cache_args_helper.set_self_args(cache_self_args)

            keyfn = cache_key
            if keyfn is None:
                cache_args_helper.ensure_auto_cache_key_friendly()
                keyfn = lambda self, *args, **kwargs: orjson_dumps(
                    cache_args_helper.get_args(self, *args, **kwargs)
                )

            @wraps(func)
            def fn(self: HasWorkingFsTrait, *args, **kwargs):
                if not isinstance(disable, bool):
                    is_disable = (
                        getattr(self, disable)
                        if isinstance(disable, str)
                        else disable(self)
                    )
                    if is_disable:
                        return func(self, *args, **kwargs)

                fs = self.get_working_fs()

                if isinstance(dirname2, str):
                    cache_dirname = dirname2
                else:
                    cache_dirname = dirname2(self, *args, **kwargs)

                cache_file = fs.get(
                    cache_dirname, key=keyfn(self, *args, **kwargs), save_key=True
                )

                if not cache_file.exists():
                    output = func(self, *args, **kwargs)
                    with fs.acquire_write_lock(), cache_file.reserve_and_track() as fpath:
                        if log_serde_time:
                            with Timer().watch_and_report(
                                f"serialize file {cache_file._realdiskpath}",
                                self.logger.debug,
                            ):
                                ser(output, fpath, compression)
                        else:
                            ser(output, fpath, compression)
                else:
                    if log_serde_time:
                        with Timer().watch_and_report(
                            f"deserialize file {cache_file._realdiskpath}",
                            self.logger.debug,
                        ):
                            output = deser(cache_file.get(), compression)
                    else:
                        output = deser(cache_file.get(), compression)
                return output

            if mem_persist:
                return Cache.mem(
                    cache_args=cache_args,
                    cache_self_args=cache_self_args,
                    cache_key=cache_key,
                    cache_attr=cache_attr,
                )(fn)
            return fn

        return wrapper_fn  # type: ignore

    @staticmethod
    def sqlite(
        ser: Callable[[Any], bytes],
        deser: Callable[[bytes], Any],
        cache_args: Optional[list[str]] = None,
        cache_self_args: Optional[Callable[..., dict]] = None,
        cache_key: Optional[CacheKeyFn] = None,
        compression: Optional[Literal["gz", "bz2", "lz4"]] = None,
        mem_persist: bool = False,
        cache_attr: str = "_cache",
        log_serde_time: bool = False,
        disable: bool | str | Callable[[Any], bool] = False,
    ) -> Callable[[F], F]:
        """Decorator to cache the result of a function to a record in a sqlite database. The function must
        be a method of a class that has trait `HasWorkingFsTrait` so that we can determine
        the directory to store the sqlite database.

        Note: It does not support function with variable number of arguments.

        Args:
            ser: A function to serialize the output of the function to bytes.
            deser: A function to deserialize the output of the function from bytes.
            cache_args: list of arguments to use for the default cache key function. If None, all arguments are used. If cache_key is provided
                this argument is ignored.
            cache_self_args: extra arguments that are derived from the instance to use for the default cache key
                function. If cache_key is provided this argument is ignored.
            cache_key: Function to use to generate the cache key. If None, the default is used. The default function
                only support arguments of types str, int, bool, and None.
            compression: whether to compress the binary.
            mem_persist: If True, the cache will also be stored in memory. This is a combination of mem and file cache.
            cache_attr: Name of the attribute to use to store the cache in the instance.
            log_serde_time: if True, will log the time it takes to fetch and deserialize the binary data.
            disable: if True (either bool or an attribute (bool type) or a function called with self returning bool), the cache is disabled.
        """
        if isinstance(disable, bool) and disable:
            return identity

        if compression == "gz":
            ser = lambda x: gzip.compress(ser(x), mtime=0)
            deser = lambda x: deser(gzip.decompress(x))
        elif compression == "bz2":
            ser = lambda x: bz2.compress(ser(x))
            deser = lambda x: deser(bz2.decompress(x))
        elif compression == "lz4":
            if lz4_frame is None:
                raise ValueError("lz4 is not installed")
            # using lambda somehow terminate the program without raising an error
            ser = Chain2(lz4_frame.compress, ser)
            deser = Chain2(deser, lz4_frame.decompress)

        def wrapper_fn(func):
            cache_args_helper = CacheArgsHelper(func)
            if cache_args is not None:
                cache_args_helper.keep_args(cache_args)

            if cache_self_args is not None:
                cache_args_helper.set_self_args(cache_self_args)

            keyfn = cache_key
            if keyfn is None:
                cache_args_helper.ensure_auto_cache_key_friendly()
                keyfn = lambda self, *args, **kwargs: orjson_dumps(
                    cache_args_helper.get_args(self, *args, **kwargs)
                )

            fname = func.__name__
            dbname = f"{fname}.db"
            dbattr = f"__sqlite_{fname}"

            @wraps(func)
            def fn(self: HasWorkingFsTrait, *args, **kwargs):
                if not isinstance(disable, bool):
                    is_disable = (
                        getattr(self, disable)
                        if isinstance(disable, str)
                        else disable(self)
                    )
                    if is_disable:
                        return func(self, *args, **kwargs)

                fs = self.get_working_fs()
                if not hasattr(self, dbattr):
                    sqlitedict = SqliteDict(
                        fs.root / dbname,
                        keytype=SqliteDictFieldType.bytes,
                        ser_value=identity,
                        deser_value=identity,
                    )
                    setattr(self, dbattr, sqlitedict)
                else:
                    sqlitedict = getattr(self, dbattr)

                key = keyfn(self, *args, **kwargs)

                if key not in sqlitedict:
                    output = func(self, *args, **kwargs)
                    if log_serde_time:
                        timer = Timer()
                        with timer.watch_and_report(
                            f"[{fname}] serialize output", self.logger.debug
                        ):
                            ser_output = ser(output)
                        with timer.watch_and_report(
                            f"[{fname}] save to db", self.logger.debug
                        ):
                            sqlitedict[key] = ser_output
                    else:
                        sqlitedict[key] = ser(output)
                else:
                    if log_serde_time:
                        timer = Timer()
                        with timer.watch_and_report(
                            f"[{fname}] load from db", self.logger.debug
                        ):
                            ser_output = sqlitedict[key]
                        with timer.watch_and_report(
                            f"[{fname}] deserialize output", self.logger.debug
                        ):
                            output = deser(ser_output)
                    else:
                        output = deser(sqlitedict[key])
                return output

            if mem_persist:
                return Cache.mem(
                    cache_args=cache_args,
                    cache_self_args=cache_self_args,
                    cache_key=cache_key,
                    cache_attr=cache_attr,
                )(fn)
            return fn

        return wrapper_fn  # type: ignore


class CacheArgsHelper:
    """Helper to working with arguments of a function. This class ensures
    that we can select a subset of arguments to use for the cache key, and
    to always put the calling arguments in the same declared order.
    """

    def __init__(self, func: Callable):
        self.args: dict[str, Parameter] = {}
        try:
            self.argtypes: dict[str, Optional[Type]] = get_type_hints(func)
        except TypeError:
            logger.error(
                "Cannot get type hints for function {}. "
                "If this is due to eval function, it's mean that the type is incorrect (i.e., incorrect Python's code). "
                "For example, we have a hugedict.prelude.RocksDBDict class, which is a class built from Rust (Python's extension module), "
                "the class is not a generic class, but we have a .pyi file that declare it as a generic class (cheating). This works fine"
                "for pylance and mypy checker, but it will cause error when we try to get type hints because the class is not subscriptable.",
                func,
            )
            raise
        for name, param in signature(func).parameters.items():
            self.args[name] = param
            if name not in self.argtypes:
                self.argtypes[name] = None

        assert (
            next(iter(self.args)) == "self"
        ), "The first argument of the method must be self, an instance of BaseActor"
        self.args.pop("self")
        self.argnames: list[str] = list(self.args.keys())
        self.cache_args = self.argnames
        self.cache_self_args = None

    def keep_args(self, names: Iterable[str]) -> None:
        self.cache_args = list(names)

    def set_self_args(self, self_args: Optional[Callable[..., dict]]) -> None:
        self.cache_self_args = self_args

    def get_cache_argtypes(self) -> dict[str, Optional[Type]]:
        return {name: self.argtypes[name] for name in self.cache_args}

    def ensure_auto_cache_key_friendly(self):
        for name in self.cache_args:
            param = self.args[name]
            if (
                param.kind == Parameter.VAR_KEYWORD
                or param.kind == Parameter.VAR_POSITIONAL
            ):
                raise TypeError(
                    f"Variable arguments are not supported for automatically generating caching key to cache function call. Found one with name: {name}"
                )

            argtype = self.argtypes[name]
            if argtype is None:
                raise TypeError(
                    f"Automatically generating caching key to cache a function call requires all arguments to be annotated. Found one without annotation: {name}"
                )
            origin = get_origin(argtype)
            if origin is None:
                if (
                    not issubclass(argtype, (str, int, bool))
                    and argtype is not NoneType
                ):
                    raise TypeError(
                        f"Automatically generating caching key to cache a function call requires all arguments to be one of type: str, int, bool, or None. Found {name} with type {argtype}"
                    )
            elif origin is Union:
                args = get_args(argtype)
                if any(
                    not issubclass(a, (str, int, bool)) and a is not NoneType
                    for a in args
                ):
                    raise TypeError(
                        f"Automatically generating caching key to cache a function call requires all arguments to be one of type: str, int, bool, or None. Found {name} with type {argtype}"
                    )
            elif origin is Literal:
                args = get_args(argtype)
                if any(
                    not isinstance(a, (str, int, bool)) and a is not NoneType
                    for a in args
                ):
                    raise TypeError(
                        f"Automatically generating caching key to cache a function call requires all arguments to be one of type: str, int, bool, None, or Literal with values of those types. Found {name} with type {argtype}"
                    )
            else:
                raise TypeError(
                    f"Automatically generating caching key to cache a function call requires all arguments to be one of type: str, int, bool, or None. Found {name} with type {argtype}"
                )

    def get_args(self, obj: HasWorkingFsTrait, *args, **kwargs) -> dict:
        # TODO: improve me!
        out = {name: value for name, value in zip(self.args, args)}
        out.update(
            [
                (name, kwargs.get(name, self.args[name].default))
                for name in self.argnames[len(args) :]
            ]
        )
        if len(self.cache_args) != len(self.argnames):
            out = {name: out[name] for name in self.cache_args}

        if self.cache_self_args is not None:
            out.update(self.cache_self_args(obj))
        return out

    def get_args_as_tuple(self, obj: HasWorkingFsTrait, *args, **kwargs) -> tuple:
        # TODO: improve me!
        return tuple(self.get_args(obj, *args, **kwargs).values())

    @staticmethod
    def gen_cache_self_args(
        *attrs: Union[str, Callable[[Any], Union[str, bool, int, None]]]
    ):
        """Generate a function that returns a dictionary of arguments extracted from self.

        Args:
            *attrs: a list of attributes of self to be extracted.
                - If an attribute is a string, it is property of self, and the value is obtained by getattr(self, attr).
                - If an attribute is a callable, it is a no-argument method of self, and the value is obtained by
                  attr(self). To specify a method of self in the decorator, just use `method_a` instead of `Class.method_a`,
                  and the method must be defined before the decorator is called.
        """
        props = [attr for attr in attrs if isinstance(attr, str)]
        funcs = [attr for attr in attrs if callable(attr)]

        def get_self_args(self):
            args = {name: getattr(self, name) for name in props}
            args.update({func.__name__: func(self) for func in funcs})
            return args

        return get_self_args


class HasWorkingFsTrait(Protocol):
    logger: Logger

    def get_working_fs(self) -> FS:
        ...


class Cacheable:
    """A class that implement HasWorkingFSTrait so it can be used with @Cache.file decorator.

    Args:
        workdir: directory to store cached files.
        disable: set to True to disable caching. The wrapper method needs to use this flag to be effective!
    """

    def __init__(self, workdir: Union[FS, Path], disable: bool = False):
        self.workdir = FS(workdir) if isinstance(workdir, Path) else workdir
        self.logger = logger.bind(name=self.__class__.__name__)

        self.disable = disable

    def get_working_fs(self) -> FS:
        return self.workdir


class CacheableFn(ABC, Cacheable):
    """This utility provides a way to break a giantic function into smaller pieces that can be cached individually."""

    def __init__(
        self, use_args: list[str], workdir: Union[FS, Path], disable: bool = False
    ):
        super().__init__(workdir, disable)
        self.use_args = use_args

    @staticmethod
    def get_cache_key(slf: CacheableFn, args: F) -> bytes:
        return orjson.dumps(
            {attr: getattr(args, attr) for attr in slf.get_use_args()},
            option=orjson.OPT_SORT_KEYS | orjson.OPT_SERIALIZE_DATACLASS,
        )

    @lru_cache()
    def get_use_args(self) -> set[str]:
        cache_args = set()
        for fn in self.get_dependable_fns():
            cache_args.update(fn.get_use_args())
        return cache_args

    @lru_cache()
    def get_dependable_fns(self) -> list[CacheableFn]:
        fns = []
        for obj in self.__dict__.values():
            if isinstance(obj, CacheableFn):
                fns.append(obj)
        return fns

    @abstractmethod
    def __call__(self, args) -> Any:
        """This is where to put the function body. To cache it, wraps it with @Cache.<X> decorators"""
        ...


class SaveLoadProtocol(Protocol):
    def save(self, file: Path) -> None:
        ...

    @classmethod
    def load(cls, file: Path) -> Self:
        ...


class SaveLoadDirProtocol(Protocol):
    def save(
        self, dir: Path, compression: Optional[AVAILABLE_COMPRESSIONS] = None
    ) -> None:
        ...

    @classmethod
    def load(
        cls, dir: Path, compression: Optional[AVAILABLE_COMPRESSIONS] = None
    ) -> Self:
        ...


def unwrap_cache_decorators(cls: type, methods: list[str] | None = None):
    """Decorator to disable caching decorator by unwrap all methods of a class."""
    for method in methods or dir(cls):
        fn = getattr(cls, method)
        iswrapped = hasattr(fn, "__wrapped__")
        while hasattr(fn, "__wrapped__"):
            fn = getattr(fn, "__wrapped__")  # type: ignore
        if iswrapped:
            setattr(cls, method, fn)

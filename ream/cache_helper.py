from __future__ import annotations

import functools
from inspect import Parameter, signature
from collections.abc import Sequence
from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    Optional,
    Protocol,
    Type,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    Sequence,
    TYPE_CHECKING,
)
from typing_extensions import Self
from loguru import logger
from ream.fs import FS

import serde.prelude as serde
from timer import Timer
from ream.helper import orjson_dumps
from serde.helper import JsonSerde

NoneType = type(None)
# arguments are (self, *args, **kwargs)
CacheKeyFn = Callable[..., bytes]

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
        disable: bool = False,
    ):
        return Cache.file(
            ser=serde.jl.ser,
            deser=functools.partial(serde.jl.deser, cls=cls),
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
        disable: bool = False,
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


class ClsSerdeCache:
    """A cache that uses the save/load methods of a class as deser/ser functions."""

    @staticmethod
    def file(
        cls: type[SaveLoadProtocol]  # type: ignore
        | type[SaveLoadCompressedProtocol]
        | Sequence[type[SaveLoadProtocol] | type[SaveLoadCompressedProtocol]],
        cache_args: Optional[list[str]] = None,
        cache_self_args: Optional[Callable[..., dict]] = None,
        cache_key: Optional[CacheKeyFn] = None,
        filename: Optional[Union[str, Callable[..., str]]] = None,
        compression: Optional[Literal["gz", "bz2", "lz4"]] = None,
        mem_persist: bool = False,
        cache_attr: str = "_cache",
        fileext: Optional[str | list[str]] = None,
        log_serde_time: bool = False,
        disable: bool = False,
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
    def get_serde(
        klass: Union[type[SaveLoadProtocol], type[SaveLoadCompressedProtocol]]
    ):
        def ser(item: Union[SaveLoadProtocol, SaveLoadCompressedProtocol], file: Path):
            return item.save(file)

        def deser(file: Path):
            return klass.load(file)

        return {"ser": ser, "deser": deser}

    @staticmethod
    def get_tuple_serde(
        classes: Sequence[
            Union[type[SaveLoadProtocol], type[SaveLoadCompressedProtocol]]
        ],
        exts: Optional[list[str]] = None,
    ):
        def ser(items: Sequence[Optional[SaveLoadProtocol]], file: Path):
            for i, item in enumerate(items):
                if item is not None:
                    ifile = file.parent / (
                        file.name + f"_{i}.{exts[i]}" if exts is not None else f"_{i}"
                    )
                    item.save(ifile)
            file.touch()

        def deser(file: Path):
            output = []
            for i, cls in enumerate(classes):
                ifile = file.parent / (
                    file.name + f"_{i}.{exts[i]}" if exts is not None else f"_{i}"
                )
                if ifile.exists():
                    output.append(cls.load(ifile))
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
    def mem(
        cache_args: Optional[list[str]] = None,
        cache_self_args: Optional[Callable[..., dict]] = None,
        cache_key: Optional[CacheKeyFn] = None,
        cache_attr: str = "_cache",
        disable: bool = False,
    ):
        """Decorator to cache the result of a function to an attribute in the instance in memory.

        Note: It does not support function with variable number of arguments.

        Args:
            cache_args: list of arguments to use for the default cache key function. If None, all arguments are used.
            cache_self_args: extra arguments that are derived from the instance to use for the default cache key
                function. If cache_key is provided this argument is ignored.
            cache_key: Function to use to generate the cache key. If None, the default is used. The default function
                only support arguments of types str, int, bool, and None.
            cache_attr: Name of the attribute to use to store the cache in the instance.
            disable: If True, the cache is disabled.
        """
        if disable:
            return lambda func: func

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

            @functools.wraps(func)
            def fn(self, *args, **kwargs):
                if not hasattr(self, cache_attr):
                    setattr(self, cache_attr, {})
                cache = getattr(self, cache_attr)
                key = (func_name, keyfn(self, *args, **kwargs))
                if key not in cache:
                    cache[key] = func(self, *args, **kwargs)
                return cache[key]

            return fn

        return wrapper_fn

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
        disable: bool = False,
    ) -> Callable:
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
            disable: if True, the cache is disabled.
        """
        if disable:
            return lambda func: func

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

            @functools.wraps(func)
            def fn(self: HasWorkingFsTrait, *args, **kwargs):
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
                                f"serialize file {cache_file.relpath}",
                                self.logger.debug,
                            ):
                                ser(output, fpath)
                        else:
                            ser(output, fpath)
                else:
                    if log_serde_time:
                        with Timer().watch_and_report(
                            f"deserialize file {cache_file.relpath}", self.logger.debug
                        ):
                            output = deser(cache_file.get())
                    else:
                        output = deser(cache_file.get())
                return output

            if mem_persist is not None:
                return Cache.mem(
                    cache_args=cache_args,
                    cache_self_args=cache_self_args,
                    cache_key=cache_key,
                    cache_attr=cache_attr,
                )(fn)
            return fn

        return wrapper_fn


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
    """A class that implement HasWorkingFSTrait so it can be used with @Cache.file decorator."""

    def __init__(self, workdir: Path):
        self.workdir = FS(workdir)
        self.logger = logger.bind(name=self.__class__.__name__)

    def get_working_fs(self) -> FS:
        return self.workdir


class SaveLoadProtocol(Protocol):
    def save(self, file: Path) -> None:
        ...

    @classmethod
    def load(cls, file: Path) -> Self:
        ...


class SaveLoadCompressedProtocol(Protocol):
    def save(self, file: Path, compression: Optional[str] = None) -> None:
        ...

    @classmethod
    def load(cls, file: Path, compression: Optional[str] = None) -> Self:
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

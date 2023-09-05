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
    Generic,
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
from loguru import logger
from serde.helper import (
    AVAILABLE_COMPRESSIONS,
    DEFAULT_ORJSON_OPTS,
    JsonSerde,
    _orjson_default,
    orjson_dumps,
)
from timer import Timer
from typing_extensions import Self

from hugedict.misc import Chain2, identity
from hugedict.sqlitedict import SqliteDict, SqliteDictFieldType
from ream.fs import FS
from ream.helper import ContextContainer, orjson_dumps

try:
    import lz4.frame as lz4_frame  # type: ignore
except ImportError:
    lz4_frame = None

NoneType = type(None)
# arguments are (self, *args, **kwargs)
CacheKeyFn = Callable[..., bytes]
ArgSer = Callable[[Any], Optional[Union[str, int, bool]]]

T = TypeVar("T")
F = TypeVar("F")
Value = Any
ARGS = Any


if TYPE_CHECKING:
    from loguru import Logger


class SqliteBackendFactory:
    @staticmethod
    def pickle(
        compression: Optional[AVAILABLE_COMPRESSIONS] = None,
        mem_persist: Optional[Union[MemBackend, bool]] = None,
        log_serde_time: bool = False,
    ):
        backend = SqliteBackend(
            ser=pickle.dumps,
            deser=pickle.loads,
            compression=compression,
        )
        return wrap_backend(backend, mem_persist, log_serde_time)


class ClsSerdeBackendFactory:
    """A cache that uses the save/load methods of a class as deser/ser functions."""

    @staticmethod
    def file(
        cls: type[SerdeProtocol],  # type: ignore
        filename: Optional[Union[str, Callable[..., str]]] = None,
        compression: Optional[AVAILABLE_COMPRESSIONS] = None,
        mem_persist: Optional[Union[MemBackend, bool]] = None,
        fileext: Optional[str | list[str]] = None,
        log_serde_time: bool = False,
    ):
        assert fileext is None or isinstance(fileext, str)
        ser = cls.ser
        deser = cls.deser

        return wrap_backend(
            FileBackend(
                ser=ser,
                deser=deser,
                filename=filename,
                compression=compression,
                fileext=fileext,
            ),
            mem_persist,
            log_serde_time,
        )

    @staticmethod
    def dir(
        cls: type[SaveLoadDirProtocol] | Sequence[type[SaveLoadDirProtocol]],  # type: ignore
        dirname: Optional[Union[str, Callable[..., str]]] = None,
        compression: Optional[AVAILABLE_COMPRESSIONS] = None,
        mem_persist: Optional[Union[MemBackend, bool]] = None,
        log_serde_time: bool = False,
    ):
        if isinstance(cls, Sequence):
            obj = ClsSerdeBackendFactory.get_tuple_serde(cls, None)
            ser = obj["ser"]
            deser = obj["deser"]
        else:
            ser = cls.save
            deser = cls.load

        return wrap_backend(
            DirBackend(
                ser=ser,
                deser=deser,
                dirname=dirname,
                compression=compression,
            ),
            mem_persist,
            log_serde_time,
        )

    @staticmethod
    def get_serde(
        klass: type[SaveLoadDirProtocol],
    ):
        def ser(
            item: SaveLoadDirProtocol,
            dir: Path,
            *args,
        ):
            return item.save(dir, *args)

        def deser(dir: Path, *args):
            return klass.load(dir, *args)

        return {"ser": ser, "deser": deser}

    @staticmethod
    def get_tuple_serde(
        classes: Sequence[type[SaveLoadDirProtocol],],
        exts: Optional[list[str]] = None,
    ):
        def ser(
            items: Sequence[Optional[SaveLoadDirProtocol]],
            dir: Path,
            *args,
        ):
            for i, item in enumerate(items):
                if item is not None:
                    ifile = dir / (f"_{i}.{exts[i]}" if exts is not None else f"_{i}")
                    item.save(ifile, *args)

        def deser(dir: Path, *args):
            output = []
            for i, cls in enumerate(classes):
                ifile = dir / (f"_{i}.{exts[i]}" if exts is not None else f"_{i}")
                if ifile.exists():
                    output.append(cls.load(ifile, *args))
                else:
                    output.append(None)
            return tuple(output)

        return {
            "ser": ser,
            "deser": deser,
        }


class FileBackendFactory:
    @staticmethod
    def pickle(
        filename: Optional[Union[str, Callable[..., str]]] = None,
        compression: Optional[AVAILABLE_COMPRESSIONS] = None,
        mem_persist: Optional[Union[MemBackend, bool]] = None,
        log_serde_time: bool = False,
    ):
        backend = FileBackend(
            ser=pickle.dumps,
            deser=pickle.loads,
            filename=filename,
            fileext="pkl",
            compression=compression,
        )
        return wrap_backend(backend, mem_persist, log_serde_time)

    @staticmethod
    def jl(
        filename: Optional[Union[str, Callable[..., str]]] = None,
        compression: Optional[AVAILABLE_COMPRESSIONS] = None,
        mem_persist: Optional[Union[MemBackend, bool]] = None,
        cls: Optional[Type[JsonSerde]] = None,
        log_serde_time: bool = False,
    ):
        backend = FileBackend(
            ser=FileBackendFactory.jl_ser,
            deser=FileBackendFactory.jl_deser
            if cls is None
            else partial(FileBackendFactory.jl_deser_cls, cls),
            filename=filename,
            fileext="jl",
            compression=compression,
        )
        return wrap_backend(backend, mem_persist, log_serde_time)

    @staticmethod
    def json(
        filename: Optional[Union[str, Callable[..., str]]] = None,
        compression: Optional[AVAILABLE_COMPRESSIONS] = None,
        mem_persist: Optional[Union[MemBackend, bool]] = None,
        cls: Optional[Type[JsonSerde]] = None,
        log_serde_time: bool = False,
        indent: Literal[0, 2] = 0,
    ):
        backend = FileBackend(
            ser=partial(
                FileBackendFactory.json_ser,
                orjson_opts=DEFAULT_ORJSON_OPTS | orjson.OPT_INDENT_2,
            )
            if indent == 2
            else FileBackendFactory.json_ser,
            deser=FileBackendFactory.json_deser
            if cls is None
            else partial(FileBackendFactory.json_deser_cls, cls),
            filename=filename,
            fileext="json",
            compression=compression,
        )
        return wrap_backend(backend, mem_persist, log_serde_time)

    @staticmethod
    def json_ser(
        obj: dict | tuple | list | JsonSerde,
        orjson_opts: int | None = DEFAULT_ORJSON_OPTS,
        orjson_default: Callable[[Any], Any] | None = None,
    ):
        if hasattr(obj, "to_dict"):
            return orjson_dumps(
                obj.to_dict(),  # type: ignore
                option=orjson_opts,
                default=orjson_default or _orjson_default,
            )
        else:
            return orjson_dumps(
                obj,
                option=orjson_opts,
                default=orjson_default or _orjson_default,
            )

    @staticmethod
    def json_deser(data: bytes):
        return orjson.loads(data)

    @staticmethod
    def json_deser_cls(clz: Type[JsonSerde], data: bytes):
        return clz.from_dict(orjson.loads(data))

    @staticmethod
    def jl_ser(
        objs: Sequence[dict] | Sequence[tuple] | Sequence[list] | Sequence[JsonSerde],
        orjson_opts: int | None = DEFAULT_ORJSON_OPTS,
        orjson_default: Callable[[Any], Any] | None = None,
    ):
        out = []
        if len(objs) > 0 and hasattr(objs[0], "to_dict"):
            for obj in objs:
                out.append(
                    orjson_dumps(
                        obj.to_dict(),  # type: ignore
                        option=orjson_opts,
                        default=orjson_default or _orjson_default,
                    )
                )
        else:
            for obj in objs:
                out.append(
                    orjson_dumps(
                        obj,
                        option=orjson_opts,
                        default=orjson_default or _orjson_default,
                    )
                )
        return b"\n".join(out)

    @staticmethod
    def jl_deser(data: bytes):
        return [orjson.loads(line) for line in data.splitlines()]

    @staticmethod
    def jl_deser_cls(clz: Type[JsonSerde], data: bytes):
        return [clz.from_dict(orjson.loads(line)) for line in data.splitlines()]


class Cache:
    file = FileBackendFactory
    sqlite = SqliteBackendFactory
    cls = ClsSerdeBackendFactory

    @staticmethod
    def cache(
        backend: Backend,
        cache_args: Optional[list[str]] = None,
        cache_self_args: Optional[str | Callable[..., dict]] = None,
        cache_ser_args: Optional[dict[str, ArgSer]] = None,
        cache_key: Optional[CacheKeyFn] = None,
        disable: bool | str | Callable[[Any], bool] = False,
    ):
        if isinstance(disable, bool) and disable:
            return identity

        if isinstance(cache_self_args, str):
            cache_self_args = CacheArgsHelper.gen_cache_self_args(cache_self_args)

        def wrapper_fn(func):
            cache_args_helper = CacheArgsHelper.from_actor_func(
                func, cache_self_args, cache_ser_args
            )
            if cache_args is not None:
                cache_args_helper.keep_args(cache_args)

            keyfn = cache_key
            if keyfn is None:
                cache_args_helper.ensure_auto_cache_key_friendly()
                keyfn = lambda self, *args, **kwargs: orjson_dumps(
                    cache_args_helper.get_args(self, *args, **kwargs)
                )

            backend.postinit(func)

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

                with backend.context(self, *args, **kwargs):
                    key = keyfn(self, *args, **kwargs)
                    if backend.has_key(key):
                        return backend.get(key)
                    else:
                        output = func(self, *args, **kwargs)
                        backend.set(key, output)
                        return output

            return fn

        return wrapper_fn

    @staticmethod
    def flat_cache(
        backend: Backend,
        cache_args: Optional[list[str]] = None,
        cache_self_args: Optional[str | Callable[..., dict]] = None,
        cache_ser_args: Optional[dict[str, ArgSer]] = None,
        cache_key: Optional[CacheKeyFn] = None,
        flat_output: Optional[Callable[..., list]] = None,
        unflat_output: Optional[Callable[..., Any]] = None,
        disable: bool | str | Callable[[Any], bool] = False,
    ) -> Callable[[F], F]:
        if isinstance(disable, bool) and disable:
            return identity

        if isinstance(cache_self_args, str):
            cache_self_args = CacheArgsHelper.gen_cache_self_args(cache_self_args)

        def wrapper_fn(func):
            cache_args_helper = CacheArgsHelper.from_actor_func(
                func, cache_self_args, cache_ser_args
            )
            if cache_args is not None:
                cache_args_helper.keep_args(cache_args)

            # since we flatten the input & output, we check the args to make sure we have only one
            # arg of type of sequence.
            seq_arg_name = None
            seq_arg_index = 0
            for i, (name, argtype) in enumerate(cache_args_helper.argtypes.items()):
                if argtype is not None and issubclass(get_origin(argtype), Sequence):
                    assert seq_arg_name is None
                    assert (
                        len(get_args(argtype)) == 1
                    )  # this is likely redundant because the annotation is usually Sequence[T]
                    seq_arg_name = name
                    seq_arg_index = i
            assert seq_arg_name is not None

            keyfn = cache_key
            if keyfn is None:
                # now we have only one sequence arg, we go ahead and change it to its item type
                cache_args_helper.argtypes[seq_arg_name] = get_args(
                    cache_args_helper.argtypes[seq_arg_name]
                )[0]

                # ensure that we can generate a cache key function
                cache_args_helper.ensure_auto_cache_key_friendly()
                keyfn = lambda self, *args, **kwargs: orjson_dumps(
                    cache_args_helper.get_args(self, *args, **kwargs)
                )

            # now let generate flatten functions for input & output
            def flat_inargs(self, *args, **kwargs):
                lst_args = []
                if len(args) > seq_arg_index:
                    # seq arg is in the positional args
                    for seq_val in args[seq_arg_index]:
                        lst_args.append(
                            (
                                args[:seq_arg_index]
                                + (seq_val,)
                                + args[seq_arg_index + 1 :],
                                kwargs,
                            )
                        )
                    return lst_args

                # seq arg is in the keyword args
                for seq_val in kwargs[seq_arg_name]:
                    new_kwargs = kwargs.copy()
                    new_kwargs[seq_arg_name] = seq_val
                    lst_args.append((args, new_kwargs))
                return lst_args

            def unflat_inargs(
                self, lst_args: list[tuple[tuple | list, dict]]
            ) -> tuple[tuple | list, dict]:
                assert len(lst_args) > 0

                args, kwargs = lst_args[0]
                if len(args) > seq_arg_index:
                    # seq arg is in the positional args
                    args = list(args)
                    args[seq_arg_index] = [pa[seq_arg_index] for pa, _ in lst_args]
                    return args, kwargs

                # seq arg is in the keyword args
                kwargs = kwargs.copy()
                kwargs[seq_arg_name] = [pka[seq_arg_name] for _, pka in lst_args]
                return args, kwargs

            if flat_output is None:

                def default_flat_output(self, output: list, *inargs, **inkwargs):
                    return output

                flat_output_fn = default_flat_output
            else:
                flat_output_fn = flat_output

            if unflat_output is None:

                def default_unflat_output(self, flatten_output, *args, **kwargs):
                    return flatten_output

                unflat_output_fn = default_unflat_output
            else:
                unflat_output_fn = unflat_output

            backend.postinit(func)

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

                lst_inargs = flat_inargs(self, *args, **kwargs)
                lst_inargs_keys = [
                    keyfn(self, *in_args, **in_kwargs)  # type: ignore
                    for in_args, in_kwargs in lst_inargs
                ]

                finished_jobs = []
                unfinished_jobs = []
                key_unfinished_jobs = {}
                unfinished_jobs_key = []
                for i, (in_args, in_kwargs) in enumerate(lst_inargs):
                    with backend.context(self, *in_args, **in_kwargs):
                        key = lst_inargs_keys[i]
                        if backend.has_key(key):
                            finished_jobs.append(backend.get(key))
                        else:
                            finished_jobs.append(None)
                            if key not in key_unfinished_jobs:
                                key_unfinished_jobs[key] = [i]
                                unfinished_jobs.append((in_args, in_kwargs))
                                unfinished_jobs_key.append(key)
                            else:
                                key_unfinished_jobs[key].append(i)

                if len(unfinished_jobs) > 0:
                    # finish the remaining jobs
                    subargs, subkwargs = unflat_inargs(self, unfinished_jobs)

                    # call the function with unfinished_args
                    output = func(self, *subargs, **subkwargs)

                    # unroll the output and catch the unfinished args
                    flatten_output = flat_output_fn(self, output, unfinished_jobs)
                    assert len(flatten_output) == len(unfinished_jobs)
                    for unfinished_job, out, key in zip(
                        unfinished_jobs,
                        flatten_output,
                        unfinished_jobs_key,
                    ):
                        with backend.context(
                            self, *unfinished_job[0], **unfinished_job[1]
                        ):
                            backend.set(key, out)
                            for i in key_unfinished_jobs[key]:
                                finished_jobs[i] = out

                assert all(job is not None for job in finished_jobs)
                # now we need to merge the result.
                return unflat_output_fn(self, finished_jobs, *args, **kwargs)

            return fn

        return wrapper_fn  # type: ignore


class CacheArgsHelper:
    """Helper to working with arguments of a function. This class ensures
    that we can select a subset of arguments to use for the cache key, and
    to always put the calling arguments in the same declared order.
    """

    def __init__(
        self,
        args: dict[str, Parameter],
        argtypes: dict[str, Optional[Type]],
        self_args: Optional[Callable[..., dict]] = None,
        cache_ser_args: Optional[dict[str, ArgSer]] = None,
    ):
        self.args = args
        self.argtypes = argtypes
        self.argnames: list[str] = list(self.args.keys())
        self.cache_args = self.argnames
        self.cache_ser_args: dict[str, ArgSer] = cache_ser_args or {}
        self.cache_self_args = self_args or None

    @staticmethod
    def from_actor_func(
        func: Callable,
        self_args: Optional[Callable[..., dict]] = None,
        cache_ser_args: Optional[dict[str, ArgSer]] = None,
    ):
        args: dict[str, Parameter] = {}
        try:
            argtypes: dict[str, Optional[Type]] = get_type_hints(func)
            if "return" in argtypes:
                argtypes.pop("return")
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
            args[name] = param
            if name not in argtypes:
                argtypes[name] = None

        assert (
            next(iter(args)) == "self"
        ), "The first argument of the method must be self, an instance of BaseActor"
        args.pop("self")

        return CacheArgsHelper(args, argtypes, self_args, cache_ser_args)

    def keep_args(self, names: Iterable[str]) -> None:
        self.cache_args = list(names)

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

            if name in self.cache_ser_args:
                # the users provide a function to serialize the argument manually, so we trust the user.
                continue

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

        for name, ser_fn in self.cache_ser_args.items():
            out[name] = ser_fn(out[name])
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


class Backend(ABC):
    def __init__(
        self,
        ser: Callable[[Any], bytes],
        deser: Callable[[bytes], Value],
        compression: Optional[AVAILABLE_COMPRESSIONS] = None,
    ):
        if compression == "gz":
            origin_ser = ser
            origin_deser = deser
            ser = lambda x: gzip.compress(origin_ser(x), mtime=0)
            deser = lambda x: origin_deser(gzip.decompress(x))
        elif compression == "bz2":
            origin_ser = ser
            origin_deser = deser
            ser = lambda x: bz2.compress(origin_ser(x))
            deser = lambda x: origin_deser(bz2.decompress(x))
        elif compression == "lz4":
            if lz4_frame is None:
                raise ValueError("lz4 is not installed")
            # using lambda somehow terminate the program without raising an error
            ser = Chain2(lz4_frame.compress, ser)
            deser = Chain2(deser, lz4_frame.decompress)

        self.compression = compression
        self.ser = ser
        self.deser = deser

    def postinit(self, func: Callable):
        if not hasattr(self, "_is_postinited"):
            self._is_postinited = True
        else:
            raise RuntimeError("Backend can only be postinited once")

    @contextmanager
    def context(self, obj: HasWorkingFsTrait, *args, **kwargs):
        yield None

    @abstractmethod
    def has_key(self, key: bytes) -> bool:
        ...

    @abstractmethod
    def get(self, key: bytes) -> Value:
        ...

    @abstractmethod
    def set(self, key: bytes, value: Value) -> None:
        ...


class MemBackend(Backend):
    def __init__(self):
        self.key2value: dict[bytes, Value] = {}

    def has_key(self, key: bytes) -> bool:
        return key in self.key2value

    def get(self, key: bytes) -> Value:
        return self.key2value[key]

    def set(self, key: bytes, value: Value) -> None:
        self.key2value[key] = value

    def clear(self):
        self.key2value.clear()


class FileBackend(Backend):
    def __init__(
        self,
        ser: Callable[[Any], bytes],
        deser: Callable[[bytes], Any],
        filename: Optional[str | Callable[..., str]] = None,
        fileext: Optional[str] = None,
        compression: Optional[AVAILABLE_COMPRESSIONS] = None,
    ):
        super().__init__(ser, deser, compression)
        self.filename = filename
        self.fileext = fileext
        self.container = ContextContainer()

    def postinit(self, func: Callable):
        super().postinit(func)
        if self.filename is None:
            self.filename = func.__name__
            if self.fileext is not None:
                assert not self.fileext.startswith(".")
                self.filename += f".{self.fileext}"

        if isinstance(self.filename, str):
            assert (
                self.filename.find(".") != -1
            ), "Must have file extension to be considered as a file"

    @contextmanager
    def context(self, obj: HasWorkingFsTrait, *args, **kwargs):
        if isinstance(self.filename, str):
            filename = self.filename
        else:
            assert self.filename is not None
            filename = self.filename(args[0], *args[1], **args[2])
            assert (
                filename.find(".") != -1
            ), "Must have file extension to be considered as a file"

        try:
            self.container.enable()
            self.container.filename = filename
            self.container.fs = obj.get_working_fs()
            self.container.cache_file = None
            yield self.container
        finally:
            self.container.disable()

    def has_key(self, key: bytes):
        if self.container.cache_file is None:
            self.container.cache_file = self.container.fs.get(
                self.container.filename, key=key, save_key=True
            )
            self.container.key = key
        return self.container.cache_file.exists()

    def get(self, key: bytes):
        if self.container.cache_file is None:
            self.container.cache_file = self.container.fs.get(
                self.container.filename, key=key, save_key=True
            )
            self.container.key = key
        else:
            assert self.container.key == key
        fpath = self.container.cache_file.get()
        with open(fpath, "rb") as f:
            return self.deser(f.read())

    def set(self, key: bytes, value: Any):
        if self.container.cache_file is None:
            self.container.cache_file = self.container.fs.get(
                self.container.filename, key=key, save_key=True
            )
            self.container.key = key
        else:
            assert self.container.key == key
        with self.container.fs.acquire_write_lock(), self.container.cache_file.reserve_and_track() as fpath:
            with open(fpath, "wb") as f:
                return f.write(self.ser(value))


class DirBackend(Backend):
    def __init__(
        self,
        ser: Callable[[Any, Path, Optional[AVAILABLE_COMPRESSIONS]], None],
        deser: Callable[[Path, Optional[AVAILABLE_COMPRESSIONS]], Any],
        dirname: Optional[str | Callable[..., str]] = None,
        compression: Optional[AVAILABLE_COMPRESSIONS] = None,
    ):
        self.dirname = dirname
        self.ser = ser
        self.deser = deser
        self.compression: Optional[AVAILABLE_COMPRESSIONS] = compression
        self.container = ContextContainer()

    def postinit(self, func: Callable):
        super().postinit(func)
        if self.dirname is None:
            self.dirname = func.__name__

        if isinstance(self.dirname, str):
            assert (
                self.dirname.find(".") == -1
            ), "Must not have file extension to be considered as a directory"

    @contextmanager
    def context(self, obj: HasWorkingFsTrait, *args, **kwargs):
        if isinstance(self.dirname, str):
            dirname = self.dirname
        else:
            assert self.dirname is not None
            dirname = self.dirname(args[0], *args[1], **args[2])
            assert (
                dirname.find(".") == -1
            ), "Must not have file extension to be considered as a directory"

        try:
            self.container.enable()
            self.container.dirname = dirname
            self.container.fs = obj.get_working_fs()
            self.container.cache_dir = None
            yield self.container
        finally:
            self.container.disable()

    def has_key(self, key: bytes):
        if self.container.cache_dir is None:
            self.container.cache_dir = self.container.fs.get(
                self.container.dirname, key=key, save_key=True
            )
            self.container.key = key
        return self.container.cache_dir.exists()

    def get(self, key: bytes):
        if self.container.cache_dir is None:
            self.container.cache_dir = self.container.fs.get(
                self.container.dirname, key=key, save_key=True
            )
            self.container.key = key
        else:
            assert self.container.key == key
        dpath = self.container.cache_dir.get()
        return self.deser(dpath, self.compression)

    def set(self, key: bytes, value: Any):
        if self.container.cache_dir is None:
            self.container.cache_dir = self.container.fs.get(
                self.container.dirname, key=key, save_key=True
            )
            self.container.key = key
        else:
            assert self.container.key == key
        with self.container.fs.acquire_write_lock(), self.container.cache_dir.reserve_and_track() as dpath:
            return self.ser(value, dpath, self.compression)


class SqliteBackend(Backend):
    def postinit(self, func: Callable):
        super().postinit(func)
        self.dbname = f"{func.__name__}.sqlite"
        self.dbconn: SqliteDict = None  # type: ignore

    @contextmanager
    def context(self, obj: HasWorkingFsTrait, *args, **kwargs):
        if self.dbconn is None:
            self.dbconn = SqliteDict(
                obj.get_working_fs().root / self.dbname,
                keytype=SqliteDictFieldType.bytes,
                ser_value=identity,
                deser_value=identity,
            )

        yield None

    def has_key(self, key: bytes) -> bool:
        return key in self.dbconn

    def get(self, key: bytes) -> Any:
        return self.deser(self.dbconn[key])

    def set(self, key: bytes, value: Any) -> None:
        self.dbconn[key] = self.ser(value)


class ReplicatedBackends(Backend):
    """A composite backend that a backend (i) is a super set (key-value) of
    its the previous backend (i-1). Accessing to this composite backend will
    slowly build up the front backends to have the same key-value pairs as
    the last backend.

    This is useful for combining MemBackend and DiskBackend.
    """

    def __init__(self, backends: list[Backend]):
        self.backends = backends

    def postinit(self, func: Callable[..., Any]):
        super().postinit(func)
        for backend in self.backends:
            backend.postinit(func)

    @contextmanager
    def context(self, obj: HasWorkingFsTrait, *args, **kwargs):
        if len(self.backends) == 2:
            with self.backends[0].context(obj, *args, **kwargs):
                with self.backends[1].context(obj, *args, **kwargs):
                    yield None
        elif len(self.backends) == 3:
            with self.backends[0].context(obj, *args, **kwargs):
                with self.backends[1].context(obj, *args, **kwargs):
                    with self.backends[2].context(obj, *args, **kwargs):
                        yield None
        elif len(self.backends) == 4:
            with self.backends[0].context(obj, *args, **kwargs):
                with self.backends[1].context(obj, *args, **kwargs):
                    with self.backends[2].context(obj, *args, **kwargs):
                        with self.backends[3].context(obj, *args, **kwargs):
                            yield None
        else:
            # recursive yield
            with self.backends[0].context(obj, *args, **kwargs):
                with ReplicatedBackends(self.backends[1:]).context(
                    obj, *args, **kwargs
                ):  # type: ignore
                    yield None

    def has_key(self, key: bytes) -> bool:
        return any(backend.has_key(key) for backend in self.backends)

    def get(self, key: bytes) -> Value:
        for i, backend in enumerate(self.backends):
            if backend.has_key(key):
                value = backend.get(key)
                if i > 0:
                    # replicate the value to the previous backend
                    for j in range(i):
                        self.backends[j].set(key, value)
                return value

    def set(self, key: bytes, value: Value):
        for backend in self.backends:
            backend.set(key, value)


class LogSerdeTimeBackend(Backend):
    def __init__(self, backend: Backend):
        self.backend = backend
        self.logger: Logger = None  # type: ignore

    def postinit(self, func: Callable):
        self.backend.postinit(func)
        self.logger = logger

    @contextmanager
    def context(self, obj: HasWorkingFsTrait, *args, **kwargs):
        if self.logger is None:
            self.logger = logger.bind(name=obj.__class__.__name__)
        with self.backend.context(obj, *args, **kwargs):
            yield None

    def has_key(self, key: bytes) -> bool:
        return self.backend.has_key(key)

    def get(self, key: bytes) -> Value:
        with Timer().watch_and_report(
            f"serialize",
            self.logger.debug,
        ):
            return self.backend.get(key)

    def set(self, key: bytes, value: Value) -> None:
        with Timer().watch_and_report(
            f"deserialize",
            self.logger.debug,
        ):
            self.backend.set(key, value)


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


class CacheableFn(Generic[T], ABC, Cacheable):
    """This utility provides a way to break a giantic function into smaller pieces that can be cached individually."""

    def __init__(
        self, use_args: list[str], workdir: Union[FS, Path], disable: bool = False
    ):
        super().__init__(workdir, disable)
        self.use_args = use_args

    @staticmethod
    def get_cache_key(slf: CacheableFn, args: T) -> bytes:
        return orjson.dumps(
            {attr: getattr(args, attr) for attr in slf.get_use_args()},
            option=orjson.OPT_SORT_KEYS | orjson.OPT_SERIALIZE_DATACLASS,
        )

    @lru_cache()
    def get_use_args(self) -> set[str]:
        cache_args = set(self.use_args)
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
    def __call__(self, args: T) -> Any:
        """This is where to put the function body. To cache it, wraps it with @Cache.<X> decorators"""
        ...


class SerdeProtocol(Protocol):
    def ser(self) -> bytes:
        ...

    @classmethod
    def deser(cls, data: bytes) -> Self:
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


def wrap_backend(
    backend: Backend,
    mem_persist: Optional[Union[MemBackend, bool]],
    log_serde_time: bool,
):
    if log_serde_time:
        backend = LogSerdeTimeBackend(backend)
    if mem_persist:
        if mem_persist is not None:
            mem_persist = MemBackend()
        backend = ReplicatedBackends([mem_persist, backend])
    return backend

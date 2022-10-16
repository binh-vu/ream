import functools
from inspect import Parameter, signature

from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Type,
    Union,
    get_args,
    get_origin,
)

import serde

from ream.actors.base import BaseActor
from ream.helper import orjson_dumps
from serde.helper import JsonSerde

NoneType = type(None)


class JLSerdeCache:
    @staticmethod
    def file(
        cache_args: Optional[List[str]] = None,
        cache_key: Optional[Callable[[dict], bytes]] = None,
        filename: Optional[Union[str, Callable[..., str]]] = None,
        compression: Optional[Literal["gz", "bz2", "lz4"]] = None,
        mem_persist: bool = False,
        cache_attr: str = "_cache",
        cls: Optional[Type[JsonSerde]] = None,
    ):
        return Cache.file(
            ser=serde.jl.ser,
            deser=functools.partial(serde.jl.deser, cls=cls),
            cache_args=cache_args,
            cache_key=cache_key,
            filename=filename,
            compression=compression,
            mem_persist=mem_persist,
            cache_attr=cache_attr,
        )


class PickleSerdeCache:
    @staticmethod
    def file(
        cache_args: Optional[List[str]] = None,
        cache_key: Optional[Callable[[dict], bytes]] = None,
        filename: Optional[Union[str, Callable[..., str]]] = None,
        compression: Optional[Literal["gz", "bz2", "lz4"]] = None,
        mem_persist: bool = False,
        cache_attr: str = "_cache",
    ):
        return Cache.file(
            ser=serde.pickle.ser,
            deser=serde.pickle.deser,
            cache_args=cache_args,
            cache_key=cache_key,
            filename=filename,
            compression=compression,
            mem_persist=mem_persist,
            cache_attr=cache_attr,
        )


class Cache:
    jl = JLSerdeCache
    pickle = PickleSerdeCache

    @staticmethod
    def mem(
        cache_args: Optional[List[str]] = None,
        cache_key: Optional[Callable[[dict], bytes]] = None,
        cache_attr: str = "_cache",
    ):
        """Decorator to cache the result of a function to an attribute in the instance in memory.

        Note: It does not support function with variable number of arguments.

        Args:
            cache_args: List of arguments to use for the cache key. If None, all arguments are used.
            cache_key: Function to use to generate the cache key. If None, the default is used. The default function
                only support arguments of types str, int, bool, and None.
            cache_attr: Name of the attribute to use to store the cache in the instance.
        """

        def wrapper_fn(func):
            func_name = func.__name__
            cache_args_helper = CacheArgsHelper(func)
            if cache_args is not None:
                cache_args_helper.keep_args(cache_args)

            keyfn = cache_key
            if keyfn is None:
                keyfn = cache_args_helper.get_args_as_tuple

            @functools.wraps(func)
            def fn(self, *args, **kwargs):
                if not hasattr(self, cache_attr):
                    setattr(self, cache_attr, {})
                cache = getattr(self, cache_attr)
                key = (func_name, keyfn(*args, **kwargs))
                if key not in cache:
                    cache[key] = func(self, *args, **kwargs)
                return cache[key]

            return fn

        return wrapper_fn

    @staticmethod
    def file(
        ser: Callable[[Any, Path], None],
        deser: Callable[[Path], Any],
        cache_args: Optional[List[str]] = None,
        cache_key: Optional[Callable[[dict], bytes]] = None,
        filename: Optional[Union[str, Callable[..., str]]] = None,
        compression: Optional[Literal["gz", "bz2", "lz4"]] = None,
        mem_persist: bool = False,
        cache_attr: str = "_cache",
    ) -> Callable:
        """Decorator to cache the result of a function to a file.

        Note: It does not support function with variable number of arguments.

        Args:
            ser: A function to serialize the output of the function to a file.
            deser: A function to deserialize the output of the function from a file.
            cache_args: List of arguments to use for the cache key. If None, all arguments are used.
            cache_key: Function to use to generate the cache key. If None, the default is used. The default function
                only support arguments of types str, int, bool, and None.
            filename: Filename to use for the cache file. If None, the name of the function is used. If it is a function,
                it will be called with the arguments of the function to generate the filename.
            compression: whether to compress the cache file, the compression is detected via the file extension. Therefore,
                this option has no effect if the filename is a function.
            mem_persist: If True, the cache will also be stored in memory. This is a combination of mem and file cache.
            cache_attr: Name of the attribute to use to store the cache in the instance.
        """

        def wrapper_fn(func):
            if filename is None:
                filename2 = func.__name__
                if compression is not None:
                    filename2 += f".{compression}"
            else:
                filename2 = filename
                if isinstance(filename2, str) and compression is not None:
                    assert filename2.endswith(compression)

            cache_args_helper = CacheArgsHelper(func)
            if cache_args is not None:
                cache_args_helper.keep_args(cache_args)

            keyfn = cache_key
            if keyfn is None:
                keyfn = lambda *args, **kwargs: orjson_dumps(
                    cache_args_helper.get_args(*args, **kwargs)
                )

            @functools.wraps(func)
            def fn(self: BaseActor, *args, **kwargs):
                fs = self.get_working_fs()

                if isinstance(filename2, str):
                    cache_filename = filename2
                else:
                    cache_filename = filename2(*args, **kwargs)

                cache_file = fs.get(
                    cache_filename, key=keyfn(*args, **kwargs), save_key=True
                )
                if not cache_file.exists():
                    with fs.acquire_write_lock(), cache_file.reserve_and_track() as fpath:
                        output = func(self, *args, **kwargs)
                        ser(output, fpath)
                else:
                    output = deser(cache_file.get())
                return output

            if mem_persist is not None:
                return Cache.mem(cache_args, cache_key, cache_attr)(fn)
            return fn

        return wrapper_fn


class CacheArgsHelper:
    """Helper to working with arguments of a function. This class ensures
    that we can select a subset of arguments to use for the cache key, and
    to always put the calling arguments in the same declared order.
    """

    def __init__(self, func: Callable):
        self.args: Dict[str, Parameter] = {}
        for name, param in signature(func).parameters.items():
            self.args[name] = param

        assert (
            next(iter(self.args)) == "self"
        ), "The first argument of the method must be self, an instance of BaseActor"
        self.args.pop("self")
        self.argnames: List[str] = list(self.args.keys())
        self.cache_args = self.argnames

    def keep_args(self, names: Iterable[str]) -> None:
        self.cache_args = list(names)

    def get_arg_type(self) -> Dict[str, Type]:
        return {
            name: self.args[name].annotation
            if self.args[name].annotation is not Parameter.empty
            else None
            for name in self.cache_args
        }

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

            argtype = param.annotation
            if argtype is Parameter.empty:
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
            elif origin is not Union:
                raise TypeError(
                    f"Automatically generating caching key to cache a function call requires all arguments to be one of type: str, int, bool, or None. Found {name} with type {argtype}"
                )
            else:
                args = get_args(argtype)
                if any(
                    not issubclass(a, (str, int, bool)) and a is not NoneType
                    for a in args
                ):
                    raise TypeError(
                        f"Automatically generating caching key to cache a function call requires all arguments to be one of type: str, int, bool, or None. Found {name} with type {argtype}"
                    )

    def get_args(self, *args, **kwargs) -> dict:
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
        return out

    def get_args_as_tuple(self, *args, **kwargs) -> tuple:
        # TODO: improve me!
        return tuple(self.get_args(*args, **kwargs).values())

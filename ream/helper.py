from pathlib import Path
import sys
from typing import Type, TypeVar, Union, get_args, get_origin, Tuple
from loguru import logger
import orjson

TYPE_ALIASES = {"typing.List": "list", "typing.Dict": "dict", "typing.Set": "set"}


def get_classpath(type: Type) -> str:
    if type.__module__ == "builtins":
        return type.__qualname__

    if hasattr(type, "__qualname__"):
        return type.__module__ + "." + type.__qualname__

    # typically a class from the typing module
    if hasattr(type, "_name") and type._name is not None:
        path = type.__module__ + "." + type._name
        if path in TYPE_ALIASES:
            path = TYPE_ALIASES[path]
    elif hasattr(type, "__origin__") and hasattr(type.__origin__, "_name"):
        # found one case which is typing.Union
        path = type.__module__ + "." + type.__origin__._name
    else:
        raise NotImplementedError(type)

    return path


def configure_loguru():
    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        format=_logger_formatter_colorful,
    )


def orjson_dumps(obj, **kwargs):
    return orjson.dumps(obj, default=_orjson_default, **kwargs)


def _logger_formatter_colorful(record):
    clsname = "{extra[cls]}." if "cls" in record["extra"] else ""
    return (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>%s{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>\n{exception}"
        % clsname
    )


def _logger_formatter(record):
    clsname = "{extra[cls]}." if "cls" in record["extra"] else ""
    return (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:%s{function}:{line} - {message}\n{exception}"
        % clsname
    )


def _orjson_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def resolve_type_arguments(
    query_type: Type, target_type: Type
) -> Tuple[Union[Type, TypeVar], ...]:
    """
    FROM: https://stackoverflow.com/a/69862817

    Resolves the type arguments of the query type as supplied by the target type of any of its bases.
    Operates in a tail-recursive fashion, and drills through the hierarchy of generic base types breadth-first in left-to-right order to correctly identify the type arguments that need to be supplied to the next recursive call.

    raises a TypeError if they target type was not an instance of the query type.

    :param query_type: Must be supplied without args (e.g. Mapping not Mapping[KT,VT]
    :param target_type: Must be supplied with args (e.g. Mapping[KT, T] or Mapping[str, int] not Mapping)
    :return: A tuple of the arguments given via target_type for the type parameters of for the query_type, if it has any parameters, otherwise an empty tuple. These arguments may themselves be TypeVars.
    """
    target_origin = get_origin(target_type)
    if target_origin is None:
        if target_type is query_type:
            return target_type.__parameters__
        else:
            target_origin = target_type
            supplied_args = None
    else:
        supplied_args = get_args(target_type)
        if target_origin is query_type:
            return supplied_args
    param_set = set()
    param_list = []
    for (i, each_base) in enumerate(target_origin.__orig_bases__):
        each_origin = get_origin(each_base)
        if each_origin is not None:
            # each base is of the form class[T], which is a private type _GenericAlias, but it is formally documented to have __parameters__
            for each_param in each_base.__parameters__:
                if each_param not in param_set:
                    param_set.add(each_param)
                    param_list.append(each_param)
            if issubclass(each_origin, query_type):
                if supplied_args is not None and len(supplied_args) > 0:
                    params_to_args = {
                        key: value for (key, value) in zip(param_list, supplied_args)
                    }
                    resolved_args = tuple(
                        params_to_args[each] for each in each_base.__parameters__
                    )
                    return resolve_type_arguments(
                        query_type, each_base[resolved_args]
                    )  # each_base[args] fowards the args to each_base, it is not quite equivalent to GenericAlias(each_origin, resolved_args)
                else:
                    return resolve_type_arguments(query_type, each_base)
        elif issubclass(each_base, query_type):
            return resolve_type_arguments(query_type, each_base)
    if not issubclass(target_origin, query_type):  # type: ignore
        raise ValueError(f"{target_type} is not a subclass of {query_type}")
    else:
        return ()

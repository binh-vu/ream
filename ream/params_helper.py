from __future__ import annotations
from copy import deepcopy
from functools import partial
from dataclasses import Field, asdict, dataclass, fields, is_dataclass, replace
from typing import Any, Dict, List, Type, Union

DataClassInstance = Any


@dataclass
class NoParams:
    """For an actor that has no parameter"""

    pass


@dataclass
class EnumParams:
    """For specify different method and its parameters.

    ## Example:

    ```python
    class Params(EnumParams):
        method: Literal["method_a", "method_b"] = filed(metadata={
            "variants": {"method_a": MethodAClass, "method_b": MethodBClass}
        })
        method_a: Optional[MethodAParams] = None
        method_b: Optional[MethodBParams] = None
    ```

    It needs a field to specified the method to use (in the example, the field is `method`), and
    other fields named after the method (in the example, the fields are `method_a` and `method_b`).

    The field `method` also needs a metadata `variants` to specify the method's class.

    At the same time, it should only only parameters of a method. For example, if `method` is `method_a`,
    then `method_a` is not None and `method_b` is `None`.
    """

    def __post_init__(self):
        for method_field in self.get_method_fields():
            method = getattr(self, method_field.name)
            assert hasattr(self, method)
            for name in method_field.metadata["variants"].keys():
                if name != method:
                    # set params of other methods to None
                    setattr(self, name, None)

    def without_method_args(self):
        """Return a shallow copy of this object with the methods' parameters set to None."""
        other = replace(self)  # type: ignore -- method in dataclass
        for field in self.get_method_fields():
            setattr(other, getattr(self, field.name), None)
        return other

    def get_method_class(self, method_field: Field) -> Type:
        method = getattr(self, method_field.name)
        return method_field.metadata["variants"][method]

    def get_method_constructor(self, method_field: Field, **kwargs):
        method = getattr(self, method_field.name)
        if (
            "variant_constructors" in method_field.metadata
            and method in method_field.metadata["variant_constructors"]
        ):
            return partial(
                method_field.metadata["variant_constructors"][method], **kwargs
            )
        return partial(self.get_method_class(method_field), **kwargs)

    def get_method_params(self, method_field: Field) -> DataClassInstance:
        method = getattr(self, method_field.name)
        return getattr(self, method)

    def get_method_fields(self) -> list[Field]:
        if not hasattr(self, "__method_fields"):
            method_fields = []
            for field in fields(self):
                if "variants" in field.metadata:
                    method_fields.append(field)

            if len(method_fields) == 0:
                raise ValueError(
                    f"No method field found in {self.__class__}. The method field is the one has `variants` property (dict type) in its metadata"
                )

            self.__method_fields = method_fields
        return self.__method_fields


def are_valid_parameters(
    params: Union[
        DataClassInstance, List[DataClassInstance], Dict[str, DataClassInstance]
    ]
):
    """Check if the parameters are valid"""
    if isinstance(params, list):
        return all(is_dataclass(param) for param in params)
    elif isinstance(params, dict):
        return all(
            isinstance(name, str) and is_dataclass(param)
            for name, param in params.items()
        )
    else:
        assert is_dataclass(params), "Parameters must be an instance of a dataclass"


def param_as_dict(param: DataClassInstance) -> dict:
    """Convert a dataclass to a dictionary"""
    if not is_dataclass(param):
        raise TypeError("Parameter must be an instance of a dataclass")
    return _param_as_dict_inner(param, dict_factory=dict)  # type: ignore


def _param_as_dict_inner(obj, dict_factory):
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if is_dataclass(obj):
        result = []
        for f in fields(obj):
            value = _param_as_dict_inner(getattr(obj, f.name), dict_factory)
            result.append((f.name, value))
        return dict_factory(result)
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
        # obj is a namedtuple.  Recurse into it, but the returned
        # object is another namedtuple of the same type.  This is
        # similar to how other list- or tuple-derived classes are
        # treated (see below), but we just need to create them
        # differently because a namedtuple's __init__ needs to be
        # called differently (see bpo-34363).

        # I'm not using namedtuple's _asdict()
        # method, because:
        # - it does not recurse in to the namedtuple fields and
        #   convert them to dicts (using dict_factory).
        # - I don't actually want to return a dict here.  The main
        #   use case here is json.dumps, and it handles converting
        #   namedtuples to lists.  Admittedly we're losing some
        #   information here when we produce a json list instead of a
        #   dict.  Note that if we returned dicts here instead of
        #   namedtuples, we could no longer call asdict() on a data
        #   structure where a namedtuple was used as a dict key.

        return type(obj)(*[_param_as_dict_inner(v, dict_factory) for v in obj])
    elif isinstance(obj, (list, tuple)):
        # Assume we can create an object of this type by passing in a
        # generator (which is not true for namedtuples, handled
        # above).
        return type(obj)(_param_as_dict_inner(v, dict_factory) for v in obj)
    elif isinstance(obj, dict):
        return type(obj)(
            (
                _param_as_dict_inner(k, dict_factory),
                _param_as_dict_inner(v, dict_factory),
            )
            for k, v in obj.items()
        )
    else:
        return deepcopy(obj)

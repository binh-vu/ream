from abc import ABC
from pathlib import Path
from typing import (
    Any,
    Generic,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Type,
    List,
    TypeVar,
    Union,
    get_type_hints,
    Tuple,
)
from dataclasses import dataclass
import lz4.frame
from nptyping.ndarray import NDArrayMeta  # type: ignore
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import orjson
import struct
from serde.helper import get_compression, get_open_fn

T = TypeVar("T")


class Index(Generic[T]):
    __slots__ = ["index"]

    def __init__(self, index: T):
        self.index = index

    def to_bytes(self) -> bytes:
        return orjson.dumps(self)

    @staticmethod
    def from_bytes(obj):
        return Index(orjson.loads(obj))


@dataclass
class NumpyDataModelMetadata:
    props: Sequence[str]
    array_props: Sequence[str]
    index_props: Sequence[str]
    array_prop_idxs: Sequence[int]
    index_prop_idxs: Sequence[Tuple[int, Type[Index]]]


class NumpyDataModel:
    """A data model that is backed by numpy arrays, holding a list of objects of type T.

    It can have two types of attributes: numpy arrays and index. For this class to work correctly,
    all attributes of this class or its subclass must have type annotations. For numpy, you can use NDArray
    (from nptyping) or numpy.ndarray.
    """

    __slots__ = []

    def __init__(self, *args, **kwargs):
        for name, arg in zip(self.__slots__, args):
            setattr(self, name, arg)
        for name, arg in kwargs.items():
            setattr(self, name, arg)

    @classmethod
    def init(cls):
        if cls._metadata is None:
            anns = get_type_hints(cls)
            array_props = []
            index_props = []
            array_prop_idxs = []
            index_prop_idxs = []
            for i, name in enumerate(cls.__slots__):
                if name not in anns:
                    raise ValueError(
                        f"Attribute {name} of {cls} must have a type annotation, even if the type is Any"
                    )

                ann = anns[name]
                if ann is np.ndarray or isinstance(ann, NDArrayMeta):
                    # it it a numpy array
                    array_prop_idxs.append(i)
                    array_props.append(name)
                elif issubclass(ann, Index):
                    index_props.append(name)
                    index_prop_idxs.append((i, ann))
                else:
                    raise ValueError(
                        f"Value of attribute {name} is not numpy array or index"
                    )

            cls._metadata = NumpyDataModelMetadata(
                props=cls.__slots__,
                array_props=array_props,
                index_props=index_props,
                array_prop_idxs=array_prop_idxs,
                index_prop_idxs=index_prop_idxs,
            )

    def __len__(self):
        """Get length of the array"""
        return len(getattr(self, self._metadata.array_props[0]))

    def shallow_clone(self):
        return self.__class__(*(getattr(self, name) for name in self.__slots__))

    def swap_(self, i: int, j: int):
        """Swap the position of two elements at index i and j.
        Note: This operator mutates the array
        """
        for field in self.__slots__:
            value = getattr(self, field)
            value[i], value[j] = value[j], value[i]

    def sort_subset_(
        self,
        i: int,
        j: int,
        sortby: str,
        sortorder: Literal["asc", "desc"],
        kind: Literal["stable", "quicksort"] = "stable",
    ):
        """Sort the elements between [i, j).
        Note: This operator mutates the array
        """
        sortedvalue = getattr(self, sortby)
        sortedindex = np.argsort(sortedvalue, kind=kind)
        if sortorder == "desc":
            sortedindex = sortedindex[::-1]

        for field in self._metadata.array_props:
            value = getattr(self, field)
            value[i:j] = value[i:j][sortedindex]

    def replace_subset_(
        self, field: str, value: Union[Sequence, np.ndarray], i: int, j: int
    ):
        getattr(self, field)[i:j] = value

    def save(self, file: Path):
        metadata = self._metadata
        if metadata is None:
            raise Exception(
                f"{self.__class__.__qualname__}.init() must be called before usage to finish setting up the schema"
            )

        compression = get_compression(file) or "snappy"
        pq.write_table(
            pa.table({name: getattr(self, name) for name in metadata.array_props}),
            str(file),
            compression=compression,
        )

        if len(metadata.index_props) > 0:
            index_file = file.parent / (file.stem + f".index.{compression}")
            with get_open_fn(index_file)(str(index_file), "wb") as f:
                f.write(struct.pack("<I", len(metadata.index_props)))
                for name in metadata.index_props:
                    serindex = getattr(self, name).to_bytes()
                    f.write(struct.pack("<I", len(serindex)))
                    f.write(serindex)

    @classmethod
    def load(cls, file: Path):
        metadata = cls._metadata
        if metadata is None:
            raise Exception(
                f"{cls.__qualname__}.init() must be called before usage to finish setting up the schema"
            )

        compression = get_compression(file) or "snappy"
        index_file = file.parent / (file.stem + f".index.{compression}")
        indices = []

        if index_file.exists():
            with get_open_fn(index_file)(str(index_file), "rb") as f:
                n_indices = struct.unpack("<I", f.read(4))[0]
                for i in range(n_indices):
                    size = struct.unpack("<I", f.read(4))[0]
                    index = metadata.index_prop_idxs[i][1].from_bytes(f.read(size))
                    indices.append(index)

        tbl = pq.read_table(str(file))
        kwargs = {}
        for name in metadata.array_props:
            kwargs[name] = tbl.column(name).to_numpy()

        if len(indices) > 0 or len(kwargs) < len(cls.__slots__):
            # the two conditions should always equal, we can use one of them
            # but we check to ensure data is valid
            assert len(indices) > 0 and len(kwargs) < len(
                cls.__slots__
            ), "The serialized data is inconsistent with the schema"
            i = 0
            for name in cls.__slots__:
                if name not in kwargs:
                    kwargs[name] = indices[i]
                    i += 1

        return cls(**kwargs)


NumpyDataModel.init()

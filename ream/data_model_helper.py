from abc import ABC
from pathlib import Path
from typing import (
    Callable,
    Generic,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Type,
    TypeVar,
    Union,
    get_origin,
    get_type_hints,
    Tuple,
)
from dataclasses import dataclass, fields, is_dataclass
from nptyping.ndarray import NDArrayMeta  # type: ignore
from nptyping.shape_expression import get_dimensions  # type: ignore
import numpy as np
import orjson
import pyarrow.parquet as pq
import pyarrow as pa
import pickle
import struct
from ream.helper import has_dict_with_nonstr_keys
from serde.helper import get_compression, get_open_fn

T = TypeVar("T")


class Index(Generic[T]):
    __slots__ = ["index"]

    def __init__(self, index: T):
        self.index = index

    def to_bytes(self) -> bytes:
        return pickle.dumps(self)

    @staticmethod
    def from_bytes(obj):
        return Index(pickle.loads(obj))


class OffsetIndex(Index[T], Generic[T]):
    __slots__ = ["offset"]

    def __init__(self, index: T, offset: int):
        super().__init__(index)
        self.offset = offset


@dataclass
class NumpyDataModelMetadata:
    cls: type
    props: Sequence[str]
    array_props: Sequence[str]
    array2d_props: Sequence[str]
    index_props: Sequence[str]
    array_prop_idxs: Sequence[int]
    index_prop_idxs: Sequence[Tuple[int, Optional[Type[Index]]]]
    default_serdict: Callable[[dict], bytes]
    default_deserdict: Callable[[bytes], dict]


class NumpyDataModel:
    """A data model that is backed by numpy arrays, holding a list of objects of type T.

    It can have two types of attributes: numpy arrays and index. For this class to work correctly,
    all attributes of this class or its subclass must have type annotations. For numpy, you can use NDArray
    (from nptyping) or numpy.ndarray.

    It can also support 2-dimensional arrays, but note that the 2nd dimension should not
    be too big
    """

    __slots__ = []
    _metadata: Optional[NumpyDataModelMetadata] = None

    def __init__(self, *args, **kwargs):
        for name, arg in zip(self.__slots__, args):
            setattr(self, name, arg)
        for name, arg in kwargs.items():
            setattr(self, name, arg)

    @classmethod
    def init(
        cls,
    ):
        if cls._metadata is None or cls._metadata.cls is not cls:
            anns = get_type_hints(cls)
            array_props = []
            index_props = []
            array_prop_idxs = []
            array2d_props = []
            index_prop_idxs = []

            # check if we can use orjson
            useorjson = True

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
                    if isinstance(ann, NDArrayMeta):
                        shp = ann.__args__[0]
                        dims = get_dimensions(shp.__args__[0])
                        if len(dims) == 2:
                            array2d_props.append(name)
                        elif len(dims) != 1:
                            raise TypeError(
                                "Do not support more than 2-dimension array"
                            )
                else:
                    ann_origin = get_origin(ann) or ann
                    if issubclass(ann_origin, (dict, list, Index)):
                        index_props.append(name)
                        if issubclass(ann_origin, Index):
                            index_prop_idxs.append((i, ann))
                        else:
                            index_prop_idxs.append((i, None))
                            if has_dict_with_nonstr_keys(ann):
                                useorjson = False
                    elif ann_origin is not type(None):
                        raise ValueError(
                            f"Value of attribute {name} is not numpy array or index or NoneType"
                        )

            if useorjson:
                default_serdict = orjson.dumps
                default_deserdict = orjson.loads
            else:
                default_serdict = pickle.dumps
                default_deserdict = pickle.loads

            cls._metadata = NumpyDataModelMetadata(
                cls=cls,
                props=cls.__slots__,
                array_props=array_props,
                array2d_props=array2d_props,
                index_props=index_props,
                array_prop_idxs=array_prop_idxs,
                index_prop_idxs=index_prop_idxs,
                default_serdict=default_serdict,
                default_deserdict=default_deserdict,
            )

    def __len__(self):
        """Get length of the array"""
        metadata: NumpyDataModelMetadata = self._metadata  # type: ignore
        return len(getattr(self, metadata.array_props[0]))

    def shallow_clone(self):
        return self.__class__(*(getattr(self, name) for name in self.__slots__))

    def replace(self, field: str, value: np.ndarray):
        assert getattr(self, field).shape == value.shape
        newobj = self.shallow_clone()
        setattr(newobj, field, value)
        return newobj

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
        metadata: NumpyDataModelMetadata = self._metadata  # type: ignore
        sortedvalue = getattr(self, sortby)
        sortedindex = np.argsort(sortedvalue, kind=kind)
        if sortorder == "desc":
            sortedindex = sortedindex[::-1]

        for field in metadata.array_props:
            value = getattr(self, field)
            value[i:j] = value[i:j][sortedindex]

    def replace_subset_(
        self, field: str, value: Union[Sequence, np.ndarray], i: int, j: int
    ):
        getattr(self, field)[i:j] = value

    def save(self, file: Path, compression: Optional[str] = None):
        metadata = self._metadata
        if metadata is None:
            raise Exception(
                f"{self.__class__.__qualname__}.init() must be called before usage to finish setting up the schema"
            )

        compression = compression or get_compression(file)
        cols = {name: getattr(self, name) for name in metadata.array_props}
        for name in metadata.array2d_props:
            array2d = cols[name]
            cols[name] = array2d[:, 0]
            for i in range(1, array2d.shape[1]):
                cols[f"{name}_{i}"] = array2d[:, i]

        pq.write_table(
            pa.table(cols),
            str(file),
            compression=compression or "NONE",
        )

        if len(metadata.index_props) > 0:
            index_file = file.parent / (
                file.stem
                + (f".index.{compression}" if compression is not None else ".index")
            )
            with get_open_fn(index_file)(str(index_file), "wb") as f:
                f.write(struct.pack("<I", len(metadata.index_props)))
                for i, name in enumerate(metadata.index_props):
                    if metadata.index_prop_idxs[i][1] is not None:
                        serindex = getattr(self, name).to_bytes()
                    else:
                        serindex = metadata.default_serdict(getattr(self, name))
                    f.write(struct.pack("<I", len(serindex)))
                    f.write(serindex)

    @classmethod
    def load(cls, file: Path, compression: Optional[str] = None):
        metadata = cls._metadata
        if metadata is None:
            raise Exception(
                f"{cls.__qualname__}.init() must be called before usage to finish setting up the schema"
            )

        compression = compression or get_compression(file)
        index_file = file.parent / (
            file.stem
            + (f".index.{compression}" if compression is not None else ".index")
        )
        indices = []

        if index_file.exists():
            with get_open_fn(index_file)(str(index_file), "rb") as f:
                n_indices = struct.unpack("<I", f.read(4))[0]
                for i in range(n_indices):
                    size = struct.unpack("<I", f.read(4))[0]
                    index_type = metadata.index_prop_idxs[i][1]
                    if index_type is not None:
                        index = index_type.from_bytes(f.read(size))
                    else:
                        index = metadata.default_deserdict(f.read(size))
                    indices.append(index)

        tbl = pq.read_table(str(file))
        kwargs = {}
        if len(metadata.array2d_props) == 0:
            for name in metadata.array_props:
                kwargs[name] = tbl.column(name).to_numpy()
        else:
            columns: List[str] = tbl.column_names
            for name in metadata.array2d_props:
                newcols = []
                names = []
                for col in columns:
                    if col.startswith(name):
                        names.append(col)
                    else:
                        newcols.append(col)
                names.sort()
                array2d = np.stack([tbl.column(s).to_numpy() for s in names], axis=1)
                kwargs[name] = array2d
                columns = newcols

            for name in columns:
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


@dataclass
class NumpyDataModelContainer:
    def save(self, dir: Path):
        for field in fields(self):
            file = dir / f"{field.name}.parq"
            getattr(self, field.name).save(file)

    @classmethod
    def load(cls, dir: Path):
        assert is_dataclass(cls)
        type_hints: dict[str, type] = get_type_hints(cls)
        kwargs = {}
        for field in fields(cls):
            file = dir / f"{field.name}.parq"
            fieldtype = type_hints[field.name]
            assert issubclass(fieldtype, NumpyDataModel)
            kwargs[field.name] = fieldtype.load(file)

        return NumpyDataModelContainer(**kwargs)


class ContiguousIndexChecker:
    """A helper class to check if the order of range of items in the numpy data model's index is contiguous"""

    def __init__(self, start: int = 0):
        self.start = start

    def next(self, start: int, end: int):
        if self.start != start:
            raise ValueError(
                f"Encounter a range that is not continuous from the previous range. Expected {self.start}, got {start}"
            )
        if start > end:
            raise ValueError(
                f"The provided range is invalid. Start {start} is greater than end {end}"
            )
        self.start = end
        return self

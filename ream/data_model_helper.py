from __future__ import annotations

import pickle
import struct
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import (
    Callable,
    Generic,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_origin,
    get_type_hints,
)
from copy import deepcopy
import numpy as np
import orjson
import pyarrow as pa
import pyarrow.parquet as pq
from nptyping.ndarray import NDArrayMeta  # type: ignore
from nptyping.shape_expression import get_dimensions  # type: ignore
from ream.helper import has_dict_with_nonstr_keys
from serde.helper import get_compression, get_open_fn
from tqdm import tqdm


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
    # list of properties of the data model
    props: Sequence[str]
    # list of properties that are numpy arrays
    array_props: Sequence[str]
    # list of properties that are numpy 2D arrays
    array2d_props: Sequence[str]
    # list of properties that are index (dictionary, list, Index)
    index_props: Sequence[str]
    # position of each property, which is in the array_props, in the props list.
    array_prop_idxs: Sequence[int]
    # position of each property, which is in the index_props, in the props list.
    index_prop_idxs: Sequence[Tuple[int, Optional[Type[Index]]]]
    # default serialize function to serialize the index
    default_serdict: Callable[[dict], bytes]
    # default deserialize function to deserialize the index
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


class NumpyDataModelHelper:
    """Generally contains helpers for a numpy data model that has a single index, which is a dictionary.
    The dictionary can have multiple levels, each level store a tuple of (start, end, nested_index),
    and the dictionaries at the lowest level store just (start, end).

    However, some helper methods can be used for other cases as well.
    """

    @classmethod
    def stack(cls, npmodels: list[NumpyDataModel], keys: Sequence[str | int]):
        """This method concatenates a list of numpy data models of the same type into a single numpy data model.
        The new index is a new dictionary that each key provided maps to the index of each model in the provided order.
        """
        if len(npmodels) == 0:
            raise ValueError("The provided list of numpy data models is empty")

        metadata: NumpyDataModelMetadata = npmodels[0]._metadata  # type: ignore
        assert len(set(npmodel.__class__ for npmodel in npmodels)) == 1
        assert len(metadata.index_props) == 1

        arrays = {prop: [] for prop in metadata.array_props}
        index = {}
        counter = 0
        for npmodel, key in zip(npmodels, keys):
            for prop, values in arrays.items():
                arr = getattr(npmodel, prop)
                values.append(arr)
                size = len(arr)
            size = len(next(iter(arrays.values()))[-1])
            npmodel_index = deepcopy(getattr(npmodel, metadata.index_props[0]))
            cls.offset_index(npmodel_index, counter)
            index[key] = (counter, counter + size, npmodel_index)
            counter += size

        for prop, arr in arrays.items():
            arrays[prop] = np.concatenate(arr)
        arrays[metadata.index_props[0]] = index
        return npmodels[0].__class__(**arrays)

    @classmethod
    def to_df(cls, npmodel: NumpyDataModel):
        """This method converts a numpy data model to a pandas dataframe. Assuming that the index is a dictionary and it only has one index"""
        import pandas as pd

        metadata: NumpyDataModelMetadata = npmodel._metadata  # type: ignore
        assert len(metadata.index_props) <= 1
        if len(metadata.index_props) == 1:
            index = getattr(npmodel, metadata.index_props[0])
            assert isinstance(index, dict)
            pdindex = pd.MultiIndex.from_tuples(cls.index_to_array(index))
            return pd.DataFrame(
                {name: getattr(npmodel, name) for name in metadata.array_props},
                index=pdindex,
            )

        return pd.DataFrame(
            {name: getattr(npmodel, name) for name in metadata.array_props}
        )

    @classmethod
    def from_df(cls, df, model_cls: Type[NumpyDataModel]):
        """This method converts a pandas dataframe to a numpy data model. The model must have only one dictionary-index"""
        metadata: NumpyDataModelMetadata = model_cls._metadata  # type: ignore
        assert len(metadata.index_props) == 1

        index = cls.index_from_array(df.index.to_list())
        kwargs = {name: df[name].to_numpy() for name in metadata.array_props}
        kwargs[metadata.index_props[0]] = index
        return model_cls(**kwargs)

    @classmethod
    def sync_index(cls, source: dict, target: dict):
        """Sync two index dictionaries. The source index must be a subset of the target index.
        Any entry in the target index that is not in the source index will be added to the source with empty range.
        The index produced by this method does not have the range contiguous.
        """
        if len(target) == 0:
            return

        for key, value in target.items():
            if isinstance(value, tuple):
                if len(value) == 2:
                    if key not in source:
                        source[key] = (0, 0)
                else:
                    if key not in source:
                        source[key] = (0, 0, {})
                    cls.sync_index(source[key][2], value[2])
            else:
                if key not in source:
                    source[key] = {}
                cls.sync_index(source[key], value)

    @classmethod
    def offset_index(cls, index: dict, offset: int):
        """This method offsets the index of a numpy data model by a given offset"""
        for key, value in index.items():
            if isinstance(value, tuple):
                if len(value) == 2:
                    index[key] = (value[0] + offset, value[1] + offset)
                else:
                    index[key] = (
                        value[0] + offset,
                        value[1] + offset,
                        cls.offset_index(value[2], offset),
                    )
            else:
                index[key] = cls.offset_index(value, offset)
        return index

    @classmethod
    def is_range_index_valid(cls, npmodel: NumpyDataModel):
        """Check if the index of a numpy data model is valid: the index is contiguous, no overlapping
        and covers the entire range of the numpy arrays.

        This function assumes that the index is a dictionary (or instance of Index[dict]), and each value in the index
        is either (start, end) if the index is a single-level index or (start, end, nested_index) if the index is a
        multi-level index.
        """
        metadata: NumpyDataModelMetadata = npmodel._metadata  # type: ignore
        size = len(npmodel)
        for name in metadata.index_props:
            index = getattr(npmodel, name)
            if isinstance(index, Index):
                index = index.index
            if not isinstance(index, dict):
                raise TypeError(
                    f"Index {name} is not a dictionary. This function only supports dictionary index"
                )

            # figure out the index level
            index_level = cls._get_range_index_level(npmodel, index)
            if index_level == 0 and size != 0:
                raise ValueError("Index is empty but the numpy array is not empty")

            indices = [index]
            for level in tqdm(
                range(index_level), desc="check index at different levels"
            ):
                cic = ContiguousIndexChecker()
                next_indices = []
                for idx in tqdm(indices):
                    if len(idx) == 0:
                        continue
                    for elem in idx.values():
                        cic.next(elem[0], elem[1])
                        if len(elem) == 3:
                            next_indices.append(elem[2])
                if not cic.start == size:
                    raise ValueError(
                        f"Index is not covered all elements in the array at level {level}. Expected {size} but got {cic.start}"
                    )
                indices = next_indices

    @classmethod
    def _get_range_index_level(cls, npmodel: NumpyDataModel, index: dict):
        """Get the level of a range index. This function assumes that the index is a dictionary, and each value in the index
        is either (start, end) if the index is a single-level index or (start, end, nested_index) if the index is a
        multi-level index."""
        if len(index) == 0:
            return 0

        level = []
        for key, value in index.items():
            if not isinstance(value, (list, tuple)):
                raise TypeError(f"Index element {value} is not a sequence")
            if len(value) == 2:
                if not isinstance(value[0], int) or not isinstance(value[1], int):
                    raise TypeError(
                        f"Index element {value} does not have the form: (start, end)"
                    )
                level.append(1)
            elif len(value) == 3:
                if (
                    not isinstance(value[0], int)
                    or not isinstance(value[1], int)
                    and not isinstance(value[2], dict)
                ):
                    raise TypeError(
                        f"Index element {value} does not have the form: (start, end, nested_index)"
                    )
                level.append(cls._get_range_index_level(npmodel, value[2]) + 1)
            else:
                raise TypeError(
                    f"Index element {value} does not have the form: (start, end) or (start, end, nested_index)"
                )

        level = set(level)
        if len(level) > 1:
            raise ValueError(
                "The index is not a balanced multi-level index as some elements have different levels"
            )
        return level.pop()

    @classmethod
    def to_nested_list(cls, arr: np.ndarray, index: dict):
        """Convert a numpy array to a nested list. The nested level is determined by the index"""
        out = []
        for key, value in index.items():
            if isinstance(value, tuple):
                if len(value) == 2:
                    out.append(arr[value[0] : value[1]])
                else:
                    out.append(cls.to_nested_list(arr, value[2]))
            elif isinstance(value, dict):
                out.append(cls.to_nested_list(arr, value))
            else:
                raise TypeError(
                    f"Index element {value} does not have the form: (start, end) or (start, end, nested_index)"
                )
        return out

    @classmethod
    def index_to_array(cls, index: dict, prefix: tuple[str, ...] = tuple()):
        """Convert a dictionary index to an array that is ready to be loaded into
        pandas's multiindex via pd.MultiIndex.from_tuples
        """
        arr = []
        for key, value in index.items():
            if isinstance(value, (tuple, list)):
                if len(value) == 2:
                    for i in range(value[0], value[1]):
                        arr.append(prefix + (key, i))
                else:
                    arr.extend(cls.index_to_array(value[2], prefix + (key,)))
            elif isinstance(value, dict):
                arr.extend(cls.index_to_array(value, prefix + (key,)))
            else:
                raise TypeError(
                    f"Index element {value} does not have the form: (start, end) or (start, end, nested_index)"
                )
        return arr

    @classmethod
    def index_from_array(cls, arr: list[tuple]):
        """Convert an array of tuples back to a dictionary index. The array is generated by index_to_array
        or pandas.MultiIndex.to_list"""
        index = {}
        if len(arr) == 0:
            return index

        if not isinstance(arr[0], (tuple, list)) and len(arr[0]) == 1:
            raise ValueError("Don't need a dictionary index for a single-level array")

        pending_lst = []
        for i, elem in enumerate(arr):
            ptr = index
            for j in range(len(elem) - 2):
                if elem[j] not in ptr:
                    ptr[elem[j]] = [i, i, {}]
                    pending_lst.append((ptr, elem[j]))
                item = ptr[elem[j]]
                item[1] += 1
                ptr = item[2]
            if elem[-2] not in ptr:
                ptr[elem[-2]] = [i, i]
                pending_lst.append((ptr, elem[-2]))
            ptr[elem[-2]][1] += 1

        for obj, key in pending_lst:
            obj[key] = tuple(obj[key])

        return index

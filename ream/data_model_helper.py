from __future__ import annotations

import os
import pickle
import struct
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from io import BufferedReader, BytesIO
from pathlib import Path, PosixPath, WindowsPath
from typing import (
    BinaryIO,
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

import numpy as np
import orjson
import pyarrow as pa
import pyarrow.parquet as pq
import serde.json
from nptyping import NDArray, Shape
from nptyping.ndarray import NDArrayMeta  # type: ignore
from nptyping.shape_expression import get_dimensions  # type: ignore
from nptyping.typing_ import Number
from serde.helper import AVAILABLE_COMPRESSIONS, get_filepath, get_open_fn
from tqdm import tqdm
from typing_extensions import Self

from ream.helper import get_classpath, has_dict_with_nonstr_keys, import_attr

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

    def shallow_clone(self) -> Self:
        return self.__class__(*(getattr(self, name) for name in self.__slots__))

    def replace(self, field: str, value: np.ndarray):
        assert getattr(self, field).shape == value.shape, (
            getattr(self, field).shape,
            value.shape,
        )
        newobj = self.shallow_clone()
        setattr(newobj, field, value)
        return newobj

    @classmethod
    def concatenate(
        cls,
        models: list[NumpyDataModel],
        merge_index: Optional[
            dict[str, Callable[[list[NumpyDataModel]], dict | list | Index]]
        ] = None,
    ):
        """Concatenate a list of NumpyDataModel objects into one object

        Args:
            models: a list of NumpyDataModel objects of the same class
            merge_index: functions to merge an index, each key is the name of the index property. It's
                optional if there is no index
        """
        metadata = cls._metadata
        if metadata is None:
            raise Exception(
                f"{cls.__qualname__}.init() must be called before usage to finish setting up the schema"
            )

        merge_index = merge_index or {}

        attrs = {}
        for prop in metadata.array_props:
            attrs[prop] = np.concatenate([getattr(m, prop) for m in models])

        for prop in metadata.index_props:
            attrs[prop] = merge_index[prop](models)

        return cls(**attrs)

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

    def save(self, dir: Path, compression: Optional[AVAILABLE_COMPRESSIONS] = None):
        """Save the data model to a directory containing 2 files: `data.parq` and `index.bin`.

        Note: this function is carefully written so that the two files can be buffered to concatenate
        multiple models to a single file using VirtualDir class. See `batch_save` for details.

        Args:
            dir: the directory to save the data model. Can be either Path or VirtualDir
        """
        metadata = self._metadata
        if metadata is None:
            raise Exception(
                f"{self.__class__.__qualname__}.init() must be called before usage to finish setting up the schema"
            )

        cols = {name: getattr(self, name) for name in metadata.array_props}
        for name in metadata.array2d_props:
            array2d = cols[name]
            cols.pop(name)
            for i in range(array2d.shape[1]):
                cols[f"{name}_{i}"] = array2d[:, i]

        dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(cols),
            dir / "data.parq",
            compression=compression or "NONE",
        )

        if len(metadata.index_props) > 0:
            index_filename = get_filepath("index.bin", compression)
            with get_open_fn(index_filename)(dir / index_filename, "wb") as f:
                f.write(struct.pack("<I", len(metadata.index_props)))
                for i, name in enumerate(metadata.index_props):
                    if metadata.index_prop_idxs[i][1] is not None:
                        serindex = getattr(self, name).to_bytes()
                    else:
                        serindex = metadata.default_serdict(getattr(self, name))
                    f.write(struct.pack("<I", len(serindex)))
                    f.write(serindex)

    @classmethod
    def batch_save(
        cls,
        batch: Sequence[NumpyDataModel],
        dir: Path,
        compression: Optional[AVAILABLE_COMPRESSIONS],
    ):
        """Save a list of numpy data models into a directory. Different from the save method of numpy data model,
        data of multiple models will be concatenated into a single file.
        """
        with BatchFileManager() as filemanager:
            virdir = VirtualDir(dir, filemanager=filemanager, mode="write")
            for npmodel in batch:
                npmodel.save(virdir, compression)
                filemanager.flush()

    @classmethod
    def load(cls, dir: Path, compression: Optional[AVAILABLE_COMPRESSIONS] = None):
        metadata = cls._metadata
        if metadata is None:
            raise Exception(
                f"{cls.__qualname__}.init() must be called before usage to finish setting up the schema"
            )

        indices = []

        if len(metadata.index_props) > 0:
            index_filename = get_filepath("index.bin", compression)
            with get_open_fn(index_filename)(dir / index_filename, "rb") as f:
                n_indices = struct.unpack("<I", f.read(4))[0]
                for i in range(n_indices):
                    size = struct.unpack("<I", f.read(4))[0]
                    index_type = metadata.index_prop_idxs[i][1]
                    if index_type is not None:
                        index = index_type.from_bytes(f.read(size))
                    else:
                        index = metadata.default_deserdict(f.read(size))
                    indices.append(index)

        tbl = pq.read_table(dir / "data.parq")
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
                names.sort(key=lambda x: int(x.replace(name + "_", "")))
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

    @classmethod
    def batch_load(
        cls, dir: Path, compression: Optional[AVAILABLE_COMPRESSIONS] = None
    ):
        with BatchFileManager() as filemanager:
            virdir = VirtualDir(dir, filemanager=filemanager, mode="read")
            output = []
            while True:
                output.append(cls.load(virdir, compression))
                filemanager.flush()
                if filemanager.is_all_read_done():
                    break
            return output


@dataclass
class NumpyDataModelContainer:
    def save(self, dir: Path, compression: Optional[AVAILABLE_COMPRESSIONS] = None):
        index_props = []
        for field in fields(self):
            obj = getattr(self, field.name)
            if isinstance(obj, (NumpyDataModel, NumpyDataModelContainer)):
                obj.save(dir / field.name, compression)
            else:
                index_props.append((field.name, obj))

        if len(index_props) > 0:
            index_filename = get_filepath("index.bin", compression)
            with get_open_fn(index_filename)(dir / index_filename, "wb") as f:
                f.write(struct.pack("<I", len(index_props)))
                for i, (name, obj) in enumerate(index_props):
                    if isinstance(obj, Index):
                        serindex = obj.to_bytes()
                    else:
                        serindex = pickle.dumps(obj)

                    sername = name.encode()
                    f.write(struct.pack("<I", len(sername)))
                    f.write(sername)
                    f.write(struct.pack("<I", len(serindex)))
                    f.write(serindex)

    @classmethod
    def batch_save(
        cls,
        containers: Sequence[C],
        dir: Path,
        compression: Optional[AVAILABLE_COMPRESSIONS] = None,
    ):
        if len(containers) == 0:
            raise ValueError("containers must not be empty")

        with BatchFileManager() as filemanager:
            virdir = VirtualDir(dir, filemanager=filemanager, mode="write")
            for container in containers:
                container.save(virdir, compression)
                filemanager.flush()

    @classmethod
    def load(cls, dir: Path, compression: Optional[AVAILABLE_COMPRESSIONS] = None):
        assert is_dataclass(cls)
        type_hints: dict[str, type] = get_type_hints(cls)
        kwargs = {}
        index_props = []
        for field in fields(cls):
            fieldtype = type_hints[field.name]
            if (ori_type := get_origin(fieldtype)) is not None:
                fieldtype = ori_type

            if issubclass(fieldtype, (NumpyDataModel, NumpyDataModelContainer)):
                kwargs[field.name] = fieldtype.load(dir / field.name, compression)
            else:
                index_props.append(field)

        if len(index_props) > 0:
            index_filename = get_filepath("index.bin", compression)
            n_npmodel = len(kwargs)

            with get_open_fn(index_filename)(dir / index_filename, "rb") as f:
                n_indices = struct.unpack("<I", f.read(4))[0]
                for i in range(n_indices):
                    size = struct.unpack("<I", f.read(4))[0]
                    name = f.read(size).decode()
                    size = struct.unpack("<I", f.read(4))[0]
                    obj = f.read(size)
                    obj_type = type_hints[name]
                    if (ori_type := get_origin(obj_type)) is not None:
                        obj_type = ori_type

                    if issubclass(obj_type, Index):
                        obj = obj_type.from_bytes(obj)
                    else:
                        obj = pickle.loads(obj)
                    kwargs[name] = obj

            assert len(kwargs) == n_npmodel + len(index_props)

        return cls(**kwargs)

    @classmethod
    def batch_load(
        cls, dir: Path, compression: Optional[AVAILABLE_COMPRESSIONS] = None
    ):
        with BatchFileManager() as filemanager:
            virdir = VirtualDir(dir, filemanager=filemanager, mode="read")
            output = []
            while True:
                output.append(cls.load(virdir, compression))
                filemanager.flush()
                if filemanager.is_all_read_done():
                    break
            return output


C = TypeVar("C", bound=NumpyDataModelContainer)


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


class BatchFileManager:
    def __init__(self):
        self.open_write_files: dict[str, BinaryIO] = {}
        self.open_read_files: dict[str, BufferedReader] = {}
        self.pend_read_files: dict[str, BinaryIO] = {}
        self.pend_write_files: dict[str, tuple[BytesIO, pa.PythonFile]] = {}

    def create(self, filepath: str):
        if filepath in self.pend_write_files:
            raise Exception(
                "Cannot request the same file twice. This is to prevent data corruption. Finishing writting data to the file and flushing it before requesting it again."
            )
        buf = BytesIO()
        self.pend_write_files[filepath] = (buf, pa.PythonFile(buf))
        if filepath not in self.open_write_files:
            self.open_write_files[filepath] = open(filepath, "wb")

        return self.pend_write_files[filepath][1]

    def read(self, filepath: str) -> BinaryIO:
        if filepath in self.pend_read_files:
            raise Exception(
                "Cannot request the same file twice. This is to prevent data corruption. Finishing reading data to the file and flushing it before requesting it again."
            )

        if filepath not in self.open_read_files:
            self.open_read_files[filepath] = open(filepath, "rb")

        file = self.open_read_files[filepath]
        size = struct.unpack("<I", file.read(4))[0]
        buf = file.read(size)

        self.pend_read_files[filepath] = BytesIO(buf)
        return self.pend_read_files[filepath]

    def __enter__(self):
        assert len(self.open_write_files) == 0 and len(self.open_read_files) == 0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()
        for file in self.open_write_files.values():
            file.close()
        for file in self.open_read_files.values():
            file.close()
        self.open_read_files.clear()
        self.open_write_files.clear()

    def flush(self):
        for filepath, (buf, file) in self.pend_write_files.items():
            file = self.open_write_files[filepath]
            buf = buf.getbuffer()
            file.write(struct.pack("<I", buf.nbytes))
            file.write(buf)

        self.pend_write_files.clear()
        self.pend_read_files.clear()

    def is_all_read_done(self) -> bool:
        return all(len(file.peek(1)) == 0 for file in self.open_read_files.values())


class VirtualDir(Path):
    __slots__ = ("filemanager", "mode")

    _flavour = getattr(WindowsPath if os.name == "nt" else PosixPath, "_flavour")
    filemanager: BatchFileManager
    mode: Literal["read", "write"]

    def __new__(cls, *args, **kwargs):
        object = Path.__new__(cls, *args, **kwargs)
        object.filemanager = kwargs["filemanager"]
        object.mode = kwargs["mode"]
        return object

    def __truediv__(self, key: str) -> Union[pa.PythonFile, VirtualDir]:
        key = str(key)
        if key.find(".") == -1:
            return VirtualDir(
                str(self), key, filemanager=self.filemanager, mode=self.mode
            )

        filepath = str(super().__truediv__(key))
        if self.mode == "write":
            return self.filemanager.create(filepath)
        else:
            return self.filemanager.read(filepath)


class NumpyDataModelHelper:
    """Generally contains helpers for a numpy data model that has a single index, which is a dictionary.
    The dictionary can have multiple levels, each level store a tuple of (start, end, nested_index),
    and the dictionaries at the lowest level store just (start, end).

    However, some helper methods can be used for other cases as well.
    """

    @classmethod
    def create_simple_index(
        cls, names: list[str], npmodels: list[NumpyDataModel]
    ) -> dict[str, tuple[int, int]]:
        """Create a simple index that map each name to the index range of the corresponding numpy data model"""
        index = {}
        offset = 0
        for name, npmodel in zip(names, npmodels):
            size = len(npmodel)
            index[name] = (offset, offset + size)
            offset += size
        return index

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

        arrays: dict = {prop: [] for prop in metadata.array_props}
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
            arrays[prop] = np.concatenate(arr)  # type: ignore
        arrays[metadata.index_props[0]] = index  # type: ignore
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


class SingleNumpyArray(NumpyDataModel):
    __slots__ = ["value"]

    value: NDArray[Shape["*"], Number]

    def __init__(self, value: NDArray[Shape["*"], Number]):
        self.value = value


class Single2DNumpyArray(NumpyDataModel):
    __slots__ = ["value"]

    value: NDArray[Shape["*,*"], Number]

    def __init__(self, value: NDArray[Shape["*,*"], Number]):
        self.value = value


SingleNumpyArray.init()
Single2DNumpyArray.init()


def ser_dict_array(
    odict: dict[str, NumpyDataModel],
    dir: Path,
    compression: Optional[AVAILABLE_COMPRESSIONS],
):
    classes = {}
    for key, value in odict.items():
        value.save(dir / key, compression=compression)
        classes[key] = get_classpath(value.__class__)
    serde.json.ser(classes, dir / "metadata.json")


def deser_dict_array(dir: Path, compression: Optional[AVAILABLE_COMPRESSIONS]):
    classes = serde.json.deser(dir / "metadata.json")
    objects = {}
    for key, cls in classes.items():
        objects[key] = import_attr(cls).load(dir / key, compression=compression)
    return objects


# dir = VirtualDir("/tmp", filetrack=FileTrack())
# print(dir.name2file, dir.filetrack)
# dir / "test.h5"
# print(dir.name2file, dir.filetrack)
# subdir = dir / "abc"
# print(dir.name2file, dir.filetrack)
# print(type(subdir))
# print(subdir.name2file, subdir.filetrack)
# # print((VirtualDir("/tmp") / "test.h5").name2file)

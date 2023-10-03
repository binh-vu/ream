from __future__ import annotations

import pickle
import struct
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterator,
    Literal,
    Optional,
    Sequence,
    TypeVar,
    get_origin,
    get_type_hints,
)

import orjson
import polars as pl
from polars.type_aliases import ParquetCompression
from serde.helper import get_filepath, get_open_fn

from ream.data_model_helper._container import DataSerdeMixin
from ream.data_model_helper._index import Index
from ream.helper import Compression, has_dict_with_nonstr_keys, to_serde_compression


@dataclass
class PolarDataModelMetadata:
    cls: type
    # list of properties of the data model
    props: Sequence[str]
    # list of properties that are pl.DataFrame
    df_props: Sequence[str]
    # list of properties that are index (dictionary, list, Index)
    index_props: Sequence[str]
    # position of each property, which is in the index_props, in the props list.
    index_prop_idxs: Sequence[tuple[int, Optional[type[Index]]]]
    # default serialize function to serialize the index
    default_serdict: Callable[[dict], bytes]
    # default deserialize function to deserialize the index
    default_deserdict: Callable[[bytes], dict]


class PolarDataModel(DataSerdeMixin):
    __slots__ = []
    _metadata: Optional[PolarDataModelMetadata] = None

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
            df_props = []
            index_props = []
            index_prop_idxs = []

            # check if we can use orjson
            useorjson = True

            for i, name in enumerate(cls.__slots__):
                if name not in anns:
                    raise ValueError(
                        f"Attribute {name} of {cls} must have a type annotation, even if the type is Any"
                    )

                ann = anns[name]
                if ann is pl.DataFrame:
                    # it it a dataframe
                    df_props.append(name)
                else:
                    ann_origin = get_origin(ann) or ann
                    if issubclass(ann_origin, (dict, list, Index)):
                        index_props.append(name)
                        if not issubclass(ann_origin, Index):
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

            cls._metadata = PolarDataModelMetadata(
                cls=cls,
                props=cls.__slots__,
                df_props=df_props,
                index_props=index_props,
                index_prop_idxs=index_prop_idxs,
                default_serdict=default_serdict,
                default_deserdict=default_deserdict,
            )

    def save(
        self,
        loc: Path,
        compression: Optional[Compression] = None,
        compression_level: Optional[int] = None,
    ):
        metadata = self._metadata
        if metadata is None:
            raise Exception(
                f"{self.__class__.__qualname__}.init() must be called before usage to finish setting up the schema"
            )

        loc.mkdir(parents=True, exist_ok=True)
        for name in metadata.df_props:
            df: pl.DataFrame = getattr(self, name)
            df.write_parquet(
                loc / f"{name}.parq",
                compression=to_polar_compression(compression),
                compression_level=compression_level,
            )

        if len(metadata.index_props) > 0:
            index_filename = get_filepath(
                "index.bin", to_serde_compression(compression)
            )
            with get_open_fn(index_filename)(loc / index_filename, "wb") as f:
                f.write(struct.pack("<I", len(metadata.index_props)))
                for i, name in enumerate(metadata.index_props):
                    if metadata.index_prop_idxs[i][1] is not None:
                        serindex = getattr(self, name).to_bytes()
                    else:
                        serindex = metadata.default_serdict(getattr(self, name))
                    f.write(struct.pack("<I", len(serindex)))
                    f.write(serindex)

    @classmethod
    def load(cls, loc: Path, compression: Optional[Compression] = None):
        metadata = cls._metadata
        if metadata is None:
            raise Exception(
                f"{cls.__qualname__}.init() must be called before usage to finish setting up the schema"
            )

        kwargs = {}

        if len(metadata.index_props) > 0:
            index_filename = get_filepath(
                "index.bin", to_serde_compression(compression)
            )
            with get_open_fn(index_filename)(loc / index_filename, "rb") as f:
                n_indices = struct.unpack("<I", f.read(4))[0]
                for i in range(n_indices):
                    size = struct.unpack("<I", f.read(4))[0]
                    index_type = metadata.index_prop_idxs[i][1]
                    if index_type is not None:
                        index = index_type.from_bytes(f.read(size))
                    else:
                        index = metadata.default_deserdict(f.read(size))
                    kwargs[metadata.index_props[i]] = index

        for name in metadata.df_props:
            df = pl.read_parquet(loc / f"{name}.parq")
            kwargs[name] = df

        return cls(**kwargs)


I = TypeVar("I")


@dataclass
class SingleLevelIndexedPLDataFrame(PolarDataModel):
    __slots__ = ["index", "value"]

    index: dict[str, tuple[int, int]]
    value: pl.DataFrame

    def get_df(self, key: str) -> pl.DataFrame:
        start, end = self.index[key]
        return self.value.slice(start, end - start)

    def __iter__(self) -> Iterator[pl.DataFrame]:
        return (
            self.value.slice(start, end - start) for start, end in self.index.values()
        )

    def __len__(self) -> int:
        return len(self.value)


SingleLevelIndexedPLDataFrame.init()


@dataclass
class SinglePolarDataFrame(PolarDataModel):
    __slots__ = ["value"]

    value: pl.DataFrame

    def __len__(self) -> int:
        return len(self.value)


SinglePolarDataFrame.init()


def to_polar_compression(
    compression: Optional[Compression],
) -> ParquetCompression:
    if compression is None:
        return "uncompressed"

    seq: Sequence[ParquetCompression] = ["snappy", "gzip", "lz4", "zstd"]
    if compression in seq:
        return compression

    raise Exception(f"Not supported compression: {compression}")

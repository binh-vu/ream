from __future__ import annotations

import pickle
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, is_dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional, get_origin, get_type_hints

import orjson
import serde.pickle
from ream.data_model_helper._io_helper import fread
from ream.helper import Compression, to_serde_compression
from serde.helper import get_filepath
from typing_extensions import Self


class DataSerdeMixin(ABC):
    @abstractmethod
    def save(
        self,
        loc: Path,
        compression: Optional[Compression] = None,
        compression_level: Optional[int] = None,
    ) -> None: ...

    @abstractmethod
    def ser(
        self,
        compression: Optional[Compression] = None,
        compression_level: Optional[int] = None,
    ) -> bytes: ...

    @classmethod
    @abstractmethod
    def load(cls, loc: Path, compression: Optional[Compression] = None) -> Self: ...

    @classmethod
    @abstractmethod
    def deser(cls, data: bytes) -> Self: ...


@dataclass
class DataContainer(DataSerdeMixin):
    def save(
        self,
        loc: Path,
        compression: Optional[Compression] = None,
        compression_level: Optional[int] = None,
    ):
        for field in fields(self):
            obj = getattr(self, field.name)
            if isinstance(obj, DataSerdeMixin):
                obj.save(
                    loc / field.name,
                    compression=compression,
                    compression_level=compression_level,
                )
            else:
                serde.pickle.ser(
                    obj,
                    loc
                    / get_filepath(
                        f"{field.name}.pkl", to_serde_compression(compression)
                    ),
                )

    def ser(
        self,
        compression: Optional[Compression] = None,
        compression_level: Optional[int] = None,
    ) -> bytes:
        out = BytesIO()

        obj_fields = fields(self)

        tmp = orjson.dumps([field.name for field in obj_fields])
        out.write(struct.pack("<I", len(tmp)))
        out.write(tmp)

        for field in obj_fields:
            obj = getattr(self, field.name)
            if isinstance(obj, DataSerdeMixin):
                tmp = obj.ser(
                    compression=compression,
                    compression_level=compression_level,
                )
            else:
                tmp = pickle.dumps(obj)

            out.write(struct.pack("<I", len(tmp)))
            out.write(tmp)
        out.flush()
        return out.getvalue()

    @classmethod
    def load(cls, loc: Path, compression: Optional[Compression] = None):
        assert is_dataclass(cls)
        type_hints: dict[str, type] = get_type_hints(cls)
        kwargs = {}

        for field in fields(cls):
            fieldtype = type_hints[field.name]
            if (ori_type := get_origin(fieldtype)) is not None:
                fieldtype = ori_type

            if issubclass(fieldtype, DataSerdeMixin):
                kwargs[field.name] = fieldtype.load(loc / field.name, compression)
            else:
                kwargs[field.name] = serde.pickle.deser(
                    loc
                    / get_filepath(
                        f"{field.name}.pkl", to_serde_compression(compression)
                    )
                )
        return cls(**kwargs)

    @classmethod
    def deser(cls, data: bytes) -> Self:
        assert is_dataclass(cls)
        type_hints: dict[str, type] = get_type_hints(cls)
        kwargs = {}

        obj_fields = fields(cls)

        f = fread(data)

        size = struct.unpack("<I", f.read(4))[0]
        assert [x.name for x in obj_fields] == orjson.loads(f.read(size))

        for field in obj_fields:
            fieldtype = type_hints[field.name]
            if (ori_type := get_origin(fieldtype)) is not None:
                fieldtype = ori_type

            size = struct.unpack("<I", f.read(4))[0]
            fieldval = f.read(size)
            if issubclass(fieldtype, DataSerdeMixin):
                kwargs[field.name] = fieldtype.deser(fieldval)
            else:
                kwargs[field.name] = pickle.loads(fieldval)

        assert f.is_eof()
        return cls(**kwargs)

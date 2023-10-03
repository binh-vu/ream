from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Optional, get_origin, get_type_hints

import serde.pickle
from serde.helper import get_filepath
from typing_extensions import Self

from ream.helper import Compression, to_serde_compression


class DataSerdeMixin(ABC):
    @abstractmethod
    def save(
        self,
        loc: Path,
        compression: Optional[Compression] = None,
        compression_level: Optional[int] = None,
    ) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, loc: Path, compression: Optional[Compression] = None) -> Self:
        ...


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

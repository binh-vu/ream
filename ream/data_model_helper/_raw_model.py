from __future__ import annotations

from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq
from ream.data_model_helper._container import DataSerdeMixin
from ream.data_model_helper._numpy_model import to_pyarrow_compression
from ream.helper import Compression, to_serde_compression
from typing_extensions import Self


class DictList(DataSerdeMixin):
    def append(self, **kwargs):
        if len(self.__dict__) > 0:
            for key, value in kwargs.items():
                self.__dict__[key].append(value)
        else:
            for key, value in kwargs.items():
                self.__dict__[key] = [value]

    def save(
        self,
        loc: Path,
        compression: Optional[Compression] = None,
        compression_level: Optional[int] = None,
    ) -> None:
        loc.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(self.__dict__),
            loc / "data.parq",
            compression=to_pyarrow_compression(compression),
            compression_level=compression_level,
        )

    @classmethod
    def load(cls, loc: Path, compression: Optional[Compression] = None) -> Self:
        tbl = pq.read_table(loc / "data.parq")
        obj = cls()
        columns: list[str] = tbl.column_names
        for name in columns:
            obj.__dict__[name] = tbl.column(name).to_numpy()
        return obj

from __future__ import annotations

from dataclasses import dataclass
from math import e
from pathlib import Path
from typing import Optional

import pandas as pd
import serde.pickle
from ream.data_model_helper._container import DataSerdeMixin
from ream.helper import Compression, to_serde_compression


@dataclass
class SinglePandasDataFrame(DataSerdeMixin):
    value: pd.DataFrame

    """A single pandas dataframe"""

    def __init__(self, value: pd.DataFrame):
        self.value = value

    def save(
        self,
        loc: Path,
        compression: Optional[Compression] = None,
        compression_level: Optional[int] = None,
    ) -> None:
        loc.mkdir(exist_ok=True, parents=True)
        serde.pickle.ser(self.value, loc / "data.pkl")

    @classmethod
    def load(
        cls, loc: Path, compression: Optional[Compression] = None
    ) -> SinglePandasDataFrame:
        return cls(serde.pickle.deser(loc / "data.pkl"))

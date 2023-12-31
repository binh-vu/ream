from __future__ import annotations

import functools
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple, TypedDict, TypeVar

import orjson
import serde.json
import serde.pickle
from loguru import logger
from serde.helper import AVAILABLE_COMPRESSIONS, get_filepath
from typing_extensions import TypeGuard

from ream.data_model_helper import DataSerdeMixin
from ream.helper import Compression, to_serde_compression


class RawSlice(TypedDict):
    value: float
    is_percentage: bool
    absolute_value: Optional[int]


E = TypeVar("E")
E_co = TypeVar("E_co", covariant=True)
E2 = TypeVar("E2")


class DatasetDict(Dict[str, E]):
    serde: tuple[Callable, Callable, Optional[str]] = (
        serde.pickle.ser,
        serde.pickle.deser,
        "pkl",
    )

    def __init__(self, name: str, subsets: dict[str, E], provenance: str = ""):
        super().__init__(subsets)
        self.name = name
        self.provenance = provenance

    def into_single_value(self) -> E:
        assert len(self) == 1
        return list(self.values())[0]

    @classmethod
    def molt(cls, obj: DatasetDict[E]):
        """Change the class of the datasetdict without changing the underlying data. This is useful
        because it relies on the class variable `serde` to determine how to serialize and deserialize examples
        """
        return cls(obj.name, dict(obj), obj.provenance)

    def map(self, fn: Callable[[E], E2]) -> DatasetDict[E2]:
        """Transform dataset from DatasetDict[E] to DatasetDict[E2]"""
        out: DatasetDict[E2] = DatasetDict(
            name=self.name, subsets={}, provenance=self.provenance
        )
        for subset, ds in self.items():
            out[subset] = fn(ds)
        return out

    def save(self, dir: Path, compression: Optional[AVAILABLE_COMPRESSIONS] = None):
        (dir / "metadata.json").write_bytes(
            orjson.dumps(
                {
                    "name": self.name,
                    "provenance": self.provenance,
                    "subsets": list(self.keys()),
                }
            )
        )

        fileext = self.serde[2]
        for subset, ds in self.items():
            if subset == "":
                assert "_empty" not in self

            if fileext is not None:
                filename = f"{subset if subset != '' else '_empty'}.{fileext}"
                filepath = get_filepath(dir / filename, compression)
                self.serde[0](ds, filepath)
            else:
                filepath = dir / (subset if subset != "" else "_empty")
                self.serde[0](ds, filepath, compression)

    @classmethod
    def load(cls, dir: Path, compression: Optional[AVAILABLE_COMPRESSIONS] = None):
        metadata = serde.json.deser(dir / "metadata.json")
        name = metadata["name"]
        provenance = metadata["provenance"]
        subsets = metadata["subsets"]

        ds = cls(name, {}, provenance)
        fileext = cls.serde[2]

        for subset in subsets:
            if fileext is not None:
                filename = f"{subset if subset != '' else '_empty'}.{fileext}"
                filepath = get_filepath(dir / filename, compression)
                ds[subset] = cls.serde[1](filepath)
            else:
                filepath = dir / (subset if subset != "" else "_empty")
                ds[subset] = cls.serde[1](filepath, compression)

        return ds


class DatasetList(List[E], DataSerdeMixin):
    serde: tuple[Callable, Callable, Optional[str]] = (
        serde.pickle.ser,
        serde.pickle.deser,
        "pkl",
    )

    def __init__(self, name: str, items: list[E], provenance: str = ""):
        super().__init__(items)
        self.name = name
        self.provenance = provenance

    def map(self, fn: Callable[[E], E2]) -> DatasetList[E2]:
        """Transform dataset from DatasetList[E] to DatasetList[E2]"""
        return DatasetList(self.name, [fn(item) for item in self], self.provenance)

    def save(
        self,
        loc: Path,
        compression: Optional[Compression] = None,
        compression_level: Optional[int] = None,
    ) -> None:
        (loc / "metadata.json").write_bytes(
            orjson.dumps(
                {
                    "name": self.name,
                    "provenance": self.provenance,
                }
            )
        )

        fileext = self.serde[2]
        if fileext is not None:
            filename = f"data.{fileext}"
            filepath = get_filepath(loc / filename, to_serde_compression(compression))
            self.serde[0](list(self), filepath)
        else:
            filename = "data"
            self.serde[0](list(self), loc / "data", to_serde_compression(compression))

    @classmethod
    def load(cls, loc: Path, compression: Optional[Compression] = None):
        metadata = serde.json.deser(loc / "metadata.json")
        name = metadata["name"]
        provenance = metadata["provenance"]

        fileext = cls.serde[2]
        if fileext is not None:
            filename = f"data.{fileext}"
            filepath = get_filepath(loc / filename, to_serde_compression(compression))
            lst = cls.serde[1](filepath)
        else:
            filepath = loc / "data"
            lst = cls.serde[1](filepath, to_serde_compression(compression))

        return cls(name, lst, provenance)


@dataclass
class AbsoluteRangeSelection:
    start: int
    end: int

    def __len__(self):
        return self.end - self.start

    def __str__(self):
        if self.start == 0:
            return f"[:{self.end}]"
        return f"[{self.start}:{self.end}]"

    def select(self, array: list[E]) -> list[E]:
        return array[self.start : self.end]


@dataclass
class PercentageRangeSelection:
    start: int  # value percentage
    end: int

    def to_absolute(self, size: int) -> AbsoluteRangeSelection:
        return AbsoluteRangeSelection(
            start=int(size * self.start / 100), end=int(size * self.end / 100)
        )

    def __len__(self):
        return self.end - self.start

    def __str__(self):
        if self.start == 0 and self.end == 100:
            return ""
        return f"[{self.start}%:{self.end}%]"

    def select(self, array: list[E]) -> list[E]:
        raise Exception(
            "PercentageRangeSelection does not support select. Convert it to AbsoluteRangeSelection first using to_absolute"
        )


@dataclass
class IndexSelection:
    index: list[int]

    def __len__(self):
        return len(self.index)

    def __str__(self):
        return "[" + ",".join([str(i) for i in self.index]) + "]"

    def select(self, array: list[E]) -> list[E]:
        return [array[i] for i in self.index]


@dataclass
class DatasetQuery:
    dataset: str
    subsets: dict[
        str, AbsoluteRangeSelection | PercentageRangeSelection | IndexSelection
    ]
    shuffle: bool  # the shuffle is done before splitting
    seed: Optional[int]
    postprocessing: list[str] = field(default_factory=list)

    @staticmethod
    @functools.lru_cache(maxsize=1024)
    def from_string(query: str) -> DatasetQuery:
        """Query format:
        - <dataset>:(<subset>[<start>:<end>]+)*(:shuffle([<seed>])?)(:<postprocessing>)?
        - <dataset>[<start>:<end>](:shuffle([<seed>])?)(:<postprocessing>)?
        - <dataset>[number(,number)*](:shuffle([<seed>])?)(:<postprocessing>)?
        - <dataset>:<subset>

        Examples:
        - wt250[:100], wt250[train[0:100], dev[0:100], test[0:100]], wt250{shuffle[42], no-unk-col}
        """
        m = re.match(
            r"".join(
                [
                    r"^",
                    r"(?P<ds>[^:\[{]+)",
                    r"(\[(?P<select>[^{]+)\])?",
                    r"({(?P<process>[^}]+)})?",
                    r"$",
                ]
            ),
            query,
        )
        if m is None:
            raise ValueError(f"Invalid dataset query: {query}")

        dataset = m.group("ds")
        select = m.group("select")
        process = m.group("process")

        subsets: dict[
            str, AbsoluteRangeSelection | PercentageRangeSelection | IndexSelection
        ] = {}
        if select is not None:
            for subset in select.split(","):
                subset = subset.strip()
                m = re.match(
                    r"^(?P<sname>[^\[]*)\[?(?P<start>\d+\%?)?:(?P<end>\d+\%?)?\]?$",
                    subset,
                )
                if m is not None:
                    grpstart = m.group("start")
                    grpend = m.group("end")

                    is_percentage = (
                        any(
                            [
                                grp.endswith("%")
                                for grp in [grpstart, grpend]
                                if grp is not None
                            ]
                        )
                        or grpend is None
                    )

                    if grpstart is None:
                        start = 0
                    else:
                        start = (
                            int(grpstart[:-1])
                            if grpstart.endswith("%")
                            else int(grpstart)
                        )

                    if grpend is None:
                        assert (
                            is_percentage
                        ), "do not support lazy initialization to mixed between absolute and percentage selection"
                        end = 100
                    else:
                        end = int(grpend[:-1]) if grpend.endswith("%") else int(grpend)

                    if is_percentage:
                        subsets[m.group("sname")] = PercentageRangeSelection(start, end)
                    else:
                        subsets[m.group("sname")] = AbsoluteRangeSelection(start, end)
                else:
                    m = re.match(
                        r"^(?P<sname>[^\[]*)\[(?P<index>\d+(?:,\d+)*)\]$",
                        subset,
                    )
                    if m is not None:
                        subsets[m.group("sname")] = IndexSelection(
                            [int(x) for x in m.group("index").split(",")]
                        )
                    else:
                        m = re.match(r"^(?P<sname>[a-zA-Z]+)$", subset)
                        assert (
                            m is not None
                        ), f"Invalid subset spec: `{subset}` in `{select}` in `{query}`"
                        subsets[m.group("sname")] = PercentageRangeSelection(0, 100)
        else:
            subsets[""] = PercentageRangeSelection(0, 100)

        shuffle = False
        seed = None
        postprocessing: list[str] = []

        if process is not None:
            for pp in process.split(","):
                pp = pp.strip()
                if pp.startswith("shuffle"):
                    m = re.match(r"shuffle(\((?P<seed>\d+)\))?$", pp)
                    assert m is not None, f"Invalid shuffle spec: `{pp}` in `{query}`"
                    shuffle = True
                    if m.group("seed") is not None:
                        seed = int(m.group("seed"))
                else:
                    postprocessing.append(pp)

        return DatasetQuery(dataset, subsets, shuffle, seed, postprocessing)

    def select(self, array: list[E]) -> DatasetDict[list[E]]:
        n_exs = len(array)

        # gate check for percentage range selection that select all data
        subsets = {
            subset: selection.to_absolute(n_exs)
            if isinstance(selection, PercentageRangeSelection)
            else selection
            for subset, selection in self.subsets.items()
        }
        if all(isinstance(s, PercentageRangeSelection) for s in self.subsets.values()):
            total_percentage = sum(
                len(selection) for selection in self.subsets.values()
            )
            n_selected = sum(len(selection) for selection in subsets.values())
            if total_percentage == 100 and n_selected != n_exs:
                logger.debug(
                    "Total percentage is 100%, but the number of selected examples do not match, adjusting the first subset"
                )
                assert n_selected < n_exs

                for i, selection in enumerate(subsets.values()):
                    assert isinstance(selection, AbsoluteRangeSelection)
                    if i != 0:
                        selection.start += n_exs - n_selected
                    selection.end += n_exs - n_selected

                assert n_exs == sum(len(selection) for selection in subsets.values())

        if self.shuffle:
            array_index = list(range(n_exs))
            random.Random(self.seed).shuffle(array_index)

            output_subsets = {}
            for subset, selection in subsets.items():
                if isinstance(selection, IndexSelection):
                    indices = set(selection.index)
                    output_subsets[subset] = [
                        array[i] for i in array_index if i in indices
                    ]
                else:
                    output_subsets[subset] = [
                        array[idx] for idx in selection.select(array_index)
                    ]
        else:
            output_subsets = {
                subset: selection.select(array) for subset, selection in subsets.items()
            }

        return DatasetDict(
            self.dataset,
            output_subsets,
        )

    def select_list(self, array: list[E]) -> DatasetList[E]:
        assert len(self.subsets) == 1 and "" in self.subsets
        return DatasetList(self.dataset, self.select(array)[""])

    def strip(self) -> DatasetQuery:
        """Remove the subset name from the select. Error when there are multiple subsets."""
        if len(self.subsets) > 1:
            raise ValueError(
                f"Cannot strip subsets from query when there are multiple subsets: {self.subsets}"
            )
        return DatasetQuery(
            self.dataset,
            {"": next(iter(self.subsets.values()))},
            self.shuffle,
            self.seed,
        )

    def subset(self, subset: str) -> DatasetQuery:
        """Select a subset from the query. Error when the subset does not exist."""
        return DatasetQuery(
            self.dataset, {subset: self.subsets[subset]}, self.shuffle, self.seed
        )

    def iter_subset(self) -> Iterator[Tuple[str, DatasetQuery]]:
        """Iterate over the subsets in the query."""
        return ((subset, self.subset(subset)) for subset in self.subsets)

    def get_query(self, subsets: Optional[str | list[str]] = None) -> str:
        """Generate a query string for retrieving the subsets of the dataset."""
        if subsets is None:
            subsets = list(self.subsets.keys())
        elif isinstance(subsets, str):
            subsets = [subsets]
        else:
            assert all(subset in self.subsets for subset in subsets)

        filter = "+".join(
            [f"{subset}{str(self.subsets[subset])}" for subset in subsets]
        )
        if len(subsets) > 1 or "" not in subsets:
            filter = f":{filter}"

        return f"{self.dataset}{filter}{':shuffle' if self.shuffle else ''}{f':{self.seed}' if self.seed is not None else ''}"

    def get_subset_disk_names(self) -> dict[str, str]:
        return {name: "_empty" if name == "" else name for name in self.subsets.keys()}

from __future__ import annotations
import re, functools, orjson
from typing import TypedDict, Dict, Optional, Tuple, List, Generic
from loguru import logger
from dataclasses import dataclass
from ream.actors.interface import E
from pathlib import Path
from serde.helper import get_compression
import serde.pickle

RawSlice = TypedDict("Slice", value=int, is_percentage=bool, absolute_value=int)


class DatasetDict(Dict[str, E]):
    serde = (serde.pickle.ser, serde.pickle.deser, "pkl")

    def __init__(self, name: str, subsets: Dict[str, E], provenance: str = ""):
        super().__init__(subsets)
        self.name = name
        self.provenance = provenance

    @classmethod
    def molt(cls, obj: DatasetDict[E]):
        """Change the class of the datasetdict without changing the underlying data. This is useful
        because it relies on the class variable `serde` to determine how to serialize and deserialize examples
        """
        return cls(obj.name, dict(obj), obj.provenance)

    def save(self, file: Path):
        outdir = file.parent / file.stem
        outdir.mkdir(parents=True, exist_ok=True)

        compression = get_compression(file)
        if compression is not None:
            compression = f".{compression}"
            # save the file as cache helper requires the file to exist
            file.touch()
        else:
            compression = ""
        ext = f".{self.serde[2]}{compression}"

        (outdir / f"metadata.json{compression}").write_bytes(
            orjson.dumps({"name": self.name, "provenance": self.provenance})
        )

        for subset, ds in self.items():
            filename = f"{subset if subset != '' else '_empty'}{ext}"
            dsfile = outdir / filename
            self.serde[0](ds, dsfile)

    @classmethod
    def load(cls, file: Path):
        outdir = file.parent / file.stem

        compression = get_compression(file)
        if compression is not None:
            compression = f".{compression}"
        else:
            compression = ""

        metadata = orjson.loads((outdir / f"metadata.json{compression}").read_bytes())
        name = metadata["name"]
        provenance = metadata["provenance"]
        ds = cls(name, {}, provenance)

        ext = f".{cls.serde[2]}{compression}"

        for dsfile in outdir.iterdir():
            if dsfile.name.endswith(ext):
                subset = dsfile.name[: -len(ext)]
                if subset == "_empty":
                    subset = ""
                ds[subset] = cls.serde[1](dsfile)
        return ds


@dataclass
class DatasetQuery(Generic[E]):
    dataset: str
    subsets: Dict[str, Tuple[RawSlice, RawSlice]]
    shuffle: bool
    seed: Optional[int]

    @staticmethod
    @functools.lru_cache(maxsize=1024)
    def from_string(query: str) -> DatasetQuery:
        """Query format:
        - <dataset>:(<subset>[<start>:<end>]+)*(:shuffle)?(:seed)?
        - <dataset>[<start>:<end>](:shuffle)?(:seed)?
        """
        m = re.match(
            r"^(?P<ds>[^:\[]+):?(?P<query>(?:[^\[]*\[(?:\d+\%?)?:(?:\d+\%?)?\]\+?)*)(?P<shuffle>:shuffle)?(?P<seed>:\d+)?$",
            query,
        )
        if m is None:
            raise ValueError(f"Invalid dataset query: {query}")

        dataset = m.group("ds")
        splitquery = m.group("query")
        shuffle = m.group("shuffle") is not None
        seed = int(m.group("seed")[1:]) if m.group("seed") is not None else None

        subsets = {}
        if splitquery != "":
            for subset in splitquery.split("+"):
                m = re.match(
                    r"^(?P<sname>[^\[]*)\[(?P<start>\d+\%?)?:(?P<end>\d+\%?)\]", subset
                )
                assert (
                    m is not None
                ), f"Invalid subset spec: `{subset}` in `{splitquery}` in `{query}`"
                slices = []
                for name in ["end", "start"]:
                    if name == "start" and m.group(name) is None:
                        slices.append(
                            {
                                "value": 0,
                                "is_percentage": slices[-1]["is_percentage"],
                                "absolute_value": 0,
                            }
                        )
                        continue
                    value = m.group(name)
                    is_percentage = value.endswith("%")
                    if is_percentage:
                        value = int(value[:-1]) / 100
                    else:
                        value = int(value)
                    slices.append(
                        {
                            "value": value,
                            "is_percentage": is_percentage,
                            "absolute_value": value,
                        }
                    )

                assert (
                    len({x["is_percentage"] for x in slices}) == 1
                ), f"Slices must be either percentage or absolute: {slices}"

                start = slices[1]
                end = slices[0]
                subsets[m.group("sname")] = (start, end)
        else:
            subsets: Dict[str, Tuple[RawSlice, RawSlice]] = {
                "": (
                    {"value": 0, "is_percentage": True, "absolute_value": 0},
                    {"value": 1, "is_percentage": True, "absolute_value": 1},
                )
            }
        return DatasetQuery(dataset, subsets, shuffle, seed)

    def select(self, array: List[E]) -> DatasetDict[List[E]]:
        n_exs = len(array)

        if all(start["is_percentage"] for (start, end) in self.subsets.values()):
            # convert percentage to absolute
            for (start, end) in self.subsets.values():
                start["absolute_value"] = int(start["value"] * n_exs)
                end["absolute_value"] = int(end["value"] * n_exs)

            total_percentage = sum(
                [
                    (end["value"] - start["value"]) * 100
                    for start, end in self.subsets.values()
                ]
            )
            n_selected = sum(
                [
                    end["absolute_value"] - start["absolute_value"]
                    for start, end in self.subsets.values()
                ]
            )
            if total_percentage == 100 and n_selected != n_exs:
                logger.debug(
                    "Total percentage is 100%, but the number of selected examples do not match, adjusting the first subset"
                )
                assert n_selected < n_exs
                for i, (start, end) in enumerate(self.subsets.values()):
                    if i != 0:
                        start["absolute_value"] += n_exs - n_selected
                    end["absolute_value"] += n_exs - n_selected
                assert n_exs == sum(
                    [
                        end["absolute_value"] - start["absolute_value"]
                        for start, end in self.subsets.values()
                    ]
                )

        return DatasetDict(
            self.dataset,
            {
                subset: array[start["absolute_value"] : end["absolute_value"]]
                for subset, (start, end) in self.subsets.items()
            },
        )

    def get_query(self, subsets: Optional[str | List[str]]) -> str:
        """Generate a query string for retrieving the subsets of the dataset."""
        if subsets is None:
            subsets = list(self.subsets.keys())
        elif isinstance(subsets, str):
            subsets = [subsets]
        else:
            assert all(subset in self.subsets for subset in subsets)

        filters = []
        for subset in subsets:
            start, end = self.subsets[subset]
            if start["is_percentage"]:
                start = f"{int(start['value'] * 100)}%"
            else:
                start = start["value"]
            if end["is_percentage"]:
                end = f"{int(end['value'] * 100)}%"
            else:
                end = end["value"]
            if start == "0%" and end == "100%":
                filters.append(f"{subset}")
            elif start == 0:
                filters.append(f"{subset}[:{end}]")
            else:
                filters.append(f"{subset}[{start}:{end}]")

        filter = "+".join(filters)
        if len(subsets) > 1 or "" not in subsets:
            filter = f":{filter}"

        return f"{self.dataset}{filter}{':shuffle' if self.shuffle else ''}{f':{self.seed}' if self.seed is not None else ''}"

    def get_subset_disk_names(self) -> Dict[str, str]:
        return {name: "_empty" if name == "" else name for name in self.subsets.keys()}

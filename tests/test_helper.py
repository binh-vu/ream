from dataclasses import dataclass
from typing import List, Literal

from ream.actors.base import BaseActor
from ream.helper import resolve_type_arguments


@dataclass
class CanGenParams:
    # type of query that will be sent to ElasticSearch
    query_type: Literal["exact-match", "fuzzy-match"]


class CandidateGeneration(BaseActor[CanGenParams]):
    VERSION = 100


def test_resolve_type_arguments():
    assert resolve_type_arguments(BaseActor, CandidateGeneration) == (
        List[str],
        CanGenParams,
    )

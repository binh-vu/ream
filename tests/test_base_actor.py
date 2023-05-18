from typing import Literal, List
from ream.prelude import BaseActor
from dataclasses import dataclass


@dataclass
class CanGenParams:
    # type of query that will be sent to ElasticSearch
    query_type: Literal["exact-match", "fuzzy-match"]


class CandidateGeneration(BaseActor[CanGenParams]):
    VERSION = 100

    def run(self, queries: List[str]):
        # generate candidate entities of the given table
        raise NotImplementedError()


def test_get_param_cls():
    # test the get_args method
    x = CandidateGeneration(CanGenParams("exact-match"))
    assert x.get_param_cls() is CanGenParams

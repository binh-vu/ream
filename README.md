# ream ![PyPI](https://img.shields.io/pypi/v/ream2) ![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)

A simple actor architecture for your research project. It helps addressing three problems so that you can focus on your main research:

1. Configuring hyper-parameters of your method
2. Speed-up the feedback cycles via easy & smart caching
3. Running each step in your method independently.

It's more powerful to combine with [`osin`](https://github.com/binh-vu/osin).

## Introduction

Let's say you are developing a method, an algorithm, or a pipeline to solve a problem. In many cases, it can be viewed as a computational graph. So why not structure your code as a computational graph, where each node is a component in your method or a step in your pipeline? It made your code more modular, and easy to release, cache, and evaluate.
To see how we can apply this architecture, let's take a look at a record linkage project (linking entities in a table). A record linkage system typically has the following steps:

1. Generate candidate entities in a table
2. Rank the candidate entities and select the best matches.

So naturally, we will have two actors for two steps: `CandidateGeneration` and `CandidateRanking`:

```python
import pandas as pd
from typing import Literal
from ream.prelude import BaseActor
from dataclasses import dataclass

@dataclass
class CanGenParams:
    # type of query that will be sent to ElasticSearch
    query_type: Literal["exact-match", "fuzzy-match"]

class CandidateGeneration(BaseActor[pd.DataFrame, CanGenParams]):
    VERSION = 100

    def run(self, table: pd.DataFrame):
        # generate candidate entities of the given table
        ...

@dataclass
class CanRankParams:
    # ranking method to use
    rank_method: Literal["pairwise", "columnwise"]

class CandidateRanking(BaseActor[pd.DataFrame, CanRankParams]):
    VERSION = 100

    def __init__(self, params: CanRankParams, cangen_actor: CandidateGeneration):
        super().__init__(params, [cangen_actor])

    def run(self, table: pd.DataFrame):
        # rank candidate entities of the given table
        ...
```

The two actors make the code more modular and closer to releasable quality. To define the linking pipeline, we can use `ActorGraph`:

```python
from ream.prelude import ActorGraph, ActorNode, ActorEdge

g = ActorGraph()
cangen = g.add_node(ActorNode.new(CandidateGeneration))
canrank = g.add_node(ActorNode.new(CandidateRanking))
g.add_edge(BaseEdge(id=-1, source=cangen, target=canrank))
```

If we provide type hints for arguments of actors, as in the examples above, you can automatically construct the graph by given the actor classes.

```python
from ream.prelude import ActorGraph

g = ActorGraph.auto([CandidateGeneration, CandidateRanking])
```

This seems boring and does not offer much, but then you can pick whatever actor and its function you want to call without manually initializing and parsing command line arguments. For example, we want to trigger the `evaluate` method on each actor. The parameters of the actors will be obtained automatically from the command line arguments, thanks to the [`yada`](https://github.com/binh-vu/yada) parser.

```python
if __name__ == "__main__":
    g.run(actor_class="CandidateGeneration", actor_method="evaluate")
```

The `evaluate` method for each actor can be very useful. On the candidate generation actor, it can tell us the upper bound accuracy of our method so we know whether we need to improve the candidate generation or candidate ranking. If a dataset actor is introduced to the computational graph as demonstrated below, its evaluate method can tell us statistics about the dataset.

```python
from ream.prelude import NoParams, BaseActor, DatasetQuery

class DatasetActor(BaseActor[str, NoParams]):
    VERSION = 100

    def run(self, query: str):
        # use a query so we can dynamically select a subset of the dataset for quickly test
        # for example: mnist[:10] -- select first 10 examples
        dsquery = DatasetQuery.from_string(query)

        # load the real dataset
        examples = ...
        return dsquery.select(examples)

    def evaluate(self, query: str):
        dsdict = self.run(query)
        for split, examples in dsdict.items():
            print(f"Dataset: {dsdict.name} - split {split} has {len(examples)} examples")
```

Let's talk about caching. Each actor when running will be uniquely identified by its name, version, and parameters (including the dependent actor parameters), and this is referred to as actor state which you can retrieve from `BaseActor.get_actor_state` function. From this, we can create a unique folder associated with that state that you can use to store your cache data (the folder can be retrieved from the function `BaseActor.get_working_fs`). Whenever the actor's dependency is updated, you will always get a new folder so no worry about managing the cache yourself! To set it up, in the file that defines the actor graph, init the ream workspace as follows:

```python
from ream.prelude import ReamWorkspace, ActorGraph

ReamWorkspace.init("<folder>/<to>/<store>/<cache>")
g = ActorGraph()
...
```

## Installation

```python
pip install ream2  # not ream
```

## Examples

Will be added later.

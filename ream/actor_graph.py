from __future__ import annotations

from dataclasses import dataclass, is_dataclass
from operator import attrgetter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import yada
from graph.interface import BaseEdge, BaseNode
from graph.retworkx.digraph import RetworkXDiGraph
from loguru import logger

from ream.actors.base import BaseActor
from ream.helper import _logger_formatter, get_classpath
from ream.params_helper import DataClassInstance


ActorEdge = BaseEdge


class ActorNode(BaseNode):
    def __init__(
        self, id: int, cls: Type[BaseActor], cls_constructor: Optional[Callable] = None
    ):
        self.id = id
        self.cls = cls
        if cls_constructor is not None:
            self.cls_constructor = cls_constructor
        else:
            self.cls_constructor = lambda params, dep_actors: cls(params, *dep_actors)

        self.clspath = get_classpath(cls)

    @staticmethod
    def new(
        type: Type[BaseActor], cls_constructor: Optional[Callable] = None
    ) -> "ActorNode":
        return ActorNode(id=-1, cls=type, cls_constructor=cls_constructor)

    def has_short_name(self, name: str) -> bool:
        return self.clspath.endswith(name)


class ActorGraph(RetworkXDiGraph[int, ActorNode, ActorEdge]):
    @dataclass
    class ActorConstructor:
        graph: ActorGraph
        main: ActorNode
        params_cls: Any
        node2param: Dict[Type, Tuple[int, int]]
        node2cls: Dict[Type, ActorNode]

        def get_params_parser(self) -> yada.YadaParser:
            return yada.YadaParser(self.params_cls)

        def create_actor(self, params: List[DataClassInstance]) -> BaseActor:
            output = {}
            self._create_actors(self.main, params, output)
            return output[self.main.cls]

        def create_actors(
            self, nodes: List[ActorNode], params: List[DataClassInstance]
        ) -> List[BaseActor]:
            output = {}
            self._create_actors(self.main, params, output)
            return [output[node.cls] for node in nodes]

        def _create_actors(
            self,
            node: ActorNode,
            params: List[DataClassInstance],
            output: Dict[Type, BaseActor],
        ):
            start, end = self.node2param[node.cls]
            if end == start + 1:
                node_params = params[start]
            else:
                node_params = params[start:end]

            inedges = self.graph.in_edges(node.id)
            if len(inedges) == 0:
                try:
                    output[node.cls] = node.cls_constructor(node_params, dep_actors=[])
                except:
                    logger.error("Error when creating actor {}", node.cls)
                    raise
                return

            dep_actors = []
            for inedge in sorted(inedges, key=attrgetter("key")):
                source = self.graph.get_node(inedge.source)
                if source.cls not in output:
                    self._create_actors(source, params, output)
                dep_actors.append(output[source.cls])

            try:
                output[node.cls] = node.cls_constructor(node_params, dep_actors)
            except:
                logger.error("Error when creating actor {}", node.cls)
                raise

    @staticmethod
    def new():
        return ActorGraph(check_cycle=True, multigraph=False)

    def create_actor(
        self, actor_class: Union[str, Type], args: Optional[Sequence[str]] = None
    ):
        """Create an actor from arguments passed through the command lines

        Args:
            actor_class: The class of the actor to run. If there are multiple actors
                in the graph having the same name, it will throw an error.
            args: The arguments to the . If not provided, it will use the arguments from sys.argv
        """
        logger.debug("Determine the actor to run...")
        actor_node = self.get_actor_by_classname(actor_class)
        logger.debug("Initializing argument parser...")
        constructor = self.get_actor_constructor(actor_node)
        parser = constructor.get_params_parser()
        params = parser.parse_args(args)

        logger.debug("Constructing the actor...")
        actor = constructor.create_actor(params)
        return actor

    def run(
        self,
        actor_class: Union[str, Type],
        actor_method: str = "evaluate",
        args: Optional[Sequence[str]] = None,
        run_args: Optional[Sequence[str]] = None,
        log_file: Optional[str] = None,
    ):
        """Run an actor in evaluation mode.

        Args:
            actor_class: The class of the actor to run. If there are multiple actors
                in the graph having the same name, it will throw an error.
            args: The arguments to the params parser. If not provided, it will use the arguments from sys.argv
            eval_args: The arguments to the evaluate method.
            log_file: whether to log the output to a file. By default the exec-actor's working folder will be used as the current directory
                so you can use a relative path to store the log file in the exec-actor's working directory.
        """
        logger.debug("Determine the actor to run...")
        actor_node = self.get_actor_by_classname(actor_class)
        logger.debug("Initializing argument parser...")
        constructor = self.get_actor_constructor(actor_node)
        parser = constructor.get_params_parser()
        params = parser.parse_args(args)

        logger.debug("Constructing the actor...")
        actor = constructor.create_actor(params)
        if log_file is not None:
            (actor.get_working_fs().root / log_file).parent.mkdir(
                parents=True, exist_ok=True
            )
            logger.add(
                str(actor.get_working_fs().root / log_file),
                colorize=True,
                format=_logger_formatter,
            )

        logger.debug("Run {}.{}...", actor.__class__.__qualname__, actor_method)
        getattr(actor, actor_method)(*(run_args or ()))

    def get_actor_constructor(self, actor_node: ActorNode) -> ActorConstructor:
        nodes = {}
        params_cls: List[Any] = []
        node2param: Dict[Type, Tuple[int, int]] = {}
        for node in self.ancestors(actor_node.id) + [actor_node]:
            if node.cls in nodes:
                raise ValueError(
                    f"Cannot generate argument parser because we have more than one instance of {node.cls} in the actor graph."
                )
            nodes[node.cls] = node
            node_param_cls = node.cls.get_param_cls()
            if isinstance(node_param_cls, list):
                node2param[node.cls] = (
                    len(params_cls),
                    len(params_cls) + len(node_param_cls),
                )
                params_cls.extend(node_param_cls)
            else:
                if not is_dataclass(node_param_cls):
                    raise TypeError(
                        f"We cannot generate argument parser automatically because parameter class of {node.cls} is not a dataclass or a list of dataclasses. It is {node_param_cls}"
                    )
                node2param[node.cls] = (len(params_cls), len(params_cls) + 1)
                params_cls.append(node_param_cls)

        return ActorGraph.ActorConstructor(
            graph=self,
            main=actor_node,
            params_cls=params_cls,
            node2param=node2param,
            node2cls=nodes,
        )

    def get_actor_by_classname(self, actor_class: Union[str, Type]) -> ActorNode:
        if isinstance(actor_class, str):
            matched_nodes = [
                node for node in self.iter_nodes() if node.has_short_name(actor_class)
            ]
        else:
            matched_nodes = [
                node for node in self.iter_nodes() if node.cls == actor_class
            ]
        if len(matched_nodes) == 0:
            raise ValueError(f"Cannot find actor with name {actor_class}")
        elif len(matched_nodes) > 1:
            raise ValueError(
                f"Multiple actors with name {actor_class} found: {[x.cls for x in matched_nodes]}"
                "Try to specify a longer name if you are using a short name."
            )
        return matched_nodes[0]

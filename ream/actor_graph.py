from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, is_dataclass
from inspect import Parameter, signature
from operator import attrgetter
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_type_hints,
)

import yada
from graph.interface import BaseEdge, BaseNode
from graph.retworkx.digraph import RetworkXDiGraph
from loguru import logger
from ream.actors.base import BaseActor
from ream.actors.interface import Actor
from ream.helper import _logger_formatter, get_classpath
from ream.params_helper import DataClassInstance, NoParams

ActorEdge = BaseEdge
A = TypeVar("A")


class ActorNode(BaseNode):
    def __init__(
        self,
        id: int,
        cls: Type[BaseActor],
        cls_constructor: Optional[Callable] = None,
        namespace: str = "",
    ):
        self.id = id
        self.cls = cls
        if cls_constructor is not None:
            self.cls_constructor = cls_constructor
        else:
            self.cls_constructor = lambda params, dep_actors: cls(params, *dep_actors)

        self.clspath = get_classpath(cls)
        # namespace is used to generate parser to parse actor node parameters
        self.namespace = namespace

    @staticmethod
    def new(
        type: Type[BaseActor],
        cls_constructor: Optional[Callable] = None,
        namespace: str = "",
    ) -> "ActorNode":
        return ActorNode(
            id=-1, cls=type, cls_constructor=cls_constructor, namespace=namespace
        )

    def has_short_name(self, name: str) -> bool:
        return self.clspath.endswith(name)


class ActorGraph(RetworkXDiGraph[int, ActorNode, ActorEdge]):
    @dataclass
    class ActorConstructor:
        graph: ActorGraph
        main: ActorNode
        params_cls: list[type[DataClassInstance]]
        params_ns: list[str]
        node2param: Dict[Type, Tuple[int, int]]
        node2cls: Dict[Type, ActorNode]

        def get_params_parser(self) -> yada.YadaParser:
            return yada.YadaParser(self.params_cls, namespaces=self.params_ns)

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

    @staticmethod
    def auto(
        actor_cls: Union[Sequence[Type[BaseActor]], Mapping[str, Type[BaseActor]]],
        strict: bool = False,
        namespaces: Optional[Sequence[str]] = None,
        auto_naming: bool = False,
    ) -> ActorGraph:
        """Automatically create an actor graph from list of actor classes.

        For each actor class, its dependent actors are discovered by inspecting arguments of the actor's init function.
        If type hint's of an argument is Actor or subclass of Actor, it must be in the list of given actors, and a dependency edge between them is created.

        Args:
            actor_cls: List of actor classes or a mapping of actor classes. If a mapping is given, the key is used as namespace for the actor parameter parser.

            strict: whether to enforce all parameters of actor's init function must be type hinted.

            namespaces: List of namespaces to be used for actor parameter parser. If None, there is no namespace. Only works if actor_cls is a sequence. The
                reason we need this is for some actors, we want to use a global namespace ""

            auto_naming: whether to automatically generate a namespace for each actor class. Only works if actor_cls is a sequence and namespaces is None.
                The namespace is generated by removing the suffix "Actor" from the actor class name.
        Returns:
            ActorGraph
        """
        g = ActorGraph.new()
        idmap = {}
        if isinstance(actor_cls, Mapping):
            for ns, cls in actor_cls.items():
                if cls in idmap:
                    raise ValueError(
                        f"Auto graph construction cannot handle duplicated actor class. Found one: {cls}"
                    )
                idmap[cls] = g.add_node(ActorNode.new(cls, namespace=ns))
        else:
            for i, cls in enumerate(actor_cls):
                if cls in idmap:
                    raise ValueError(
                        f"Auto graph construction cannot handle duplicated actor class. Found one: {cls}"
                    )

                if namespaces is not None:
                    namespace = namespaces[i]
                elif auto_naming:
                    clsname = cls.__qualname__
                    if clsname.endswith("Actor"):
                        clsname = clsname[: -len("Actor")]
                    namespace = re.sub(r"(?<!^)(?=[A-Z])", "_", clsname).lower()
                else:
                    namespace = ""

                namespace = namespaces[i] if namespaces is not None else ""
                idmap[cls] = g.add_node(ActorNode.new(cls, namespace=namespace))

        for cls in (
            actor_cls if not isinstance(actor_cls, Mapping) else actor_cls.values()
        ):
            i = 0
            argtypes = get_type_hints(cls.__init__)
            for i, name in enumerate(signature(cls.__init__).parameters.keys()):
                if name not in argtypes:
                    if strict and i == 0 and name != "self":
                        raise TypeError(
                            f"Argument {name} of actor {cls} is not type hinted"
                        )
                    continue
                argtype = argtypes[name]
                if issubclass(argtype, (BaseActor, Actor)):
                    if argtype not in idmap:
                        raise ValueError(
                            f"Cannot find actor class {argtype} in the list of given actors: {actor_cls}"
                        )
                    g.add_edge(
                        ActorEdge(
                            id=-1, source=idmap[argtype], target=idmap[cls], key=i
                        )
                    )
                    i += 1

        return g

    def auto_add_actor(
        self,
        cls: Type[BaseActor],
        strict: bool = False,
        namespace: Optional[str] = None,
        auto_naming: bool = False,
    ):
        if namespace is None:
            if auto_naming:
                clsname = cls.__qualname__
                if clsname.endswith("Actor"):
                    clsname = clsname[: -len("Actor")]
                namespace = re.sub(r"(?<!^)(?=[A-Z])", "_", clsname).lower()
            else:
                namespace = ""

        target_actor_id = self.add_node(ActorNode.new(cls, namespace=namespace))
        argtypes = get_type_hints(cls.__init__)
        for i, name in enumerate(signature(cls.__init__).parameters.keys()):
            if name not in argtypes:
                if strict and i == 0 and name != "self":
                    raise TypeError(
                        f"Argument {name} of actor {cls} is not type hinted"
                    )
                continue
            argtype = argtypes[name]
            if issubclass(argtype, (BaseActor, Actor)):
                source_actor_id = self.get_actor_by_classname(argtype).id
                self.add_edge(
                    ActorEdge(
                        id=-1, source=source_actor_id, target=target_actor_id, key=i
                    )
                )
                i += 1

    def create_actor(
        self,
        actor_class: Union[str, type[A]],
        args: Optional[Union[Sequence[str], Sequence[DataClassInstance]]] = None,
        log_file: Optional[str] = None,
    ) -> A:
        """Create an actor from arguments passed through the command lines

        Args:
            actor_class: The class of the actor to run. If there are multiple actors
                in the graph having the same name, it will throw an error.
            args: The arguments to the actor and its dependencies. If not provided, it will use the arguments from sys.argv.
                If it is a sequence of string, then it will be passed to the params parser. If it is a sequence of DataClassInstance,
                then it will be passed directly to the actor's init function.
        """
        logger.debug("Determine the actor to run...")
        actor_node = self.get_actor_by_classname(actor_class)
        logger.debug("Initializing argument parser...")
        constructor = self.get_actor_constructor(actor_node)
        if args is not None and len(args) > 0 and not isinstance(args[0], str):
            # re-order the arguments to match the order of the param_cls.
            # for this to work, we assume that the classes of parameters are unique
            type2arg = {type(arg): arg for arg in args}
            n_no_params = sum(1 for arg in args if isinstance(arg, NoParams))
            if len(type2arg) != len(args) - (n_no_params - 1 if n_no_params > 1 else 0):
                raise ValueError(
                    "Cannot create actor from list of parameter instances because there are duplicated parameter classes (ream.params_helper.NoParams does not count)."
                )
            params = [type2arg[param_cls] for param_cls in constructor.params_cls]
        else:
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
        return actor

    def run(
        self,
        actor_class: Union[str, Type],
        actor_method: str = "evaluate",
        args: Optional[Sequence[str]] = None,
        run_args: Optional[Union[Sequence[Any], Any]] = None,
        log_file: Optional[str] = None,
        allow_unknown_args: bool = False,
    ):
        """Run an actor in evaluation mode.

        Args:
            actor_class: The class of the actor to run. If there are multiple actors
                in the graph having the same name, it will throw an error.
            args: The arguments to the params parser. If not provided, it will use the arguments from sys.argv
            run_args: The arguments to the evaluate method. If it is a sequence, then it will be passed as *args.
            log_file: whether to log the output to a file. By default the exec-actor's working folder will be used as the current directory
                so you can use a relative path to store the log file in the exec-actor's working directory.
            allow_unknown_args: whether to allow unknown arguments to be passed to the params parser.
        """
        logger.debug("Determine the actor to run...")
        actor_node = self.get_actor_by_classname(actor_class)
        logger.debug("Initializing argument parser...")
        constructor = self.get_actor_constructor(actor_node)
        parser = constructor.get_params_parser()
        params = (
            parser.parse_args(args)
            if not allow_unknown_args
            else parser.parse_known_args(args)[0]
        )

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
        method = getattr(actor, actor_method)
        if run_args is None:
            method()
        elif isinstance(run_args, Sequence):
            method(*(run_args or ()))
        else:
            method(run_args)

    def get_actor_constructor(self, actor_node: ActorNode) -> ActorConstructor:
        nodes = {}
        params_cls: List[Any] = []
        params_ns: List[str] = []
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
                params_ns.extend([node.namespace] * len(node_param_cls))
            else:
                if not is_dataclass(node_param_cls):
                    raise TypeError(
                        f"We cannot generate argument parser automatically because parameter class of {node.cls} is not a dataclass or a list of dataclasses. It is {node_param_cls}"
                    )
                node2param[node.cls] = (len(params_cls), len(params_cls) + 1)
                params_cls.append(node_param_cls)
                params_ns.append(node.namespace)

        return ActorGraph.ActorConstructor(
            graph=self,
            main=actor_node,
            params_cls=params_cls,
            params_ns=params_ns,
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

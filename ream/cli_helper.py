import copy
from dataclasses import dataclass, field, fields, make_dataclass, asdict
from inspect import signature
from typing import Any, Optional, get_type_hints

from loguru import logger
from ream.actor_graph import ActorGraph
import yada


@dataclass
class CLI:
    actor: str = field(
        metadata={"help": "Actor to run"},
    )
    run_args: object = field(
        metadata={
            "help": "The arguments to be passed to the actor's method. The value of this argument is a dataclass, which is dynamically generated from the actor's method's signature."
        },
    )
    run: str = field(
        default="evaluate",
        metadata={"help": "The actor's method to be called"},
    )
    logfile: Optional[str] = field(
        default="logs/run_{time}.log",
        metadata={"help": "The log file to be used"},
    )
    allow_unknown_args: bool = field(
        default=False,
        metadata={"help": "Whether to allow unknown arguments"},
    )

    @classmethod
    def main(cls, graph: ActorGraph, sysargs=None):
        """The entry point of the CLI program."""
        logger.debug("Started!")

        CLI = cls.prep_cli(graph, sysargs)
        args, remain_args = yada.Parser1(CLI).parse_known_args(sysargs)

        logger.debug("Finished preparing arguments. Start the graph!")
        graph.run(
            actor_class=args.actor,
            actor_method=args.run,
            run_args=[getattr(args.run_args, f.name) for f in fields(args.run_args)],
            args=remain_args,
            log_file=args.logfile,
            allow_unknown_args=args.allow_unknown_args,
        )
        logger.debug("Finished!")

    @classmethod
    def get_actor(cls, graph: ActorGraph, sysargs=None):
        args, remain_args = yada.Parser1(
            make_dataclass(
                "CLI",
                fields=[
                    (field.name, field.type, field)
                    for field in fields(cls)
                    if field.name != "run_args" and field.name != "run"
                ],
            )
        ).parse_known_args(sysargs)

        return graph.create_actor(
            actor_class=args.actor,
            args=remain_args,
            log_file=args.logfile,
        )

    @classmethod
    def prep_cli(cls, graph: ActorGraph, sysargs: Optional[list[str]] = None):
        args, remain_args = yada.Parser1(
            make_dataclass(
                "PreCLI",
                fields=[
                    (field.name, field.type, field)
                    for field in fields(cls)
                    if field.name != "run_args"
                ],
            )
        ).parse_known_args(sysargs)

        method = getattr(graph.get_actor_by_classname(args.actor).cls, args.run)
        params = list(signature(method).parameters.keys())
        assert (
            params[0] == "self"
        ), "The first parameter of the actor's method must be 'self'"

        type_hints = get_type_hints(method)

        RunArgs = make_dataclass(
            "RunArgs", fields=[(name, type_hints[name]) for name in params[1:]]
        )
        new_fields = []
        for field in fields(cls):
            if field.name == "run_args":
                field = copy.copy(field)
                field.type = RunArgs
            new_fields.append((field.name, field.type, field))

        return make_dataclass("CLI", fields=new_fields)

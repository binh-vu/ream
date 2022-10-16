from __future__ import annotations
from dataclasses import is_dataclass
import functools
import os
from pathlib import Path
from typing import (
    Optional,
    List,
    Type,
    TypeVar,
    Generic,
)
from ream.actor_state import ActorState
from ream.helper import resolve_type_arguments
from ream.params_helper import EnumParams
from ream.fs import FS
from ream.workspace import ReamWorkspace
from loguru import logger
from ream.actors.interface import Actor, E

P = TypeVar("P")


class BaseActor(Generic[E, P], Actor[E]):
    def __init__(
        self,
        params: P,
        dep_actors: Optional[List[BaseActor]] = None,
    ):
        self._working_fs: Optional[FS] = None
        self.dep_actors: List[BaseActor] = dep_actors or []
        self.params = params
        self.logger = logger.bind(cls=self.__class__.__name__)

    def get_actor_state(self) -> ActorState:
        """Get the state of this actor"""
        deps = [actor.get_actor_state() for actor in self.dep_actors]

        if isinstance(self.params, EnumParams):
            deps.append(
                ActorState.create(
                    self.params.get_method_class(), self.params.get_method_params()
                )
            )
            params = self.params.without_method_args()
        else:
            params = self.params

        return ActorState.create(
            self.__class__,
            params,
            dependencies=deps,
        )

    def get_working_fs(self) -> FS:
        """Get a working directory for this actor that can be used to store the results of each example."""
        if self._working_fs is None:
            state = self.get_actor_state()
            cache_dir = ReamWorkspace.get_instance().reserve_working_dir(state)
            self.logger.debug(
                "Using working directory: {}",
                cache_dir,
            )
            self._working_fs = FS(cache_dir)
        return self._working_fs

    @classmethod
    def get_param_cls(cls) -> Type[P]:
        """Get the parameter class of this actor"""
        args = resolve_type_arguments(BaseActor, cls)
        assert len(args) == 2
        assert is_dataclass(args[1])
        return args[1]  # type: ignore

    def get_verbose_level(self) -> int:
        """Get the verbose level of this actor from the environment variable"""
        return int(os.environ.get(self.__class__.__name__.upper() + "_VERBOSE", "0"))

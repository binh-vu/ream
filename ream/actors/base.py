from __future__ import annotations

import os
from dataclasses import is_dataclass
from typing import TYPE_CHECKING, Generic, List, Optional, Protocol, Type, TypeVar

from loguru import logger

from ream.actor_state import ActorState
from ream.actors.interface import Actor
from ream.fs import FS
from ream.helper import resolve_type_arguments
from ream.params_helper import EnumParams
from ream.workspace import ReamWorkspace

if TYPE_CHECKING:
    from loguru import Logger

P = TypeVar("P", covariant=True)


class BaseActorProtocol(Protocol[P]):
    @property
    def params(self) -> P:
        ...

    @property
    def logger(self) -> Logger:
        ...

    @property
    def dep_actors(self) -> list[BaseActorProtocol]:
        ...


class BaseActor(Actor, Generic[P]):
    """A based for parameterized actor that its output is always determisitic given: the input, actor's state, and actor's provenance.

    The actor's provenance is an optional mechanism to provide additional guarantee to always have deterministic output if
    the combination of input and actor's state is not sufficient. For example, training a model with a random seed and the store the model
    for later use, the provenance can be the path to the model. Normally, we advise to avoid using provenance if possible.

    The actor state is determined based on: its parameters and the dependant actors' states.
    """

    def __init__(
        self,
        params: P,
        dep_actors: Optional[List[BaseActor]] = None,
    ):
        self._working_fs: Optional[FS] = None
        self.dep_actors: List[BaseActor] = dep_actors or []
        self.params = params
        self.logger = logger.bind(name=self.__class__.__name__)

    def get_actor_state(self) -> ActorState:
        """Get the state of this actor"""
        deps = [actor.get_actor_state() for actor in self.dep_actors]

        if isinstance(self.params, EnumParams):
            for field in self.params.get_method_fields():
                deps.append(
                    ActorState.create(
                        self.params.get_method_class(field),
                        self.params.get_method_params(field),
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

    def get_provenance(self):
        """Return the provenance of this actor. The provenance is used when there are cases where even if we have the same state, the result can be different.

        For example:
            - training a model with a random seed and store the model for later use
            - use external method that has a version itself and the version can be changed later

        To use this provenance as a key to cache result of a function, use `cache_self_args=CacheArgsHelper.gen_cache_self_args(get_provenance)` in Cache decorator.
        """
        return ""

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
        assert len(args) == 1
        assert is_dataclass(args[0])
        return args[0]  # type: ignore

    def get_verbose_level(self) -> int:
        """Get the verbose level of this actor from the environment variable"""
        return int(os.environ.get(self.__class__.__name__.upper() + "_VERBOSE", "0"))

    def _fmt_prov(self, *prov: str) -> str:
        """Format the provenances"""
        return ";".join(prov)

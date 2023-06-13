from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, List, Optional, TypeVar, cast

from ream.actor_state import ActorState
from ream.actors.base import BaseActor, P
from ream.params_helper import EnumParams


class EnumInterface(ABC):
    @abstractmethod
    def exec(self, *args, **kwargs):
        pass

    def get_provenance(self) -> str:
        return ""


EP = TypeVar("EP", bound=EnumParams)
EI = TypeVar("EI", bound=EnumInterface)  # enum interface


class EnumActor(Generic[EP, EI], BaseActor[EP]):
    def __init__(self, params: EP, dep_actors: Optional[List[BaseActor]] = None):
        super().__init__(params, dep_actors)

        # extract the method
        enum_fields = self.params.get_method_fields()
        if len(enum_fields) != 1:
            raise ValueError(
                f"EnumActor must have exactly one method field, got {len(enum_fields)}"
            )

        enum_field = enum_fields[0]
        cls = self.params.get_method_class(enum_field)
        params = self.params.get_method_params(enum_field)

        # construct the selected method
        self.method = cast(EI, cls(params))

    def get_provenance(self):
        return self.method.get_provenance()

    def exec(self, *args, **kwargs):
        return self.method.exec(*args, **kwargs)

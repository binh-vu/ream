from __future__ import annotations
from typing import Generic, TypeVar
from ream.actor_state import ActorState
from ream.actors.base import BaseActor, P
from ream.params_helper import EnumParams

EP = TypeVar("EP", bound=EnumParams)


class EnumActor(BaseActor[EP]):
    def get_actor_state(self) -> ActorState:
        deps = [actor.get_actor_state() for actor in self.dep_actors]

        enum_fields = self.params.get_method_fields()
        if not len(enum_fields) != 1:
            raise ValueError(
                f"EnumActor must have exactly one method field, got {len(enum_fields)}"
            )

        enum_field = enum_fields[0]
        cls = self.params.get_method_class(enum_field)
        params = self.params.get_method_params(enum_field)
        if params is None:
            raise ValueError(
                f"Parameter for an enum variant {cls} must be provided, got None"
            )
        return ActorState.create(cls, params, dependencies=deps)

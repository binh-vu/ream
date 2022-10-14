from __future__ import annotations
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Type,
    Union,
)
from ream.params_helper import param_as_dict
from ream.helper import get_classpath


@dataclass
class ActorState:
    """Represent a state of an actor, including its class, versions, and parameters"""

    classpath: str
    classversion: str
    params: Union[Any, List[Any], Dict[str, Any]]
    dependencies: List[ActorState]

    @staticmethod
    def create(
        CLS: Type,
        args: Union[Any, List[Any], Dict[str, Any]],
        version: Optional[str] = None,
        dependencies: Optional[List[ActorState]] = None,
    ) -> ActorState:
        """Compute a unique cache id"""
        if version is None:
            assert hasattr(CLS, "VERSION"), "Class must have a VERSION attribute"
            version = getattr(CLS, "VERSION")

        assert isinstance(version, str), "Version must be a string"

        return ActorState(
            classpath=get_classpath(CLS),
            classversion=version,
            params=args,
            dependencies=dependencies or [],
        )

    def get_classname(self) -> str:
        return self.classpath.split(".")[-1]

    def to_dict(self) -> dict:
        """Return the state in dictionary form, mainly used for comparing the state"""
        if isinstance(self.params, list):
            params = [param_as_dict(p) for p in self.params]
        elif isinstance(self.params, dict):
            params = {k: param_as_dict(v) for k, v in self.params.items()}
        else:
            params = param_as_dict(self.params)

        return {
            "classpath": self.classpath,
            "classversion": self.classversion,
            "params": params,
            "dependencies": [d.to_dict() for d in self.dependencies],
        }

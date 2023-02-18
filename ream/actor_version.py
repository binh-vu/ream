from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

import orjson


K = TypeVar("K", str, int)


@dataclass
class ActorVersion(Generic[K]):
    """Represent a version of an actor that must be consistent with the version of other dependencies."""

    value: K
    dependencies: dict[str, K]

    @staticmethod
    def create(value: K, dependencies: list[type]):
        value = value
        deps = {}
        for cls in dependencies:
            assert hasattr(cls, "VERSION"), "Class must have a VERSION attribute"
            assert cls.__name__ not in deps, "Class name must be unique"
            deps[cls.__name__] = getattr(cls, "VERSION")
        return ActorVersion(value, deps)

    def __str__(self):
        return str(self.value)

    def save(self, path: Path):
        """Save the version to a file"""
        path.write_bytes(
            orjson.dumps(
                {
                    "version": self.value,
                    "dependencies": self.dependencies,
                }
            )
        )

    @classmethod
    def load(cls, path: Path):
        obj = orjson.loads(path.read_bytes())
        return ActorVersion(obj["version"], obj["dependencies"])

    def assert_equal(self, version: ActorVersion):
        """Assert that the version is equal to the given version"""
        if self.value != version.value:
            raise ValueError(f"Version is not equal: {self.value} != {version.value}")

        diff_keys = set(self.dependencies.keys()).symmetric_difference(
            version.dependencies.keys()
        )
        if len(diff_keys) > 0:
            raise ValueError(
                f"Found classes that are not presented in both versions: {diff_keys}"
            )

        for name, v in self.dependencies.items():
            if v != version.dependencies[name]:
                raise ValueError(
                    f"Version is not equal for {name}: {v} != {version.dependencies[name]}"
                )

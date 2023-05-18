from __future__ import annotations

import os
from pathlib import Path
from typing import Union

from ream.actor_state import ActorState
from ream.actor_version import ActorVersion
from ream.fs import FS
from slugify import slugify


class ReamWorkspace:
    instance = None

    def __init__(self, workdir: Union[str, Path]):
        self.workdir = Path(workdir)
        self.fs = FS(self.workdir)

    @staticmethod
    def get_instance() -> ReamWorkspace:
        if ReamWorkspace.instance is None:
            raise Exception("ReamWorkspace must be initialized before using")
        return ReamWorkspace.instance

    @staticmethod
    def init(workdir: Union[Path, str]):
        ReamWorkspace.instance = ReamWorkspace(workdir)
        return ReamWorkspace.instance

    def reserve_working_dir(self, state: ActorState) -> Path:
        classversion = slugify(str(state.classversion)).replace("-", "_")
        relpath = os.path.join(state.classpath, classversion)
        key = state.to_dict()
        diskpath = f"{state.get_classname()}/v{classversion}"
        path = self.fs.get(
            relpath=relpath,
            key=key,
            diskpath=diskpath,
            save_key=True,
            subdir=True,
            subdir_incr=True,
        )

        if path.exists():
            return self.ensure_consistent_version(state.classversion, path.get())

        # reserve the working directory for this actor, as we do not perform any
        # action that need to track, we need to mark it as success otherwise, the
        # next time we reserve it, it is going to delete the content of the directory
        with path.reserve_and_track() as realpath:
            return self.ensure_consistent_version(state.classversion, realpath)

    def ensure_consistent_version(
        self, version: Union[str, int, ActorVersion], realpath: Path
    ):
        """Used in reserve_working_dir to ensure the version is consistent"""
        if isinstance(version, ActorVersion):
            version_file = realpath.parent / "version.json"
            if not version_file.exists():
                version.save(version_file)
            else:
                version.assert_equal(ActorVersion.load(version_file))

        return realpath

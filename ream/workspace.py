from __future__ import annotations

import os
from pathlib import Path
from typing import Union

from ream.actor_state import ActorState
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
        relpath = os.path.join(
            state.classpath, slugify(str(state.classversion)).replace("-", "_")
        )
        key = state.to_dict()
        diskpath = f"{state.get_classname()}/v{state.classversion}"
        path = self.fs.get(
            relpath=relpath, key=key, diskpath=diskpath, save_key=True, subdir=True
        )

        if path.exists():
            return path.get()
        return path.reserve()

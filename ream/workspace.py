from __future__ import annotations

import os
from pathlib import Path
from typing import Union, TYPE_CHECKING

import orjson
from slugify import slugify


from ream.actor_version import ActorVersion
from ream.fs import FS
from loguru import logger

if TYPE_CHECKING:
    from ream.actor_state import ActorState


class ReamWorkspace:
    instance = None

    def __init__(self, workdir: Union[str, Path]):
        self.workdir = Path(workdir)
        self.fs = FS(self.workdir)

        # register base paths to abstract away the exact locations of disk paths
        # similar to prefix & namespace
        self.registered_base_paths: dict[str, Path] = {}

    @staticmethod
    def get_instance() -> ReamWorkspace:
        if ReamWorkspace.instance is None:
            raise Exception("ReamWorkspace must be initialized before using")
        return ReamWorkspace.instance

    @staticmethod
    def init(workdir: Union[Path, str]):
        if ReamWorkspace.instance is not None:
            # allow calling re-initialization if the workdir is the same
            assert ReamWorkspace.instance.workdir == Path(workdir)
        else:
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

    def export_working_dir(self, workdir: Path, outfile: Path):
        assert workdir.is_relative_to(self.workdir)
        metadata = orjson.dumps(self.fs.get_record(workdir))
        FS(workdir).export_fs(outfile, metadata=metadata)

    def import_working_dir(self, infile: Path):
        metadata = orjson.loads(FS.read_fs_export_metadata(infile))
        logger.info("Import working dir: {}", metadata['diskpath'])
        FS(self.workdir / metadata["diskpath"]).import_fs(infile)
        self.fs.add_record(metadata)

    def register_base_paths(self, **kwargs: Path):
        for prefix, basepath in kwargs.items():
            self.registered_base_paths[prefix] = Path(basepath)

    def get_rel_path(self, path: Path) -> str:
        for prefix, basepath in self.registered_base_paths.items():
            if path.is_relative_to(basepath):
                return f"{prefix}:{path.relative_to(basepath)}"
        raise ValueError(f"Cannot find the base path of {path}")

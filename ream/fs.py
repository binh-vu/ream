from __future__ import annotations
from contextlib import contextmanager
import os
import sqlite3, enum, shutil
from pathlib import Path
from typing import Generator, Optional, Union
from loguru import logger
from slugify import slugify
from dataclasses import dataclass
from ream.helper import orjson_dumps
from filelock import FileLock


class FS:
    """Provide additional features:
    - different folder/file with the same name but different keys (dictionary)
    - mark when the file is successfully written with `reserve_and_track`
    """

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(exist_ok=True, parents=True)
        self.dbfile = root / "fs.db"

        need_init = not self.dbfile.exists()
        self.db = sqlite3.connect(str(self.dbfile))
        if need_init:
            with self.db:
                self.db.execute(
                    "CREATE TABLE files(path, diskpath, success INT, key BLOB)"
                )

        self.lock: Optional[FileLock] = None

    def get(
        self,
        relpath: str,
        key: Optional[Union[dict, str, bytes]] = None,
        diskpath: Optional[str] = None,
        save_key: bool = False,
        subdir: bool = False,
    ) -> FSPath:
        """Get a path associated with a virtual relpath with key.

        If the relpath is a directory (has extensions) and is already exists but with different key, it will create a new directory
        with the unique number suffix when `subdir` is False, otherwise, it will create a subdirectory
        named by the unique number.
        """
        if key is None:
            ser_key = b""
        elif isinstance(key, str):
            ser_key = key.encode()
        elif isinstance(key, dict):
            ser_key = orjson_dumps(key)
        else:
            assert isinstance(key, bytes)
            ser_key = key

        # clean irregular characters
        if diskpath is None:
            diskpath = relpath
        pdiskpath = Path(diskpath)
        ext = "".join(Path(diskpath).suffixes)
        pdiskpath = Path("").joinpath(
            *[
                slugify(part, lowercase=False).replace("-", "_")
                for part in pdiskpath.parent.parts
            ],
            pdiskpath.name[: len(pdiskpath.name) - len(ext)],
        )
        pdiskpath = pdiskpath.parent / (pdiskpath.name + ext)
        diskpath = str(pdiskpath)

        return FSPath(
            relpath,
            diskpath=diskpath,
            ser_key=ser_key,
            save_key=save_key,
            subdir=subdir,
            fs=self,
        )

    @contextmanager
    def acquire_write_lock(self):
        """Acquire a write lock on the current directory. You should use this before
        any attempt to write to the cache directory to prevent multiple processes
        from writing to the same directory at the same time.
        """
        if self.lock is None:
            self.lock = FileLock(self.root / "_LOCK")

        logger.trace(
            "[Process {}] Acquiring lock on the directory: {}", os.getpid(), self.root
        )
        with self.lock.acquire(timeout=15):
            logger.trace(
                "[Process {}] Acquiring lock on the directory: {}... SUCCESS!",
                os.getpid(),
                self.root,
            )
            yield


class ItemStatus(int, enum.Enum):
    DoNotExist = 0
    Success = 1
    NotSuccess = 2


@dataclass
class FSPath:
    relpath: str
    diskpath: str
    ser_key: bytes
    save_key: bool
    subdir: bool
    fs: FS
    _id: Optional[int] = None
    _status: Optional[ItemStatus] = None
    _realdiskpath: str = ""

    def status(self) -> ItemStatus:
        """"""
        if self._status is not None:
            return self._status
        with self.fs.db:
            record = self.fs.db.execute(
                "SELECT rowid, path, diskpath, success, key FROM files WHERE path = ? AND key = ?",
                (self.relpath, self.ser_key),
            ).fetchone()
            if record is None:
                self._status = ItemStatus.DoNotExist
            elif not (self.fs.root / record[2]).exists():
                # the file is not in disk, but in the database
                # we need to remove it from the database
                self.fs.db.execute("DELETE FROM files WHERE rowid = ?", (record[0],))
                self._status = ItemStatus.DoNotExist
            else:
                self._id = record[0]
                self._status = ItemStatus(record[3])
                self._realdiskpath = record[2]

        return self._status

    def exists(self) -> bool:
        return self.status() == ItemStatus.Success

    def get(self) -> Path:
        """Obtain the real path in disk"""
        if self.status() == ItemStatus.Success:
            return self.fs.root / self._realdiskpath
        raise FileNotFoundError(f"File {self.relpath} does not exist")

    @contextmanager
    def reserve_and_track(self) -> Generator[Path, None, None]:
        """Reserve a path and make sure its status is updated to success if all operations are successful"""
        try:
            yield self.reserve()
            with self.fs.db:
                self.fs.db.execute(
                    "UPDATE files SET success = ? WHERE rowid = ?",
                    (ItemStatus.Success, self._id),
                )
            self._status = ItemStatus.Success
        except Exception:
            self._status = ItemStatus.NotSuccess
            with self.fs.db:
                self.fs.db.execute(
                    "UPDATE files SET success = ? WHERE rowid = ?",
                    (ItemStatus.NotSuccess, self._id),
                )
            raise

    def reserve(self) -> Path:
        """Reserve a real path in disk"""
        status = self.status()
        if status == ItemStatus.Success:
            raise FileExistsError(f"File {self.relpath} already exists")

        if status == ItemStatus.DoNotExist:
            with self.fs.db:
                last_id = self.fs.db.execute("SELECT MAX(rowid) FROM files").fetchone()[
                    0
                ]
                if last_id is None:
                    last_id = 0

                pdiskpath = Path(self.diskpath)
                ext = "".join(pdiskpath.suffixes)
                if self.subdir and ext == "":
                    self._realdiskpath = str(pdiskpath / f"{last_id + 1:03d}")
                else:
                    self._realdiskpath = str(
                        pdiskpath.parent
                        / (
                            pdiskpath.name[: len(pdiskpath.name) - len(ext)]
                            + f"_{last_id + 1:03d}"
                            + ext
                        )
                    )
                cur = self.fs.db.execute(
                    "INSERT INTO files VALUES (?, ?, ?, ?)",
                    (
                        self.relpath,
                        self._realdiskpath,
                        ItemStatus.NotSuccess,
                        self.ser_key,
                    ),
                )
                self._id = cur.lastrowid

                path = self.fs.root / self._realdiskpath
                if ext == "":
                    # create the folder
                    path.mkdir(exist_ok=True, parents=True)
                    if self.save_key:
                        # only save key when it's a folder
                        (path / "_KEY").write_bytes(self.ser_key)
                else:
                    path.parent.mkdir(exist_ok=True, parents=True)

                self._status = ItemStatus.NotSuccess
                return path

        if status == ItemStatus.NotSuccess:
            # remove the file and reserve a new one
            assert self._realdiskpath != ""
            pdiskpath = Path(self.diskpath)
            ext = "".join(pdiskpath.suffixes)

            if ext != "":
                # this is the folder, remove everything else except _KEY
                for f in (self.fs.root / self._realdiskpath).iterdir():
                    if f.name == "_KEY":
                        continue
                    if f.is_dir():
                        shutil.rmtree(f)
                    else:
                        f.unlink()
            return self.fs.root / self._realdiskpath

        raise Exception("Unreachable!")

from __future__ import annotations

import enum
import os
import pickle
import shutil
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Union
from zipfile import ZipFile

import orjson
import serde.pickle
from filelock import FileLock
from loguru import logger
from ream.helper import orjson_dumps
from slugify import slugify


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
                    "CREATE TABLE files(path, diskpath, success INT, key BLOB, UNIQUE(diskpath))"
                )

        self.lock: Optional[FileLock] = None

    def get(
        self,
        relpath: str,
        key: Optional[Union[dict, str, bytes]] = None,
        diskpath: Optional[str] = None,
        save_key: bool = False,
        subdir: bool = False,
        subdir_incr: bool = False,
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
        ext = "".join(
            "." + slugify(suffix[1:]).replace("-", "_")
            for suffix in Path(diskpath).suffixes
        )
        pdiskpath = Path("").joinpath(
            *[
                slugify(part, lowercase=False).replace("-", "_")
                for part in pdiskpath.parent.parts
            ],
            slugify(pdiskpath.name.split(".", 1)[0]).replace("-", "_"),
        )
        pdiskpath = pdiskpath.parent / (pdiskpath.name + ext)
        diskpath = str(pdiskpath)

        return FSPath(
            relpath,
            diskpath=diskpath,
            ser_key=ser_key,
            save_key=save_key,
            subdir=subdir,
            subdir_incr=subdir_incr,
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

    def export_fs(self, outfile: Path, metadata: bytes):
        """Export the following FS"""
        with ZipFile(outfile, "w") as f:
            f.writestr("fs.db", self.export_db())
            f.writestr("_METADATA", metadata)

            for file in self.root.iterdir():
                if file.is_dir():
                    f.mkdir(str(file.relative_to(self.root)))
                    os_walk_it = os.walk(str(file))
                    os_walk_root = next(os_walk_it)[0]  # skip the root
                    assert Path(os_walk_root) == file
                    for dirpath, dirnames, filenames in os.walk(str(file)):
                        dirpath = Path(dirpath)
                        rel_dirpath = dirpath.relative_to(self.root)
                        f.mkdir(str(rel_dirpath))
                        for filename in filenames:
                            if filename == "_LOCK":
                                continue
                            f.writestr(
                                str(rel_dirpath / filename),
                                (dirpath / filename).read_bytes(),
                            )
                    continue

                if file.name == "fs.db" or file.name == "_LOCK":
                    continue

                f.writestr(str(file.relative_to(self.root)), file.read_bytes())

    @staticmethod
    def read_fs_export_metadata(infile: Path):
        with ZipFile(infile, "r") as f:
            return f.read("_METADATA")

    def import_fs(self, infile: Path):
        with ZipFile(infile, "r") as zf:
            for file in zf.infolist():
                fpath = Path(file.filename)
                if (
                    file.filename == "_METADATA"
                    or file.filename == "_LOCK"
                    or file.filename == "fs.db"
                ):
                    continue

                with zf.open(file, mode="r") as f:
                    (self.root / fpath).write_bytes(f.read())

            self.import_db(zf.read("fs.db"))

    def export_db(self):
        return pickle.dumps(
            self.db.execute("SELECT path, diskpath, success, key FROM files").fetchall()
        )

    def import_db(self, data: bytes):
        records = pickle.loads(data)
        with self.db:
            self.db.executemany(
                "INSERT OR REPLACE INTO files (path, diskpath, success, key) VALUES (?, ?, ?, ?)",
                records,
            )

    def get_record(self, disk_path: Path) -> Optional[dict]:
        if disk_path.is_absolute():
            assert disk_path.is_relative_to(self.root)
            disk_path = disk_path.relative_to(self.root)

        lst = self.db.execute(
            "SELECT path, diskpath, success, key FROM files WHERE diskpath = ?",
            (str(disk_path),),
        ).fetchall()

        if len(lst) == 0:
            return None

        assert len(lst) == 1
        return {
            "path": lst[0][0],
            "diskpath": lst[0][1],
            "success": lst[0][2],
            "key": lst[0][3].decode(),
        }

    def add_record(self, record: dict):
        with self.db:
            prev_record = self.get_record(Path(record["diskpath"]))
            if prev_record is not None:
                assert record == prev_record
                return

            self.db.execute(
                "INSERT INTO files (path, diskpath, success, key) VALUES (?, ?, ?, ?)",
                (
                    record["path"],
                    record["diskpath"],
                    record["success"],
                    record["key"].encode(),
                ),
            )


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
    subdir: bool  # whether to create a subdirectory for different keys
    subdir_incr: bool  # whether to name the subdirectory with an incremental number or to use the last row number
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

    def update_status(self, status: ItemStatus):
        """Update the status of the file"""
        if self._id is None:
            raise ValueError("The file does not exist in the database")
        with self.fs.db:
            self.fs.db.execute(
                "UPDATE files SET success = ? WHERE rowid = ?",
                (status, self._id),
            )
        self._status = status

    def exists(self) -> bool:
        return self.status() == ItemStatus.Success

    def get(self) -> Path:
        """Obtain the real path in disk"""
        if self.status() == ItemStatus.Success:
            return self.fs.root / self._realdiskpath
        raise FileNotFoundError(f"File {self.relpath} does not exist")

    def get_or_create(self) -> Path:
        if self.exists():
            return self.get()
        path = self.reserve()
        self.update_status(ItemStatus.Success)
        return path

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
                    if self.subdir_incr:
                        if (self.fs.root / pdiskpath).exists():
                            dirs = [
                                int(d.name)
                                for d in (self.fs.root / pdiskpath).iterdir()
                                if d.is_dir() and d.name.isdigit()
                            ]
                        else:
                            dirs = []
                        if len(dirs) == 0:
                            subdirname = f"{0:03d}"
                        else:
                            subdirname = f"{max(dirs) + 1:03d}"
                        self._realdiskpath = str(pdiskpath / subdirname)
                    else:
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

            if ext == "":
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
        raise Exception("Unreachable!")

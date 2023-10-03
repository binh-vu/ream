from __future__ import annotations

import os
import struct
from io import BufferedReader, BytesIO
from pathlib import Path, PosixPath, WindowsPath
from typing import BinaryIO, Literal, Union

import pyarrow as pa


class BatchFileManager:
    def __init__(self):
        self.open_write_files: dict[str, BinaryIO] = {}
        self.open_read_files: dict[str, BufferedReader] = {}
        self.pend_read_files: dict[str, BinaryIO] = {}
        self.pend_write_files: dict[str, tuple[BytesIO, pa.PythonFile]] = {}

    def create(self, filepath: str):
        if filepath in self.pend_write_files:
            raise Exception(
                "Cannot request the same file twice. This is to prevent data corruption. Finishing writting data to the file and flushing it before requesting it again."
            )
        buf = BytesIO()
        self.pend_write_files[filepath] = (buf, pa.PythonFile(buf))
        if filepath not in self.open_write_files:
            self.open_write_files[filepath] = open(filepath, "wb")

        return self.pend_write_files[filepath][1]

    def read(self, filepath: str) -> BinaryIO:
        if filepath in self.pend_read_files:
            raise Exception(
                "Cannot request the same file twice. This is to prevent data corruption. Finishing reading data to the file and flushing it before requesting it again."
            )

        if filepath not in self.open_read_files:
            self.open_read_files[filepath] = open(filepath, "rb")

        file = self.open_read_files[filepath]
        size = struct.unpack("<I", file.read(4))[0]
        buf = file.read(size)

        self.pend_read_files[filepath] = BytesIO(buf)
        return self.pend_read_files[filepath]

    def __enter__(self):
        assert len(self.open_write_files) == 0 and len(self.open_read_files) == 0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()
        for file in self.open_write_files.values():
            file.close()
        for file in self.open_read_files.values():
            file.close()
        self.open_read_files.clear()
        self.open_write_files.clear()

    def flush(self):
        for filepath, (buf, file) in self.pend_write_files.items():
            file = self.open_write_files[filepath]
            buf = buf.getbuffer()
            file.write(struct.pack("<I", buf.nbytes))
            file.write(buf)

        self.pend_write_files.clear()
        self.pend_read_files.clear()

    def is_all_read_done(self) -> bool:
        return all(len(file.peek(1)) == 0 for file in self.open_read_files.values())


class VirtualDir(Path):
    __slots__ = ("filemanager", "mode")

    _flavour = getattr(WindowsPath if os.name == "nt" else PosixPath, "_flavour")
    filemanager: BatchFileManager
    mode: Literal["read", "write"]

    def __new__(cls, *args, **kwargs):
        object = Path.__new__(cls, *args, **kwargs)
        object.filemanager = kwargs["filemanager"]
        object.mode = kwargs["mode"]
        return object

    def __truediv__(self, key: str) -> Union[pa.PythonFile, VirtualDir]:
        key = str(key)
        if key.find(".") == -1:
            return VirtualDir(
                str(self), key, filemanager=self.filemanager, mode=self.mode
            )

        filepath = str(super().__truediv__(key))
        if self.mode == "write":
            return self.filemanager.create(filepath)
        else:
            return self.filemanager.read(filepath)

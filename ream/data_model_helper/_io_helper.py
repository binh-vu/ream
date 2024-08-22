from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any


@dataclass
class fwrite:
    stream: Any

    def write_int(self, val: int) -> None:
        self.stream.write(struct.pack("<I", val))

    def write_ser_obj(self, data: bytes) -> None:
        self.stream.write(struct.pack("<I", len(data)))
        self.stream.write(data)

    def flush(self):
        self.stream.flush()


@dataclass
class fread:
    data: bytes
    counter: int = 0

    def read(self, size: int) -> bytes:
        data = self.data[self.counter : self.counter + size]
        self.counter += size
        return data

    def read_int(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def read_ser_obj(self) -> bytes:
        size = struct.unpack("<I", self.read(4))[0]
        return self.read(size)

    def is_eof(self) -> bool:
        return self.counter >= len(self.data)

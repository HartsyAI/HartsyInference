"""Reads individual tensors of a (large) torch zip checkpoint over HTTP range requests, without torch.load.

Used to pull only the tensors a test needs out of the 11 GB released controlfoley.pth.
"""
import collections
import io
import pickle
import struct
import urllib.request
import zipfile

import numpy as np


class RangeFile(io.RawIOBase):
    """Seekable read-only file over an HTTP URL (resolves the redirect once, 1 MiB read-ahead cache)."""

    def __init__(self, url: str):
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            self._url = r.geturl()
            self._size = int(r.headers["Content-Length"])
        self._pos = 0
        self._cache_start = 0
        self._cache = b""

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, offset, whence=0):
        self._pos = offset if whence == 0 else self._pos + offset if whence == 1 else self._size + offset
        return self._pos

    def _fetch(self, start: int, length: int) -> bytes:
        req = urllib.request.Request(self._url, headers={"Range": f"bytes={start}-{start + length - 1}"})
        with urllib.request.urlopen(req) as r:
            return r.read()

    def read(self, n=-1):
        if n < 0:
            n = self._size - self._pos
        n = min(n, self._size - self._pos)
        if n <= 0:
            return b""
        end = self._pos + n
        if self._cache_start <= self._pos and end <= self._cache_start + len(self._cache):
            out = self._cache[self._pos - self._cache_start:end - self._cache_start]
        elif n >= 1 << 20:
            out = self._fetch(self._pos, n)
        else:
            self._cache = self._fetch(self._pos, min(1 << 20, self._size - self._pos))
            self._cache_start = self._pos
            out = self._cache[:n]
        self._pos += len(out)
        return out

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


_DTYPES = {"FloatStorage": np.float32, "HalfStorage": np.float16, "LongStorage": np.int64, "IntStorage": np.int32,
           "BoolStorage": np.bool_}


class _Storage:
    def __init__(self, key, dtype):
        self.key, self.dtype = key, dtype


class _LazyTensor:
    def __init__(self, storage, offset, size, stride):
        self.storage, self.offset, self.size, self.stride = storage, offset, tuple(size), tuple(stride)


def _rebuild_tensor_v2(storage, offset, size, stride, *rest):
    return _LazyTensor(storage, offset, size, stride)


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if module == "torch" and name in _DTYPES:
            return name
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")

    def persistent_load(self, pid):
        _, storage_type, key, _loc, _n = pid
        return _Storage(key, _DTYPES[storage_type])


class RemotePth:
    """Lazy view of a torch zip checkpoint, local path or http(s) URL."""

    def __init__(self, source: str):
        self._file = RangeFile(source) if source.startswith("http") else open(source, "rb")
        self._zip = zipfile.ZipFile(self._file)
        names = self._zip.namelist()
        self._prefix = names[0].split("/")[0]
        with self._zip.open(f"{self._prefix}/data.pkl") as f:
            self.index = _Unpickler(io.BytesIO(f.read())).load()

    def keys(self):
        return list(self.index.keys())

    def shape(self, key):
        return self.index[key].size

    def read(self, key) -> np.ndarray:
        t = self.index[key]
        count = 1
        for s in t.size:
            count *= s
        info = self._zip.getinfo(f"{self._prefix}/data/{t.storage.key}")
        assert info.compress_type == zipfile.ZIP_STORED
        item = np.dtype(t.storage.dtype).itemsize
        self._file.seek(info.header_offset)
        header = self._file.read(30)
        name_len, extra_len = struct.unpack("<HH", header[26:30])
        start = info.header_offset + 30 + name_len + extra_len + t.offset * item
        self._file.seek(start)
        raw = self._file.read(count * item)
        return np.frombuffer(raw, dtype=t.storage.dtype).reshape(t.size).copy()

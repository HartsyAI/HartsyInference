"""Streams apple/DFN5B-CLIP-ViT-H-14-384's open_clip_pytorch_model.bin over HTTP range requests (no local copy of the
3.9 GB fp32 pickle) and writes an fp16 safetensors with the open_clip key layout.

Usage: python clip_convert_remote.py <output.safetensors> [url]   (needs torch, safetensors, requests)
"""
import sys

import requests
import torch
from safetensors.torch import save_file

URL = "https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-378/resolve/main/open_clip_pytorch_model.bin"
BLOCK = 16 << 20


class RangeFile:
    """Read-only seekable file backed by HTTP range requests with a one-block cache."""

    def __init__(self, url: str):
        self._session = requests.Session()
        head = self._session.head(url, allow_redirects=True)
        head.raise_for_status()
        self._url = head.url
        self._size = int(head.headers["content-length"])
        self._pos = 0
        self._start = 0
        self._block = b""

    def _fetch(self, start: int, end: int) -> bytes:
        r = None
        for _ in range(5):
            r = self._session.get(self._url, headers={"Range": f"bytes={start}-{end}"})
            if r.status_code in (200, 206) and len(r.content) == end - start + 1:
                return r.content
        raise IOError(f"range {start}-{end} failed: {r.status_code if r is not None else 'n/a'}")

    def seek(self, offset: int, whence: int = 0) -> int:
        self._pos = offset if whence == 0 else self._pos + offset if whence == 1 else self._size + offset
        return self._pos

    def tell(self) -> int:
        return self._pos

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def read(self, n: int = -1) -> bytes:
        if n < 0 or self._pos + n > self._size:
            n = self._size - self._pos
        if n == 0:
            return b""
        end = self._pos + n
        if self._start <= self._pos and end <= self._start + len(self._block):
            out = self._block[self._pos - self._start:end - self._start]
        elif n >= BLOCK:
            out = self._fetch(self._pos, end - 1)
        else:
            self._start = self._pos
            self._block = self._fetch(self._pos, min(self._size, self._pos + BLOCK) - 1)
            out = self._block[:n]
        self._pos = end
        return out

    def readinto(self, buf) -> int:
        data = self.read(len(buf))
        buf[:len(data)] = data
        return len(data)


out = sys.argv[1]
state = torch.load(RangeFile(sys.argv[2] if len(sys.argv) > 2 else URL), map_location="cpu", weights_only=True)
if "state_dict" in state:
    state = state["state_dict"]
half = {}
for key in list(state.keys()):
    half[key] = state.pop(key).to(torch.float16).contiguous()
print(len(half), "tensors", sum(v.numel() for v in half.values()) / 1e6, "M params")
save_file(half, out)

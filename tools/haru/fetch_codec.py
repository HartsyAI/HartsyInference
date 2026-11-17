"""Range-fetch only the codec.* tensors of Haru-4B @ the pinned revision into a standalone safetensors file."""
import json
import struct
import sys
import urllib.request
from pathlib import Path

REV = "8d6e83bcd3701a76fc27a764965665da1df7da78"
URL = f"https://huggingface.co/higurashi-ai/Haru-4B/resolve/{REV}/model.safetensors"
SIZE = 10540113510
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)


def get(start, end):
    req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    assert len(data) == end - start + 1, (len(data), end - start + 1)
    return data


n = struct.unpack("<Q", get(0, 7))[0]
header = json.loads(get(8, 8 + n - 1))
base = 8 + n
meta = header.pop("__metadata__", None)
codec = {k: v for k, v in header.items() if k.startswith("codec.")}
lo = min(v["data_offsets"][0] for v in codec.values())
hi = max(v["data_offsets"][1] for v in codec.values())
total = sum(v["data_offsets"][1] - v["data_offsets"][0] for v in codec.values())
print(f"codec tensors {len(codec)}, span {lo}..{hi} ({hi - lo} B), tensor bytes {total}, contiguous={hi - lo == total}")
assert hi - lo == total, "codec tensors are not one contiguous range"

dst = out / "haru_codec_only.safetensors"
new_header = {}
for k in sorted(codec):
    s, e = codec[k]["data_offsets"]
    new_header[k] = {"dtype": codec[k]["dtype"], "shape": codec[k]["shape"], "data_offsets": [s - lo, e - lo]}
hb = json.dumps(new_header, separators=(",", ":")).encode()
hb += b" " * ((8 - len(hb) % 8) % 8)
with open(dst, "wb") as f:
    f.write(struct.pack("<Q", len(hb)))
    f.write(hb)
    pos = base + lo
    end = base + hi
    chunk = 64 * 1024 * 1024
    while pos < end:
        stop = min(pos + chunk, end) - 1
        f.write(get(pos, stop))
        pos = stop + 1
        print(f"  {pos - (base + lo)} / {hi - lo} bytes", flush=True)
print("wrote", dst, dst.stat().st_size)

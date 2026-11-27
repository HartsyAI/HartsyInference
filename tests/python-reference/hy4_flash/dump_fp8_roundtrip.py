"""Cases for the Hy4 Flash Engram fp8 round trip: the reference function (_fp8_round_trip of transformers PR 50903 at eb1d5db) applied to fixed vectors, with torch.

Run in the hartsy-ref:cpu image (needs torch only):  python dump_fp8_roundtrip.py <output.json>
"""

import json
import sys

import numpy as np
import torch


def fp8_round_trip(x):
    info = torch.finfo(torch.float8_e4m3fn)
    scale = x.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / info.max
    quantized = (x / scale).clamp(-info.max, info.max).to(torch.float8_e4m3fn).to(x.dtype) * scale
    return x + (quantized - x).detach()


def main():
    rng = np.random.default_rng(3)
    vectors = []
    for scale in (1.0, 0.003, 40.0, 1e-5):
        vectors.append((rng.standard_normal(96) * scale).astype(np.float32))
    vectors.append(np.zeros(96, dtype=np.float32))
    x = torch.from_numpy(np.stack(vectors))
    y = fp8_round_trip(x)
    # The reference returns x + (q - x), which equals q up to one float rounding; record both the returned value and q.
    info = torch.finfo(torch.float8_e4m3fn)
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / info.max
    q = (x / scale).clamp(-info.max, info.max).to(torch.float8_e4m3fn).to(x.dtype) * scale
    json.dump({"inputs": x.tolist(), "returned": y.tolist(), "quantized": q.tolist()}, open(sys.argv[1], "w"), separators=(",", ":"))
    print("wrote", sys.argv[1], "max |returned - quantized| =", float((y - q).abs().max()))


if __name__ == "__main__":
    main()

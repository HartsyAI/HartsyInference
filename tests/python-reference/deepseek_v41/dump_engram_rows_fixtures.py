"""Dump Engram table-row decode cases for the three storage layouts (official FP8+E8M0, GGUF row264, MLX affine 4-bit).

Two kinds of case go into fixtures/engram_row_decode.json:
  * real: a handful of rows range-fetched (a few hundred bytes each, no bulk download) from the pinned public repos, with
    the byte offsets used, the raw bytes and the python dequantization (bf16 bits);
  * synthetic: small seeded tables laid out exactly like the real shard tensors (row-major fp8 [rows,256] + e8m0 [rows,8];
    GGUF I8 [rows,264]; MLX U32 [rows,32] + F32 scales/biases [rows,4]) with python dequantization of every row.
Dequantization is the python reference the C# layouts are compared against; it is not upstream code (upstream dequantizes
inside torch kernels). fp8: float(e4m3) * 2^(e8m0-127) per 32-column block in float32; mlx: q*scale+bias per 64-column
group in float32; both rounded to bf16 with torch's round-to-nearest-even cast.

Usage: python dump_engram_rows_fixtures.py [--out fixtures/engram_row_decode.json] [--no-real]
"""
import argparse
import base64
import json
import os
import urllib.request

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OFFICIAL_REV = "dba1be0a40aa45a94ad051997016db3960a90277"
GGUF_REV = "dd8a266f7145edc19e2334b46e19b6821f221dc7"
MLX_REV = "100694c1c65d34e331f4f0c2841212f0e849a0ea"
OFFICIAL = f"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/{OFFICIAL_REV}/"
GGUF = f"https://huggingface.co/antirez/deepseek-v4.1-flash-gguf/resolve/{GGUF_REV}/DeepSeek-V4.1-Flash-Q2.gguf"
MLX = f"https://huggingface.co/mlx-community/DeepSeek-V4.1-Flash-MLX-4bit/resolve/{MLX_REV}/"
DIM = 256

# Byte locations of the embed tables in the pinned files, read once from the safetensors / GGUF headers.
# official: absolute file offset of the F8_E4M3 [rows,256] tensor and the F8_E8M0 [rows,8] tensor (8 + header_len + data_offsets[0]).
LAYOUT = {
    "official": {
        "1": {"file": "model-00047-of-00048.safetensors", "rows": 384006168, "embedOffset": 664, "scaleOffset": 664 + 98305579008},
        "14": {"file": "model-00048-of-00048.safetensors", "rows": 384016682, "embedOffset": 672, "scaleOffset": 672 + 98308270592},
    },
    # gguf: I8 ne=[264, rows], absolute offset = aligned data start + tensor offset
    "gguf": {"1": {"rows": 384006168, "offset": 162955640832}, "14": {"rows": 384016682, "offset": 264333279232}},
    "mlx": {
        "1": {"rows": 384006168,
              "weight": {"file": "model-00074-of-00048.safetensors", "offset": 132205937},
              "scales": {"file": "model-00075-of-00048.safetensors", "offset": 140},
              "biases": {"file": "model-00076-of-00048.safetensors", "offset": 140}},
        "14": {"rows": 384016682,
               "weight": {"file": "model-00004-of-00048.safetensors", "offset": 664},
               "scales": {"file": "model-00005-of-00048.safetensors", "offset": 141},
               "biases": {"file": "model-00006-of-00048.safetensors", "offset": 141}},
    },
}


def fetch(url: str, start: int, length: int) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + length - 1}"})
    with urllib.request.urlopen(request) as response:
        data = response.read()
    assert len(data) == length, (url, start, length, len(data))
    return data


def e8m0_scale(b: np.ndarray) -> np.ndarray:
    """2^(b-127) as float32; 255 is NaN. Built from the bit pattern so b=0 gives the float32 subnormal 2^-127."""
    bits = np.where(b == 0, 0x00400000, b.astype(np.uint32) << 23).astype(np.uint32)
    out = bits.view(np.float32).copy()
    out[b == 255] = np.nan
    return out


def bf16_bits(x: np.ndarray) -> np.ndarray:
    return torch.from_numpy(x.astype(np.float32)).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)


def dequant_fp8(fp8: np.ndarray, e8m0: np.ndarray) -> np.ndarray:
    """fp8 [rows,256] uint8, e8m0 [rows,8] uint8 -> bf16 bits [rows,256]."""
    values = torch.from_numpy(fp8.copy()).view(torch.float8_e4m3fn).to(torch.float32).numpy()
    scale = np.repeat(e8m0_scale(e8m0), 32, axis=1)
    with np.errstate(over="ignore"):
        return bf16_bits(values * scale)


def dequant_mlx(words: np.ndarray, scales: np.ndarray, biases: np.ndarray) -> np.ndarray:
    """words uint32 [rows,32] (8 nibbles each, low first), scales/biases float32 [rows,4] -> bf16 bits [rows,256]."""
    nibbles = np.stack([(words >> (4 * j)) & 0xF for j in range(8)], axis=-1).reshape(words.shape[0], DIM).astype(np.float32)
    s = np.repeat(scales, 64, axis=1).astype(np.float32)
    b = np.repeat(biases, 64, axis=1).astype(np.float32)
    return bf16_bits(nibbles * s + b)


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def synthetic(rng: np.random.Generator, rows: int) -> dict:
    fp8 = rng.integers(0, 256, size=(rows, DIM), dtype=np.uint8)
    fp8[(fp8 & 0x7F) == 0x7F] = 0x38  # no NaN in the bulk rows; NaN and extremes live in the edge rows
    e8m0 = rng.integers(110, 141, size=(rows, 8), dtype=np.uint8)
    words = rng.integers(0, 2**32, size=(rows, 32), dtype=np.uint32)
    scales = rng.uniform(1e-3, 0.1, size=(rows, 4)).astype(np.float32)
    biases = rng.uniform(-1.0, 1.0, size=(rows, 4)).astype(np.float32)
    official = dequant_fp8(fp8, e8m0)
    return {
        "rows": rows,
        "official": {"fp8": b64(fp8), "e8m0": b64(e8m0), "bf16": b64(official)},
        "gguf": {"row264": b64(np.concatenate([fp8, e8m0], axis=1)), "bf16": b64(official)},
        "mlx": {"weight": b64(words.astype("<u4")), "scales": b64(scales.astype("<f4")),
                "biases": b64(biases.astype("<f4")), "bf16": b64(dequant_mlx(words, scales, biases))},
    }


def edge_rows() -> dict:
    """Fixed patterns: zeros, +-0, e4m3 max/min normal/subnormal/NaN, e8m0 0/1/127/254/255 (NaN), mlx nibble corners."""
    rows = []
    patterns = [
        ([0x00] * DIM, [127] * 8),
        ([0x80] * DIM, [127] * 8),
        ([0x7E] * DIM, [127] * 8),   # +448
        ([0xFE] * DIM, [127] * 8),   # -448
        ([0x01] * DIM, [127] * 8),   # smallest subnormal
        ([0x08] * DIM, [127] * 8),   # smallest normal
        ([0x38] * DIM, [0, 1, 2, 3, 252, 253, 254, 127]),
        ([0x7F] * DIM, [127] * 8),   # NaN
        ([0x38] * DIM, [255] * 8),   # NaN scale
        ([0x7E] * DIM, [254] * 8),   # overflows bf16 -> inf
        ([0x01] * DIM, [0] * 8),     # underflows toward zero
        ([(i * 7 + 3) & 0xFF if ((i * 7 + 3) & 0x7F) != 0x7F else 0x38 for i in range(DIM)], [120, 121, 122, 123, 124, 125, 126, 127]),
    ]
    fp8 = np.array([p[0] for p in patterns], dtype=np.uint8)
    e8 = np.array([p[1] for p in patterns], dtype=np.uint8)
    n = len(patterns)
    words = np.zeros((n, 32), dtype=np.uint32)
    words[0, :] = 0
    words[1, :] = 0xFFFFFFFF
    words[2, :] = 0x76543210
    words[3, :] = 0xFEDCBA98
    words[4, :] = 0x0F0F0F0F
    words[5:, :] = np.arange(32, dtype=np.uint32)[None, :] * 0x01010101 + np.arange(5, n)[:, None].astype(np.uint32)
    scales = np.array([[0.5, 0.25, 1.0, 0.0078125]] * n, dtype=np.float32)
    biases = np.array([[0.0, -1.0, 0.125, 2.0]] * n, dtype=np.float32)
    official = dequant_fp8(fp8, e8)
    return {
        "rows": n,
        "official": {"fp8": b64(fp8), "e8m0": b64(e8), "bf16": b64(official)},
        "gguf": {"row264": b64(np.concatenate([fp8, e8], axis=1)), "bf16": b64(official)},
        "mlx": {"weight": b64(words.astype("<u4")), "scales": b64(scales), "biases": b64(biases),
                "bf16": b64(dequant_mlx(words, scales, biases))},
    }


def real_cases() -> dict:
    tinfo = LAYOUT
    out = {"official": [], "gguf": [], "mlx": []}
    # official shards 47 (layer 1) / 48 (layer 14): F8_E4M3 [rows,256] at data-relative 0, F8_E8M0 [rows,8] right after
    for layer, shard in tinfo["official"].items():
        rows = shard["rows"]
        picks = [0, 1, 2, 3, 4095, 1_000_003, rows // 2, rows - 2, rows - 1] if layer == "1" else [0, 5, rows - 1]
        for r in picks:
            fp8 = fetch(OFFICIAL + shard["file"], shard["embedOffset"] + r * DIM, DIM)
            e8 = fetch(OFFICIAL + shard["file"], shard["scaleOffset"] + r * 8, 8)
            bf = dequant_fp8(np.frombuffer(fp8, np.uint8)[None], np.frombuffer(e8, np.uint8)[None])[0]
            out["official"].append({"layer": int(layer), "row": r, "fp8": b64(np.frombuffer(fp8, np.uint8)),
                                    "e8m0": b64(np.frombuffer(e8, np.uint8)), "bf16": b64(bf)})
    for layer, t in tinfo["gguf"].items():
        rows = t["rows"]
        for r in ([0, 1, 2, 3, 4095, 1_000_003, rows - 1] if layer == "1" else [0, rows - 1]):
            raw = fetch(GGUF, t["offset"] + r * 264, 264)
            arr = np.frombuffer(raw, np.uint8)
            bf = dequant_fp8(arr[None, :DIM], arr[None, DIM:])[0]
            out["gguf"].append({"layer": int(layer), "row": r, "row264": b64(arr), "bf16": b64(bf)})
    for layer, t in tinfo["mlx"].items():
        rows = t["rows"]
        for r in ([0, 1, 2, 3, 4095, 1_000_003, rows - 1] if layer == "1" else [0, rows - 1]):
            w = np.frombuffer(fetch(MLX + t["weight"]["file"], t["weight"]["offset"] + r * 128, 128), "<u4")
            s = np.frombuffer(fetch(MLX + t["scales"]["file"], t["scales"]["offset"] + r * 16, 16), "<f4")
            bi = np.frombuffer(fetch(MLX + t["biases"]["file"], t["biases"]["offset"] + r * 16, 16), "<f4")
            bf = dequant_mlx(w[None], s[None], bi[None])[0]
            out["mlx"].append({"layer": int(layer), "row": r, "weight": b64(w), "scales": b64(s), "biases": b64(bi),
                               "bf16": b64(bf)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "fixtures", "engram_row_decode.json"))
    ap.add_argument("--no-real", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(20260930)
    payload = {
        "dim": DIM,
        "dequant": "fp8: float32(e4m3)*2^(e8m0-127) per 32 cols; mlx: nibble*scale+bias per 64 cols (float32); bf16 = RNE",
        "revisions": {"official": OFFICIAL_REV, "gguf": GGUF_REV, "mlx": MLX_REV},
        "edge": edge_rows(),
        "synthetic": synthetic(rng, 96),
    }
    if not a.no_real:
        payload["real"] = real_cases()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
        f.write("\n")
    print(os.path.getsize(a.out), "bytes")


if __name__ == "__main__":
    main()

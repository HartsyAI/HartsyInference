#!/usr/bin/env python3
"""First rows of MiniMax-M3.1-Flash's two weight formats, with independently dequantized F32 values.

Fetches (HTTP range requests, no full download) from MiniMaxAI/MiniMax-M3.1-Flash:
  * an MXFP8 attention weight: language_model.model.layers.3.self_attn.q_proj.{weight, weight_scale_inv}
    (F8_E4M3 [out, in] plus U8 E8M0 scales [out, in/32]; value = e4m3 * 2^(scale - 127));
  * a compressed-tensors NVFP4 expert weight: language_model.model.layers.3.block_sparse_moe.experts.0.w1.{weight_packed,
    weight_scale, weight_global_scale, input_global_scale} (value = e2m1 * e4m3(scale) / weight_global_scale).
Writes tests/HartsyInference.ModelAssets.Tests/Fixtures/MiniMaxM3/quant_rows.safetensors (the stored rows under the checkpoint's names)
and quant_rows_expected.safetensors (F32), using plain Python float math, not engine code.

Run: python -I tests/python-reference/dump_minimax_m3_quant_rows.py  (needs torch + safetensors for the F8 dtype, and network)
"""
import json
import os
import struct
import urllib.request

import torch
from safetensors.torch import save_file

REPO = "https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/resolve/main/"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "HartsyInference.ModelAssets.Tests", "Fixtures", "MiniMaxM3")
ROWS = 3
LAYER = "language_model.model.layers.3."
ATTN = LAYER + "self_attn.q_proj"
EXPERT = LAYER + "block_sparse_moe.experts.0.w1"


def fetch(path, start, stop):
    request = urllib.request.Request(REPO + path, headers={"Range": f"bytes={start}-{stop - 1}"})
    return urllib.request.urlopen(request, timeout=120).read()


weight_map = json.loads(urllib.request.urlopen(REPO + "model.safetensors.index.json", timeout=120).read())["weight_map"]
headers = {}


def rows_of(key, rows):
    shard = weight_map[key]
    if shard not in headers:
        length = struct.unpack("<Q", fetch(shard, 0, 8))[0]
        headers[shard] = (json.loads(fetch(shard, 8, 8 + length)), 8 + length)
    header, base = headers[shard]
    meta = header[key]
    begin = meta["data_offsets"][0]
    shape = meta["shape"]
    row_bytes = (meta["data_offsets"][1] - begin) // shape[0] if shape else meta["data_offsets"][1] - begin
    count = rows if shape else 1
    return meta["dtype"], shape, fetch(shard, base + begin, base + begin + count * row_bytes)


def e4m3(byte):
    sign = -1.0 if byte & 0x80 else 1.0
    exponent, mantissa = (byte >> 3) & 15, byte & 7
    return sign * (mantissa / 8 * 2.0 ** -6 if exponent == 0 else (1 + mantissa / 8) * 2.0 ** (exponent - 7))


E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1 += [-v for v in E2M1]
stored, expected = {}, {}

_, shape, raw = rows_of(ATTN + ".weight", ROWS)
_, scale_shape, scale_raw = rows_of(ATTN + ".weight_scale_inv", ROWS)
stored[ATTN + ".weight"] = torch.frombuffer(bytearray(raw), dtype=torch.float8_e4m3fn).reshape(ROWS, shape[1])
stored[ATTN + ".weight_scale_inv"] = torch.frombuffer(bytearray(scale_raw), dtype=torch.uint8).reshape(ROWS, scale_shape[1])
cols = shape[1]
expected["attn"] = torch.tensor([[e4m3(raw[r * cols + c]) * 2.0 ** (scale_raw[r * scale_shape[1] + c // 32] - 127) for c in range(cols)]
                                 for r in range(ROWS)], dtype=torch.float32)

_, shape, packed = rows_of(EXPERT + ".weight_packed", ROWS)
_, scale_shape, scale_raw = rows_of(EXPERT + ".weight_scale", ROWS)
_, _, global_raw = rows_of(EXPERT + ".weight_global_scale", 1)
_, _, input_raw = rows_of(EXPERT + ".input_global_scale", 1)
global_scale = struct.unpack("<f", global_raw)[0]
stored[EXPERT + ".weight_packed"] = torch.frombuffer(bytearray(packed), dtype=torch.uint8).reshape(ROWS, shape[1])
stored[EXPERT + ".weight_scale"] = torch.frombuffer(bytearray(scale_raw), dtype=torch.float8_e4m3fn).reshape(ROWS, scale_shape[1])
stored[EXPERT + ".weight_global_scale"] = torch.tensor(global_scale, dtype=torch.float32)
stored[EXPERT + ".input_global_scale"] = torch.tensor(struct.unpack("<f", input_raw)[0], dtype=torch.float32)
width = shape[1]
values = []
for r in range(ROWS):
    for c in range(width * 2):
        byte = packed[r * width + c // 2]
        nibble = byte & 15 if c % 2 == 0 else byte >> 4
        values.append(E2M1[nibble] * e4m3(scale_raw[r * scale_shape[1] + c // 16]) / global_scale)
expected["expert"] = torch.tensor(values, dtype=torch.float32).reshape(ROWS, width * 2)

os.makedirs(OUT, exist_ok=True)
save_file(stored, os.path.join(OUT, "quant_rows.safetensors"))
save_file(expected, os.path.join(OUT, "quant_rows_expected.safetensors"))
print("wrote", OUT, "global scale", global_scale, "attn std", expected["attn"].std().item(), "expert std", expected["expert"].std().item())

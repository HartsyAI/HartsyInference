#!/usr/bin/env python3
"""Bit-exact ground truth for every GGUF codec, from llama.cpp's own C code.

ggml's quantizer (ggml_quantize_chunk, with a flat importance matrix where a type needs one) turns rows of seeded normal
floats, one row scaled up a thousandfold, into real blocks of each type; ggml's dequantizer (type_traits.to_float) gives
the floats those blocks decode to. GgmlCodecGroundTruthTests runs the engine's codec on the same blocks and requires the
same floats bit for bit. Writes tests/HartsyInference.ModelAssets.Tests/Fixtures/GgmlCodecs/ggml_codec_blocks.safetensors
with <type>.blocks (U8) and <type>.values (F32) per type.

Run: python -I tests/python-reference/dump_ggml_codec_blocks.py --ggml-lib <llama.cpp>/build/bin/libggml-base.so
"""
import argparse
import os
import sys

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ggml_lib import GgmlLib  # noqa: E402

# ggml_type ids (ggml/include/ggml.h) of every type the engine registers a GGUF codec for.
TYPES = {"Q4_0": 2, "Q4_1": 3, "Q5_0": 6, "Q5_1": 7, "Q8_0": 8, "Q2_K": 10, "Q3_K": 11, "Q4_K": 12, "Q5_K": 13, "Q6_K": 14,
         "IQ2_XXS": 16, "IQ2_XS": 17, "IQ3_XXS": 18, "IQ1_S": 19, "IQ4_NL": 20, "IQ3_S": 21, "IQ2_S": 22, "IQ4_XS": 23,
         "IQ1_M": 29, "TQ1_0": 34, "TQ2_0": 35, "MXFP4": 39, "Q1_0": 41, "Q2_0": 42}
ROWS, ROW = 4, 512

parser = argparse.ArgumentParser()
parser.add_argument("--ggml-lib", required=True)
parser.add_argument("--seed", type=int, default=11)
args = parser.parse_args()
ggml = GgmlLib(args.ggml_lib)

rng = np.random.default_rng(args.seed)
src = rng.standard_normal((ROWS, ROW), dtype=np.float32)
src[1] *= 1000.0
imatrix = np.ones(ROW, dtype=np.float32)
out = {}
for name, type_id in TYPES.items():
    assert ggml.traits(type_id).type_name.decode().upper() == name, name
    blocks = ggml.quantize(type_id, src, imatrix)
    values = ggml.dequantize(type_id, blocks, src.size)
    out[f"{name}.blocks"] = torch.from_numpy(blocks)
    out[f"{name}.values"] = torch.from_numpy(values)
    print(f"{name}: {blocks.size} bytes -> {values.size} values, max |v| {np.abs(values).max():.4g}")

here = os.path.dirname(os.path.abspath(__file__))
dest = os.path.join(here, "..", "HartsyInference.ModelAssets.Tests", "Fixtures", "GgmlCodecs")
os.makedirs(dest, exist_ok=True)
save_file(out, os.path.join(dest, "ggml_codec_blocks.safetensors"))

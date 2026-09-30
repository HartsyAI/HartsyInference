"""Dump small deterministic fixtures for the DeepSeek-V4.1-Flash derivative quant formats (NVFP4-ModelOpt, Quark MXFP4, MLX affine 4/8-bit).

Each reference decoder is written here from the format's own semantics, never from the C# codecs:
  * NVFP4: modelopt NVFP4QTensor.dequantize (low nibble = even element, row-major E4M3 scale per 16, F32 weight_scale_2, scales multiplied first).
  * Quark MXFP4: U8 E2M1 pairs (low nibble first) with raw U8 E8M0 scale bytes per 32, value = fp4 * 2**(byte - 127).
  * MLX affine: mlx.core.quantize / mlx.core.dequantize on CPU, bits 4 and 8, group size 64 (needs `pip install mlx[cpu]`).
Output is the committed Unit fixture fixtures/derivative_quant_codecs.json. Values are stored as IEEE-754 bit patterns.
Usage: python dump_derivative_quant_fixtures.py [--out fixtures/derivative_quant_codecs.json]
"""
import argparse
import json
import os

import numpy as np
import torch

HERE: str = os.path.dirname(os.path.abspath(__file__))
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)


def bits(a: np.ndarray) -> list:
    return [int(v) for v in np.ascontiguousarray(a, dtype=np.float32).reshape(-1).view(np.uint32)]


def unpack_low_first(packed: np.ndarray) -> np.ndarray:
    out = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.int64)
    out[:, 0::2] = packed & 15
    out[:, 1::2] = packed >> 4
    return out


def nvfp4_case(seed: int, rows: int, cols: int, global_scale: float) -> dict:
    rng = np.random.default_rng(seed)
    packed = rng.integers(0, 256, size=(rows, cols // 2), dtype=np.uint8)
    scale_bytes = rng.integers(0, 256, size=(rows, cols // 16), dtype=np.uint8)
    scale_bytes[0, 0] = 0x7F  # NaN
    scale_bytes[0, 1] = 0x01  # smallest subnormal
    scale_bytes[1, 0] = 0x7E  # 448
    scale_bytes[1, 1] = 0x00
    packed[0, 0] = 0x88  # nibble 8 is +0.0, not -0.0
    scales = torch.from_numpy(scale_bytes.copy()).view(torch.float8_e4m3fn).to(torch.float32).numpy()
    combined = scales * np.float32(global_scale)
    values = E2M1[unpack_low_first(packed)].reshape(rows, cols // 16, 16) * combined[:, :, None]
    return {"rows": rows, "cols": cols, "globalScaleBits": bits(np.float32([global_scale]))[0],
            "packedHex": packed.tobytes().hex(), "scaleHex": scale_bytes.tobytes().hex(),
            "expectedBits": bits(values.reshape(rows, cols))}


def quark_case(seed: int, rows: int, cols: int) -> dict:
    rng = np.random.default_rng(seed)
    packed = rng.integers(0, 256, size=(rows, cols // 2), dtype=np.uint8)
    scale_bytes = rng.integers(100, 150, size=(rows, cols // 32), dtype=np.uint8)
    scale_bytes[0, 0] = 255
    scale_bytes[1, 0] = 0
    with np.errstate(over="ignore"):
        e8m0 = np.where(scale_bytes == 255, np.float32("nan"),
                    np.where(scale_bytes == 0, np.float32(2.0) ** np.float32(-127), np.exp2(scale_bytes.astype(np.float32) - 127)))
    values = E2M1[unpack_low_first(packed)].reshape(rows, cols // 32, 32) * e8m0.astype(np.float32)[:, :, None]
    return {"rows": rows, "cols": cols, "packedHex": packed.tobytes().hex(), "scaleHex": scale_bytes.tobytes().hex(),
            "expectedBits": bits(values.reshape(rows, cols))}


def mlx_case(seed: int, rows: int, cols: int, nbits: int) -> dict:
    import mlx.core as mx

    rng = np.random.default_rng(seed)
    w = mx.array((rng.standard_normal((rows, cols)) * 0.08).astype(np.float32))
    wq, scales, biases = mx.quantize(w, group_size=64, bits=nbits)
    # Real conversions store scales and biases as F32, so the reference dequantizes with F32 parameters.
    deq = mx.dequantize(wq, scales.astype(mx.float32), biases.astype(mx.float32), group_size=64, bits=nbits)
    mx.eval(wq, scales, biases, deq)
    packed = np.array(wq).view(np.uint8).reshape(rows, -1)
    return {"rows": rows, "cols": cols, "bits": nbits, "packedHex": packed.tobytes().hex(),
            "scaleBits": bits(np.array(scales, dtype=np.float32)), "biasBits": bits(np.array(biases, dtype=np.float32)),
            "expectedBits": bits(np.array(deq, dtype=np.float32))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "fixtures", "derivative_quant_codecs.json"))
    a = ap.parse_args()
    out = {
        "nvfp4": [nvfp4_case(11, 4, 64, 0.0031), nvfp4_case(12, 3, 96, 7.5)],
        "quark": [quark_case(21, 3, 96)],
        "mlxAffine": [mlx_case(31, 4, 128, 4), mlx_case(32, 3, 192, 4), mlx_case(33, 4, 128, 8), mlx_case(34, 3, 192, 8)],
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()

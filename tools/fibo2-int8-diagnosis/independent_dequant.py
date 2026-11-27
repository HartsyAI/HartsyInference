"""Independent dequantization check for the Fibo 2.0 int8_tensorwise + ConvRot checkpoint. numpy only; no engine code.

Definition (from ComfyUI's comfy_kitchen tensor/int8_utils.py, read from the pinned source):
  H_G  = normalized Kronecker power of H4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]], scaled by 1/sqrt(G), G a power of 4.
  quantize: W_rot = W_grouped @ H_G^T (per group of G input columns), then int8 q with per-row scale s:  W_rot ~= q * s[row].
  dequantize: W_rot = q * s[row];  W_grouped = W_rot_grouped @ H_G  (H_G is orthogonal, so its inverse of H_G^T is H_G).

Usage:
  python independent_dequant.py compare <int8.safetensors> <engine_dequant_bf16.safetensors> [--max-keys N]
  python independent_dequant.py selftest

compare: for each int8 weight that has a .weight_scale and a .comfy_quant descriptor, dequantize it here and compare with the
same key in the engine's output file (read as BF16). Prints max abs and relative error per tensor and a summary.
"""

from __future__ import annotations

import json
import struct
import sys

import numpy as np

H4 = np.array([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=np.float64)


def hadamard(group: int) -> np.ndarray:
    if not _is_pow4(group):
        raise ValueError(f"ConvRot group must be a power of 4, got {group}")
    h = H4.copy()
    size = 4
    while size < group:
        h = np.kron(h, H4)
        size *= 4
    return h / np.sqrt(group)


def _is_pow4(n: int) -> bool:
    if n < 4:
        return False
    while n > 1:
        if n % 4:
            return False
        n //= 4
    return n == 1


def read_safetensors(path: str) -> dict:
    """name -> (dtype, shape, raw bytes). Reads the header JSON directly, so BF16 is kept as raw bits."""
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
        base = 8 + header_len
        out = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            start, end = meta["data_offsets"]
            f.seek(base + start)
            out[name] = (meta["dtype"], tuple(meta["shape"]), f.read(end - start))
    return out


def as_float(dtype: str, shape: tuple, raw: bytes) -> np.ndarray:
    if dtype == "BF16":
        bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        return bits.view(np.float32).reshape(shape).astype(np.float64)
    if dtype == "F32":
        return np.frombuffer(raw, dtype="<f4").reshape(shape).astype(np.float64)
    if dtype == "I8":
        return np.frombuffer(raw, dtype=np.int8).reshape(shape).astype(np.float64)
    raise ValueError(f"unsupported dtype {dtype}")


def descriptor_group(raw: bytes) -> tuple[str, int]:
    """Reads the .comfy_quant JSON (trailing NUL padding tolerated). Returns (format, ConvRot group or 0)."""
    text = raw.rstrip(b"\x00").decode("utf-8")
    doc = json.loads(text)
    params = doc.get("params", {}) or {}
    group = int(params.get("convrot_groupsize", 256)) if params.get("convrot") else 0
    return doc["format"], group


def dequantize(q: np.ndarray, scale: np.ndarray, group: int) -> np.ndarray:
    rows, cols = q.shape
    w_rot = q * (scale.reshape(-1, 1) if scale.size == rows else scale.reshape(1, 1))
    if group == 0:
        return w_rot
    if cols % group:
        raise ValueError(f"in_features {cols} not divisible by ConvRot group {group}")
    h = hadamard(group)
    grouped = w_rot.reshape(rows, cols // group, group) @ h
    return grouped.reshape(rows, cols)


def quantize_for_selftest(w: np.ndarray, group: int) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = w.shape
    if group:
        h = hadamard(group)
        w = (w.reshape(rows, cols // group, group) @ h.T).reshape(rows, cols)
    scale = np.abs(w).max(axis=1) / 127.0
    scale[scale == 0] = 1.0
    q = np.clip(np.round(w / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale.astype(np.float32)


def compare(int8_path: str, engine_path: str, max_keys: int | None) -> int:
    src = read_safetensors(int8_path)
    eng = read_safetensors(engine_path)
    checked = 0
    worst_rel = 0.0
    failures = []
    for name, (dtype, shape, raw) in src.items():
        if dtype != "I8" or not name.endswith(".weight"):
            continue
        base = name[: -len(".weight")]
        scale_key, desc_key = base + ".weight_scale", base + ".comfy_quant"
        if scale_key not in src or desc_key not in src:
            continue
        if name not in eng:
            failures.append(f"{name}: missing in the engine output")
            continue
        _, group = descriptor_group(src[desc_key][2])
        q = as_float(dtype, shape, raw)
        scale = as_float(*src[scale_key])
        mine = dequantize(q, scale, group)
        ref = as_float(*eng[name])
        diff = np.abs(mine - ref)
        denom = np.maximum(np.abs(ref), 1e-12)
        rel = float(np.max(diff / denom))
        worst_rel = max(worst_rel, rel)
        checked += 1
        print(f"{name}: group={group} max_abs={diff.max():.3e} max_rel={rel:.3e}")
        if max_keys is not None and checked >= max_keys:
            break
    print(f"checked {checked} int8 weights; worst max_rel {worst_rel:.3e}")
    if checked == 0:
        failures.append("no int8 weight with a scale and descriptor was found")
    for line in failures:
        print("FAIL", line)
    return 1 if failures else 0


def selftest() -> int:
    rng = np.random.default_rng(0)
    for group in (0, 16, 64):
        cols = 64
        w = rng.standard_normal((8, cols))
        q, s = quantize_for_selftest(w, group)
        back = dequantize(q.astype(np.float64), s.astype(np.float64), group)
        err = np.abs(back - w).max() / np.abs(w).max()
        print(f"group {group}: worst relative error {err:.4f} (int8 grid limit ~0.5/127 per row)")
        if err > 0.02:
            print("FAIL: dequantization does not invert quantization")
            return 1
    print("selftest ok")
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "selftest":
        sys.exit(selftest())
    if len(sys.argv) >= 4 and sys.argv[1] == "compare":
        limit = None
        if "--max-keys" in sys.argv:
            limit = int(sys.argv[sys.argv.index("--max-keys") + 1])
        sys.exit(compare(sys.argv[2], sys.argv[3], limit))
    print(__doc__)
    sys.exit(2)

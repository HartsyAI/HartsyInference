"""Independent dequantization check for the Fibo 2.0 int8_tensorwise + ConvRot checkpoint. numpy only; no engine code.

Definition (from ComfyUI's comfy_kitchen tensor/int8_utils.py, read from the pinned source):
  H_G  = normalized Kronecker power of H4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]], scaled by 1/sqrt(G), G a power of 4.
  quantize: W_rot = W_grouped @ H_G^T (per group of G input columns), then int8 q with per-row scale s:  W_rot ~= q * s[row].
  dequantize: W_rot = q * s[row];  W_grouped = W_rot_grouped @ H_G  (H_G is orthogonal, so its inverse of H_G^T is H_G).

Usage:
  python independent_dequant.py compare <int8.safetensors> <engine_dequant_bf16.safetensors> [--max-keys N]
  python independent_dequant.py selftest

compare: for each int8 tensor, dequantize it here (scale from <base>.weight_scale, ConvRot settings from
<base>.comfy_quant) and compare with the same key in the engine's output file, read as BF16. Per tensor it prints
the max absolute error and the error divided by the tensor's largest absolute value. It fails when that ratio is
above the BF16 bound, when an int8 tensor has no scale or descriptor, when a key is missing from the engine output,
or when a shape or format does not match. Exit code 1 on any failure.

BF16 bound: the engine writes each value as one BF16 rounding of the exact result. Rounding to 8 significant bits moves
a value by at most half an ulp, which is at most 2^-8 of the value, so at most 2^-8 of the tensor's largest value.
A tensor above that bound means the engine did more than one rounding (for example it rotated in BF16) or computed a
different result.
"""

from __future__ import annotations

import contextlib
import io
import json
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

H4 = np.array([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=np.float64)
BF16_BOUND = 2.0 ** -8


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


def write_safetensors(path: str, tensors: dict) -> None:
    """Writes name -> (dtype, shape, raw bytes) in the safetensors layout, for the selftest."""
    header = {}
    offset = 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
    hdr = json.dumps(header).encode("utf-8")
    hdr += b" " * (-len(hdr) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr)))
        f.write(hdr)
        for _, _, raw in tensors.values():
            f.write(raw)


def as_float(dtype: str, shape: tuple, raw: bytes) -> np.ndarray:
    if dtype == "BF16":
        bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        return bits.view(np.float32).reshape(shape).astype(np.float64)
    if dtype == "F32":
        return np.frombuffer(raw, dtype="<f4").reshape(shape).astype(np.float64)
    if dtype == "I8":
        return np.frombuffer(raw, dtype=np.int8).reshape(shape).astype(np.float64)
    raise ValueError(f"unsupported dtype {dtype}")


def f32_to_bf16_bytes(values: np.ndarray) -> bytes:
    """Round to nearest, ties to even, as BF16 bits (inputs here are finite)."""
    bits = np.asarray(values, dtype=np.float32).reshape(-1).view(np.uint32).astype(np.uint64)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return rounded.astype("<u2").tobytes()


def descriptor_settings(raw: bytes) -> tuple[str, int, str]:
    """Reads the .comfy_quant JSON (trailing NUL padding tolerated). Returns (format, ConvRot group or 0, where).

    ComfyUI writes the ConvRot keys either at the top level or under "params"; both are read. If both places set a key,
    the values must agree. A set convrot flag without a group size raises instead of guessing a size.
    """
    doc = json.loads(raw.rstrip(b"\x00").decode("utf-8"))
    if "format" not in doc:
        raise ValueError("comfy_quant descriptor has no format")
    sources = {"top": doc}
    if isinstance(doc.get("params"), dict):
        sources["params"] = doc["params"]
    flags = {where: bool(src["convrot"]) for where, src in sources.items() if "convrot" in src}
    sizes = {where: int(src["convrot_groupsize"]) for where, src in sources.items() if "convrot_groupsize" in src}
    if len(set(flags.values())) > 1:
        raise ValueError(f"convrot flag disagrees between layouts: {flags}")
    if len(set(sizes.values())) > 1:
        raise ValueError(f"convrot_groupsize disagrees between layouts: {sizes}")
    if not any(flags.values()):
        return doc["format"], 0, "no convrot"
    if not sizes:
        raise ValueError("convrot is set but convrot_groupsize is absent at the top level and under params")
    where = "/".join(sorted(sizes))
    return doc["format"], next(iter(sizes.values())), where


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
    worst, worst_name = 0.0, ""
    failures = []
    for name, (dtype, shape, raw) in src.items():
        if dtype != "I8":
            continue
        base = name[: -len(".weight")] if name.endswith(".weight") else name
        scale_key, desc_key = base + ".weight_scale", base + ".comfy_quant"
        if scale_key not in src or desc_key not in src:
            failures.append(f"{name}: int8 tensor without {scale_key} or {desc_key}")
            continue
        if name not in eng:
            failures.append(f"{name}: missing in the engine output")
            continue
        fmt, group, where = descriptor_settings(src[desc_key][2])
        if fmt != "int8_tensorwise":
            failures.append(f"{name}: descriptor format {fmt!r}, expected 'int8_tensorwise'")
            continue
        ref_dtype, ref_shape, ref_raw = eng[name]
        if tuple(ref_shape) != tuple(shape):
            failures.append(f"{name}: shape {tuple(shape)} in the int8 file, {tuple(ref_shape)} in the engine output")
            continue
        mine = dequantize(as_float(dtype, shape, raw), as_float(*src[scale_key]), group)
        ref = as_float(ref_dtype, ref_shape, ref_raw)
        diff = np.abs(mine - ref)
        tensor_max = float(np.abs(ref).max())
        rel = float(diff.max()) / tensor_max if tensor_max > 0 else (0.0 if diff.max() == 0 else float("inf"))
        checked += 1
        print(f"{name}: convrot={group} ({where}) max_abs={diff.max():.3e} tensor_max={tensor_max:.3e} "
              f"error/tensor_max={rel:.3e}")
        if rel > BF16_BOUND:
            failures.append(f"{name}: error {rel:.3e} of the tensor max is above the BF16 bound {BF16_BOUND:.3e}")
        if rel > worst:
            worst, worst_name = rel, name
        if max_keys is not None and checked >= max_keys:
            break
    print(f"checked {checked} int8 tensors; worst error/tensor_max {worst:.3e} at {worst_name or '-'}; "
          f"BF16 bound {BF16_BOUND:.3e}")
    if checked == 0:
        failures.append("no int8 tensor with a scale and descriptor was compared")
    for line in failures:
        print("FAIL", line)
    return 1 if failures else 0


def _settings_cases() -> None:
    ok_cases = [
        (b'{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":16}', ("int8_tensorwise", 16)),
        (b'{"format":"int8_tensorwise","params":{"convrot":true,"convrot_groupsize":16}}', ("int8_tensorwise", 16)),
        (b'{"format":"int8_tensorwise","params":{}}', ("int8_tensorwise", 0)),
        (b'{"format":"int8_tensorwise","convrot":false}\x00\x00', ("int8_tensorwise", 0)),
    ]
    for raw, want in ok_cases:
        got = descriptor_settings(raw)[:2]
        if got != want:
            raise AssertionError(f"descriptor {raw!r}: got {got}, want {want}")
    for raw in (
        b'{"format":"int8_tensorwise","convrot":true}',
        b'{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":16,"params":{"convrot_groupsize":64}}',
    ):
        try:
            descriptor_settings(raw)
        except ValueError:
            continue
        raise AssertionError(f"descriptor {raw!r} should have been refused")


def _end_to_end(tmp: Path, layout: str, group: int, engine_group: int, perturb: bool) -> int:
    rng = np.random.default_rng(1)
    w = rng.standard_normal((8, 64))
    q, s = quantize_for_selftest(w, group)
    if layout == "top":
        desc = {"format": "int8_tensorwise", "convrot": group > 0, "convrot_groupsize": group}
    else:
        desc = {"format": "int8_tensorwise", "params": {"convrot": group > 0, "convrot_groupsize": group}}
    if group == 0:
        desc = {"format": "int8_tensorwise", "params": {}}
    desc_bytes = json.dumps(desc).encode("utf-8")
    int8_file = tmp / f"int8_{layout}_{group}.safetensors"
    engine_file = tmp / f"engine_{layout}_{group}.safetensors"
    write_safetensors(str(int8_file), {
        "blk.weight": ("I8", q.shape, q.tobytes()),
        "blk.weight_scale": ("F32", s.shape, s.astype("<f4").tobytes()),
        "blk.comfy_quant": ("U8", (len(desc_bytes),), desc_bytes),
    })
    ref = dequantize(q.astype(np.float64), s.astype(np.float64), engine_group)
    if perturb:
        ref[0, 0] += 0.02 * np.abs(ref).max()
    write_safetensors(str(engine_file), {"blk.weight": ("BF16", ref.shape, f32_to_bf16_bytes(ref))})
    with contextlib.redirect_stdout(io.StringIO()):
        return compare(str(int8_file), str(engine_file), None)


def selftest() -> int:
    rng = np.random.default_rng(0)
    for group in (0, 16, 64):
        w = rng.standard_normal((8, 64))
        q, s = quantize_for_selftest(w, group)
        back = dequantize(q.astype(np.float64), s.astype(np.float64), group)
        err = np.abs(back - w).max() / np.abs(w).max()
        print(f"group {group}: worst relative error {err:.4f} (int8 grid limit ~0.5/127 per row)")
        if err > 0.02:
            print("FAIL: dequantization does not invert quantization")
            return 1
    _settings_cases()
    print("descriptor layouts ok (top-level, params, conflicts refused)")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        checks = [
            ("top-level ConvRot, engine rotated", _end_to_end(tmp, "top", 16, 16, False), 0),
            ("params ConvRot, engine rotated", _end_to_end(tmp, "params", 16, 16, False), 0),
            ("engine unrotated must fail", _end_to_end(tmp, "top", 16, 0, False), 1),
            ("a 2% element error must fail", _end_to_end(tmp, "top", 16, 16, True), 1),
        ]
    for label, got, want in checks:
        print(f"end to end, {label}: exit {got} (want {want})")
        if got != want:
            print("FAIL: compare did not behave as the case requires")
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

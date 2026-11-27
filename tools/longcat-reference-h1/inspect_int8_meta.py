"""Read-only inspection of the INT8 DiT build: meta device, the same Linear replacement as load_quantized_dit, and the checkpoint's
shard headers and index. Reads no tensor data: the model lives on the meta device, and each shard is read only as its JSON header.

Run from the reference checkout root, in the reference venv:
  python inspect_int8_meta.py <checkpoint_dir> [--out <file>]
  e.g. python inspect_int8_meta.py /mnt/model-storage/Models/LongCat/avatar-1.6 --out /home/hartsy/refs/inspect-int8.txt

<checkpoint_dir> must contain base_model_int8/config.json, base_model_int8/quantized_model.safetensors.index.json and the shards.

Output:
- parameter count and buffer count of the meta model, and the total element count;
- the key comparison: model state_dict keys against the index keys, with shape and dtype compared to each shard header;
  missing keys (model, not in checkpoint), unexpected keys (checkpoint, not in model), shape mismatches, dtype mismatches;
- every parameter and buffer, one line each: name, dtype, shape, persistent or not, and whether it is in the checkpoint index;
- every tensor-valued attribute on a module that is not a parameter or buffer (for example RoPE caches stored as attributes),
  because a meta-device build leaves those on meta unless they are materialised.
Exit status 1 if any key comparison fails.
"""

import json
import os
import struct
import sys

import torch
import torch.nn as nn

REF_ROOT = os.getcwd()
sys.path.insert(0, REF_ROOT)

from longcat_video.modules.quantization import DEFAULT_SKIP_PATTERNS, QuantizedLinear  # noqa: E402
from longcat_video.modules.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel  # noqa: E402

DTYPE_BYTES = {"F32": 4, "F16": 2, "BF16": 2, "I8": 1, "U8": 1, "I32": 4, "I64": 8}


def read_safetensors_header(path):
    """Only the JSON header of a .safetensors file: its tensors' dtype, shape and offsets. No tensor data is read."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header


def build_meta_model(config):
    """The same construction and replacement as load_quantized_dit (quantization.py lines 187 to 205), on the meta device."""
    with torch.device("meta"):
        model = LongCatVideoAvatarTransformer3DModel(**config)
        replacements = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and not any(pattern in name for pattern in DEFAULT_SKIP_PATTERNS):
                replacements[name] = QuantizedLinear(module.in_features, module.out_features, bias=module.bias is not None)
        for name, ql in replacements.items():
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], ql)
    return model, len(replacements)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    checkpoint_dir = sys.argv[1]
    out_path = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    lines = []

    def emit(text=""):
        print(text)
        lines.append(text)

    qdir = os.path.join(checkpoint_dir, "base_model_int8")
    config = json.load(open(os.path.join(qdir, "config.json"), encoding="utf-8"))
    for key in ["_class_name", "architectures", "_diffusers_version", "model_max_length"]:
        config.pop(key, None)

    model, replaced = build_meta_model(config)
    params = list(model.named_parameters())
    buffers = list(model.named_buffers())
    persistent = set(model.state_dict().keys())
    emit(f"Linear layers replaced by QuantizedLinear: {replaced}")
    emit(f"parameters: {len(params)}, elements: {sum(p.numel() for _, p in params):,}")
    emit(f"buffers: {len(buffers)}, elements: {sum(b.numel() for _, b in buffers):,}")

    index_path = os.path.join(qdir, "quantized_model.safetensors.index.json")
    index = json.load(open(index_path, encoding="utf-8"))
    weight_map = index["weight_map"]
    shard_headers = {}
    for shard in sorted(set(weight_map.values())):
        shard_headers.update({name: (shard, meta) for name, meta in read_safetensors_header(os.path.join(qdir, shard)).items()})

    failures = 0
    model_state = model.state_dict()
    missing = sorted(k for k in model_state if k not in shard_headers)
    unexpected = sorted(k for k in shard_headers if k not in model_state)
    # The only allowed dtype difference: a float32 model parameter whose checkpoint tensor is BF16. load_quantized_dit
    # converts it to BF16 at load (the stock path copies it into float32 and casts back to BF16, the patched path converts
    # it directly; both give the same BF16 values). Every other shape or dtype difference is a failure.
    ALLOWED_CONVERSIONS = 486
    fp32_params = {name for name, p in params if p.dtype == torch.float32}
    mismatched = []
    converted = []
    for k, tensor in model_state.items():
        if k not in shard_headers:
            continue
        _, meta = shard_headers[k]
        if list(tensor.shape) != meta["shape"]:
            mismatched.append((k, "shape", list(tensor.shape), meta["shape"]))
        expected_dtype = {torch.float32: "F32", torch.bfloat16: "BF16", torch.float16: "F16", torch.int8: "I8", torch.uint8: "U8"}.get(tensor.dtype)
        if expected_dtype != meta["dtype"]:
            if k in fp32_params and meta["dtype"] == "BF16":
                converted.append(k)
            else:
                mismatched.append((k, "dtype", str(tensor.dtype), meta["dtype"]))
    emit()
    emit(f"model state_dict keys: {len(model_state)}; checkpoint tensors: {len(shard_headers)}")
    emit(f"missing from checkpoint (model only): {len(missing)}")
    for k in missing[:200]:
        emit(f"  MISSING {k}")
    emit(f"unexpected in checkpoint (not in model): {len(unexpected)}")
    for k in unexpected[:200]:
        emit(f"  UNEXPECTED {k}")
    emit(f"shape or dtype mismatches (other than the allowed float32-model / BF16-checkpoint conversion): {len(mismatched)}")
    for item in mismatched[:200]:
        emit(f"  MISMATCH {item}")
    emit(f"allowed conversions (float32 model parameter, BF16 checkpoint tensor): {len(converted)} (must be {ALLOWED_CONVERSIONS})")
    if missing or unexpected or mismatched:
        failures += 1
    if len(converted) != ALLOWED_CONVERSIONS:
        emit(f"  CONVERSION COUNT {len(converted)} != {ALLOWED_CONVERSIONS}")
        failures += 1

    emit()
    emit("every parameter and buffer (name | dtype | shape | persistent | in checkpoint):")
    for name, p in params:
        emit(f"  P {name} | {p.dtype} | {list(p.shape)} | yes | {'yes' if name in shard_headers else 'NO'}")
    for name, b in buffers:
        emit(f"  B {name} | {b.dtype} | {list(b.shape)} | {'yes' if name in persistent else 'NO (non-persistent)'} | "
             f"{'yes' if name in shard_headers else 'NO'}")

    emit()
    emit("tensor-valued attributes that are not parameters or buffers (a meta build leaves these on meta):")
    attr_count = 0
    for mod_name, module in model.named_modules():
        for attr, value in vars(module).items():
            if attr.startswith("_"):
                continue
            if isinstance(value, torch.Tensor):
                attr_count += 1
                emit(f"  A {mod_name}.{attr} | {value.dtype} | {list(value.shape)} | meta={value.is_meta}")
    if attr_count == 0:
        emit("  none")

    emit()
    emit(f"RESULT: {'FAIL' if failures else 'PASS'} (key and shape checks); tensor-valued attributes: {attr_count}")
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

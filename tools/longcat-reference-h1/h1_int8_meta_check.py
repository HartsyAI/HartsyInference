"""CPU check of the H1 meta-device load (h1-int8-meta.patch) against the stock load_quantized_dit, on a tiny config.

Run: python h1_int8_meta_check.py <reference root> <patched quantization.py> <config.json of the 1.6 INT8 release>

The tiny checkpoint is written in the release layout: BF16 parameters, INT8 weight buffers, float32 weight scales, BF16 bias
buffers, one shard and quantized_model.safetensors.index.json. The stock and the patched loaders load the same directory.
The check passes only if every tensor matches the stock load in value and dtype, nothing is left on the meta device, and two
broken checkpoints are refused: one with a missing key, one with a wrong dtype on an INT8 buffer.
"""

import importlib.machinery
import importlib.util
import json
import os
import sys
import tempfile
import types

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import h1_sdpa_check as stubs  # noqa: E402  (triton and block-sparse stubs)

TINY = {"hidden_size": 64, "depth": 2, "num_heads": 4, "caption_channels": 64}
SHARD = "quantized_model-00001-of-00001.safetensors"


def tiny_config(release_config):
    config = dict(release_config)
    for key in ["_class_name", "architectures", "_diffusers_version", "model_max_length"]:
        config.pop(key, None)
    config.update(TINY)
    return config


def install_flash_placeholders():
    """The avatar modules may import flash_attn at import time. It is never called here (no forward pass)."""
    try:
        import flash_attn  # noqa: F401
        return
    except ImportError:
        pass

    def unused(*args, **kwargs):
        raise RuntimeError("flash_attn is not called by this check")

    fake = types.ModuleType("flash_attn")
    fake.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)
    fake.flash_attn_func = unused
    fake.flash_attn_varlen_func = unused
    sys.modules["flash_attn"] = fake


def build_stock_model(stock, config):
    from longcat_video.modules.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel
    model = LongCatVideoAvatarTransformer3DModel(**config)
    replacements = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not any(p in name for p in stock.DEFAULT_SKIP_PATTERNS):
            replacements[name] = stock.QuantizedLinear(module.in_features, module.out_features, bias=module.bias is not None)
    for name, ql in replacements.items():
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], ql)
    return model


def write_checkpoint(stock, config, root):
    """A tiny checkpoint in the release layout: BF16 parameters, INT8 weights, float32 scales, BF16 biases."""
    torch.manual_seed(0)
    model = build_stock_model(stock, config)
    tensors = {}
    for name, param in model.named_parameters():
        tensors[name] = torch.randn(param.shape).to(torch.bfloat16)
    for name, buf in model.named_buffers():
        if buf.dtype == torch.int8:
            tensors[name] = torch.randint(-127, 128, buf.shape, dtype=torch.int8)
        elif name.endswith("weight_scale"):
            tensors[name] = (torch.rand(buf.shape) + 0.5).to(torch.float32)
        else:
            tensors[name] = torch.randn(buf.shape).to(buf.dtype)
    qdir = os.path.join(root, "base_model_int8")
    os.makedirs(qdir)
    with open(os.path.join(qdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f)
    save_file(tensors, os.path.join(qdir, SHARD))
    index = {"metadata": {"total_size": 0}, "weight_map": {k: SHARD for k in tensors}}
    with open(os.path.join(qdir, "quantized_model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f)
    return root


def load_patched(patched_path):
    spec = importlib.util.spec_from_file_location("longcat_video.modules._quant_patched", patched_path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "longcat_video.modules"
    spec.loader.exec_module(module)
    return module


def main(reference_root, patched_path, release_config_path):
    stubs.install_stubs(reference_root)
    install_flash_placeholders()
    torch.compile = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
    from longcat_video.modules import quantization as stock
    patched = load_patched(patched_path)
    # Guards: "stock" must be the pristine pinned file, or the comparison is patched against patched.
    if hasattr(stock, "_h1_assign_state"):
        raise SystemExit("reference root's quantization.py already has the H1 patch; run the check on a pristine checkout")
    if not hasattr(patched, "_h1_assign_state"):
        raise SystemExit("the patched file has no _h1_assign_state: it is not the H1 patch")
    with open(release_config_path, encoding="utf-8") as f:
        config = tiny_config(json.load(f))

    with tempfile.TemporaryDirectory() as tmp:
        root = write_checkpoint(stock, config, os.path.join(tmp, "good"))
        stock_model = stock.load_quantized_dit(root)
        patched_model = patched.load_quantized_dit(root)

        stock_state = stock_model.state_dict()
        patched_state = patched_model.state_dict()
        assert stock_state.keys() == patched_state.keys(), "key sets differ"
        differences = []
        for name, a in stock_state.items():
            b = patched_state[name]
            if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
                differences.append(name)
        assert not differences, f"{len(differences)} tensors differ, e.g. {differences[:5]}"
        meta_left = [n for n, t in list(patched_model.named_parameters()) + list(patched_model.named_buffers()) if t.is_meta]
        assert not meta_left, meta_left
        dtype_counts = {}
        for t in patched_state.values():
            dtype_counts[str(t.dtype)] = dtype_counts.get(str(t.dtype), 0) + 1
        print(f"PASS: {len(stock_state)} tensors identical to the stock load in value, shape and dtype; "
              f"dtypes {dtype_counts}; nothing on meta.")

        def expect_refused(label, ckpt_root):
            try:
                patched.load_quantized_dit(ckpt_root)
            except RuntimeError as exc:
                print(f"PASS (refused, {label}): {str(exc)[:140]}")
            else:
                raise AssertionError(f"{label}: the patched load accepted a broken checkpoint")

        missing_root = write_checkpoint(stock, config, os.path.join(tmp, "missing"))
        qdir = os.path.join(missing_root, "base_model_int8")
        shard_path = os.path.join(qdir, SHARD)
        index_path = os.path.join(qdir, "quantized_model.safetensors.index.json")
        tensors = load_file(shard_path)
        dropped = sorted(tensors)[0]
        del tensors[dropped]
        save_file(tensors, shard_path)
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
        del index["weight_map"][dropped]
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f)
        expect_refused(f"missing key {dropped}", missing_root)

        dtype_root = write_checkpoint(stock, config, os.path.join(tmp, "dtype"))
        shard_path = os.path.join(dtype_root, "base_model_int8", SHARD)
        tensors = load_file(shard_path)
        int8_key = next(k for k, v in tensors.items() if v.dtype == torch.int8)
        tensors[int8_key] = tensors[int8_key].to(torch.float32)
        save_file(tensors, shard_path)
        expect_refused(f"wrong dtype on {int8_key}", dtype_root)

        extra_root = write_checkpoint(stock, config, os.path.join(tmp, "extra"))
        qdir = os.path.join(extra_root, "base_model_int8")
        shard_path = os.path.join(qdir, SHARD)
        index_path = os.path.join(qdir, "quantized_model.safetensors.index.json")
        tensors = load_file(shard_path)
        extra = "unexpected.extra_tensor"
        tensors[extra] = torch.zeros(4, dtype=torch.bfloat16)
        save_file(tensors, shard_path)
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
        index["weight_map"][extra] = SHARD
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f)
        expect_refused(f"unexpected key {extra}", extra_root)
    print("ALL PASS")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])

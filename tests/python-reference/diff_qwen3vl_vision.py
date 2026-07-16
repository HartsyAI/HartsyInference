"""Qwen3-VL vision-tower parity: HF reference vs the HartsyInference dump.

Usage:
  HARTSY_QWEN3VL_PATH=<checkpoint.safetensors> HARTSY_QWEN3VL_DUMP_DIR=<dir> python diff_qwen3vl_vision.py

Loads the BF16 `visual.*` weights from the same checkpoint into HF's Qwen3VLVisionModel (float32, CPU),
feeds the engine-dumped pixel_values/grid, and reports correlation + max-abs-diff for the merged tokens
and each deepstack feature. Also cross-checks the engine's preprocessed pixel_values against HF's
Qwen2VLImageProcessor on the dumped RGB image.
"""
import json
import os
import struct
import sys

import numpy as np
import torch
from safetensors import safe_open
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

CKPT = os.environ["HARTSY_QWEN3VL_PATH"]
DUMP = os.environ["HARTSY_QWEN3VL_DUMP_DIR"]


def read_bin(name, shape):
    a = np.fromfile(os.path.join(DUMP, name), dtype=np.float32)
    return a.reshape(shape)


def stats(name, ref, got):
    ref = ref.reshape(-1).astype(np.float64)
    got = got.reshape(-1).astype(np.float64)
    corr = float(np.corrcoef(ref, got)[0, 1])
    mad = float(np.max(np.abs(ref - got)))
    rel = mad / (float(np.max(np.abs(ref))) + 1e-9)
    print(f"{name:18s} corr={corr:.6f} maxAbs={mad:.4e} rel={rel:.4e} refAbsMax={np.max(np.abs(ref)):.3f}")
    return corr


def main():
    gt, gh, gw = (int(x) for x in open(os.path.join(DUMP, "grid.txt")).read().split())
    seq = gt * gh * gw
    pix = read_bin("pixel_values.bin", (seq, 1536))

    cfg = Qwen3VLVisionConfig(
        depth=27, hidden_size=1152, num_heads=16, intermediate_size=4304,
        patch_size=16, spatial_merge_size=2, temporal_patch_size=2,
        out_hidden_size=4096, num_position_embeddings=2304,
        deepstack_visual_indexes=[8, 16, 24], hidden_act="gelu_pytorch_tanh", in_channels=3,
    )
    model = Qwen3VLVisionModel(cfg).eval().float()

    weights = {}
    with safe_open(CKPT, framework="pt") as f:
        for k in f.keys():
            i = k.find("visual.")
            if i < 0:
                continue
            weights[k[i + len("visual."):]] = f.get_tensor(k).float()
    missing, unexpected = model.load_state_dict(weights, strict=False)
    print("missing:", [m for m in missing][:8], "unexpected:", [u for u in unexpected][:8])

    with torch.no_grad():
        out = model(
            hidden_states=torch.from_numpy(pix),
            grid_thw=torch.tensor([[gt, gh, gw]], dtype=torch.long),
        )
    hidden = out.pooler_output          # merged image tokens [n_merged, out_hidden]
    deepstack = out.deepstack_features

    n_merged = seq // 4
    merged_engine = read_bin("merged_tokens.bin", (n_merged, 4096))
    ok = stats("merged_tokens", hidden.numpy(), merged_engine)
    for i in range(3):
        de = read_bin(f"deepstack_{i}.bin", (n_merged, 4096))
        stats(f"deepstack_{i}", deepstack[i].numpy(), de)

    # image-processor cross-check
    try:
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
        rgb = read_bin("rgb.bin", (3, 224, 168))
        proc = Qwen2VLImageProcessor(
            patch_size=16, merge_size=2, temporal_patch_size=2,
            min_pixels=32 * 32 * 4, max_pixels=32 * 32 * 1024,
        )
        out = proc(images=[torch.from_numpy(rgb)], do_rescale=False, return_tensors="pt")
        hf_pix = out["pixel_values"].numpy()
        hf_grid = out["image_grid_thw"].numpy()
        print("hf grid:", hf_grid, "engine grid:", (gt, gh, gw))
        if hf_pix.shape == pix.shape:
            stats("pixel_values", hf_pix, pix)
        else:
            print("pixel_values shape mismatch:", hf_pix.shape, pix.shape)
    except Exception as e:
        print("image-processor cross-check skipped:", e)

    sys.exit(0 if ok > 0.999 else 1)


if __name__ == "__main__":
    main()

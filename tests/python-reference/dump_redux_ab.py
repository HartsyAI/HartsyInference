#!/usr/bin/env python3
"""Numeric A/B reference for the FLUX.1 Redux conditioning path.

Builds the exact diffusers/BFL reference (SigLIP-so400m/14-384 vision tower ->
last_hidden_state -> redux_down(silu(redux_up(x)))) from LOCAL checkpoints (no downloads):
  - clip_vision/sigclip_vision_patch14_384.safetensors  (HF SiglipVisionModel state dict)
  - style_models/flux1-redux-dev.safetensors            (redux_up/redux_down)

Dumps to --out:
  pixel_values.bin      [1,3,384,384] F32 — HF SiglipImageProcessor preprocessing of the input image
                        (bicubic PIL resize + rescale + 0.5/0.5 normalize). Shared input for the C# side.
  ref_hidden.bin        [1,729,1152] F32 — SigLIP last_hidden_state (post-layernorm)
  ref_tokens.bin        [1,729,4096] F32 — Redux projected conditioning tokens
  meta.json

Usage:
  python dump_redux_ab.py --image bus.png --sigclip <path> --redux <path> --out <dir>
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import SiglipVisionConfig, SiglipVisionModel
from transformers.models.siglip.image_processing_siglip import SiglipImageProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--sigclip", required=True)
    ap.add_argument("--redux", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    config = SiglipVisionConfig(
        hidden_size=1152, intermediate_size=4304, num_hidden_layers=27,
        num_attention_heads=16, image_size=384, patch_size=14,
    )
    model = SiglipVisionModel(config).eval()
    # transformers 5.x SiglipVisionModel state-dict keys have no "vision_model." prefix;
    # try the file keys as-is first, then stripped.
    sd = {k: v.float() for k, v in load_file(args.sigclip).items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        sd = {k.removeprefix("vision_model."): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
    print("missing:", missing[:5], "unexpected:", unexpected[:5])
    assert not missing, missing[:10]

    processor = SiglipImageProcessor(
        do_resize=True, size={"height": 384, "width": 384}, resample=3,
        do_rescale=True, do_normalize=True, image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5],
    )
    image = Image.open(args.image).convert("RGB")
    pixel_values = processor(images=[image], return_tensors="pt")["pixel_values"]
    pixel_values.numpy().astype(np.float32).tofile(os.path.join(args.out, "pixel_values.bin"))

    with torch.no_grad():
        hidden = model(pixel_values=pixel_values).last_hidden_state
    hidden.numpy().astype(np.float32).tofile(os.path.join(args.out, "ref_hidden.bin"))

    redux_sd = {k: v.float() for k, v in load_file(args.redux).items()}
    up_w, up_b = redux_sd["redux_up.weight"], redux_sd["redux_up.bias"]
    down_w, down_b = redux_sd["redux_down.weight"], redux_sd["redux_down.bias"]
    with torch.no_grad():
        tokens = torch.nn.functional.linear(
            torch.nn.functional.silu(torch.nn.functional.linear(hidden, up_w, up_b)), down_w, down_b)
    tokens.numpy().astype(np.float32).tofile(os.path.join(args.out, "ref_tokens.bin"))

    meta = {
        "image": os.path.abspath(args.image),
        "hidden_shape": list(hidden.shape),
        "tokens_shape": list(tokens.shape),
        "tokens_mean": float(tokens.mean()), "tokens_std": float(tokens.std()),
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

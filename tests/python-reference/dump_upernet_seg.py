#!/usr/bin/env python3
"""Dump UperNet-ConvNeXt-Small ADE20K semantic-segmentation reference tensors.

This is the reference the diffusers / HF ecosystem documents for control_v11p_sd15_seg
conditioning (transformers UperNetForSemanticSegmentation, checkpoint
openmmlab/upernet-convnext-small). Runs the model on a real image resized to 512x512
(SegformerImageProcessor semantics: PIL bilinear stretch, /255, ImageNet mean/std) and dumps
stage taps plus the final ADE20K-palette RGB map, for the C# parity test
(UperNetSegParityTests).

Usage:
    python dump_upernet_seg.py --ckpt /path/upernet_convnext_small.bin [--image bus.png]

Outputs to tests/python-reference/upernet_seg_reference_tensors/:
    image.bin       512x512x3 uint8 source image (already resized — C# consumes this directly)
    input.bin       [1,3,512,512] F32 normalized network input
    feat_{0..3}.bin backbone feature maps after the per-stage hidden_states_norms
    psp_out.bin     PSP bottleneck output [1,512,16,16]
    fpn_out.bin     FPN bottleneck output [1,512,128,128]
    logits_q.bin    classifier logits at 1/4 resolution [1,150,128,128]
    seg_u8.bin      512x512 uint8 argmax class map (after bilinear logit upsample to 512)
    seg_rgb.bin     512x512x3 uint8 ADE20K-palette colorized map
    meta.json
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from transformers import ConvNextConfig, UperNetConfig, UperNetForSemanticSegmentation

SIZE = 512
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def ade_palette():
    """ADE20K palette, identical to controlnet_aux / comfyui_controlnet_aux ade_palette()."""
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ade20k_palette.json")) as f:
        return json.load(f)


def build_model(ckpt):
    backbone = ConvNextConfig(
        depths=[3, 3, 27, 3],
        hidden_sizes=[96, 192, 384, 768],
        out_features=["stage1", "stage2", "stage3", "stage4"],
    )
    config = UperNetConfig(
        backbone_config=backbone,
        hidden_size=512,
        pool_scales=[1, 2, 3, 6],
        num_labels=150,
        use_auxiliary_head=True,
        auxiliary_in_channels=384,
    )
    model = UperNetForSemanticSegmentation(config)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing, missing
    assert not unexpected, unexpected
    return model.float().eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out or os.path.join(here, "upernet_seg_reference_tensors")
    os.makedirs(out_dir, exist_ok=True)
    image_path = args.image or os.path.join(
        here, "..", "HartsyInference.Vision.Tests", "TestData", "bus.png")

    img = Image.open(image_path).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    img = np.asarray(img, np.uint8)
    img.tofile(os.path.join(out_dir, "image.bin"))

    x = (img.astype(np.float32) / 255.0 - MEAN) / STD
    x = torch.from_numpy(x.transpose(2, 0, 1)[None])
    x.numpy().astype(np.float32).tofile(os.path.join(out_dir, "input.bin"))

    model = build_model(args.ckpt)

    taps = {}

    def save_output(name):
        def hook(_m, _inp, out):
            taps[name] = out.detach()
        return hook

    model.decode_head.bottleneck.register_forward_hook(save_output("psp_out"))
    model.decode_head.fpn_bottleneck.register_forward_hook(save_output("fpn_out"))
    model.decode_head.classifier.register_forward_hook(save_output("logits_q"))

    with torch.no_grad():
        feats = model.backbone(x).feature_maps
        for i, f in enumerate(feats):
            f.numpy().astype(np.float32).tofile(os.path.join(out_dir, f"feat_{i}.bin"))
        outputs = model(x)

    for name, t in taps.items():
        t.numpy().astype(np.float32).tofile(os.path.join(out_dir, f"{name}.bin"))

    logits = outputs.logits  # [1,150,512,512], already upsampled align_corners=False
    seg = logits.argmax(1)[0].numpy().astype(np.uint8)
    seg.tofile(os.path.join(out_dir, "seg_u8.bin"))

    palette = np.array(ade_palette(), np.uint8)
    seg_rgb = palette[seg]
    seg_rgb.tofile(os.path.join(out_dir, "seg_rgb.bin"))

    hist = np.bincount(seg.ravel(), minlength=150)
    top = sorted(enumerate(hist.tolist()), key=lambda kv: -kv[1])[:8]
    meta = {
        "H": SIZE,
        "W": SIZE,
        "image": os.path.basename(image_path),
        "feat_shapes": [list(f.shape) for f in feats],
        "logits_q_shape": list(taps["logits_q"].shape),
        "top_classes": top,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("done:", out_dir)
    print("top classes:", top)


if __name__ == "__main__":
    main()

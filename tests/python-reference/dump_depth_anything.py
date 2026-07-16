#!/usr/bin/env python3
"""Dump Depth-Anything-V2 reference tensors from the OFFICIAL implementation.

Runs the official DepthAnythingV2 model (https://github.com/DepthAnything/Depth-Anything-V2,
cloned locally — set DEPTH_ANYTHING_V2_REPO) on a fixed deterministic input and dumps every
DPT-head intermediate, for the C# parity test (DepthAnythingParityTests) and diff script
(diff_depth_anything.py).

Usage:
    python dump_depth_anything.py --encoder vits --ckpt /path/depth_anything_v2_vits.pth
    python dump_depth_anything.py --encoder vitl --ckpt /path/depth_anything_v2_vitl.pth

Outputs to tests/python-reference/depth_anything_reference_tensors/<encoder>/:
    input.bin                 [1,3,H,W] the fixed network input (already "preprocessed")
    feat_{0..3}.bin           the 4 tapped patch-token features [1,N,C] (after final norm)
    layer_{1..4}.bin          after projects + resize convs
    layer_{1..4}_rn.bin       after the scratch 3x3 channel-unify convs
    path_{4..1}.bin           after each fusion block
    depth.bin                 final relative depth [1,1,H,W] (after ReLU)
    preprocess_input.bin      synthetic uint8 HWC image bytes (for the preprocessor test)
    preprocess_ref.bin        official transform output [3,h,w] for that image
    meta.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

SCRATCH_DEFAULT = os.path.expanduser("~/.cache/depth_anything_v2_repo")
REPO = os.environ.get("DEPTH_ANYTHING_V2_REPO", SCRATCH_DEFAULT)
sys.path.insert(0, REPO)

from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

# Non-square, non-native size to exercise pos-embed interpolation: H=280 (20 patches), W=378 (27).
H, W = 280, 378


def dump(path, t):
    t.detach().cpu().float().numpy().astype(np.float32).tofile(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="vits", choices=list(MODEL_CONFIGS))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "depth_anything_reference_tensors", args.encoder)
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(0)
    model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
    sd = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()

    rng = np.random.RandomState(42)
    # Plausible normalized-image statistics: uniform pixel in [0,1] then ImageNet normalize.
    pix = rng.rand(1, 3, H, W).astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)
    x = torch.from_numpy((pix - mean) / std)
    dump(os.path.join(out_dir, "input.bin"), x)

    patch_h, patch_w = H // 14, W // 14
    with torch.no_grad():
        feats = model.pretrained.get_intermediate_layers(
            x, model.intermediate_layer_idx[args.encoder], return_class_token=True)
        head = model.depth_head

        out = []
        for i, (px, cls) in enumerate(feats):
            dump(os.path.join(out_dir, f"feat_{i}.bin"), px)
            y = px.permute(0, 2, 1).reshape((px.shape[0], px.shape[-1], patch_h, patch_w))
            y = head.projects[i](y)
            y = head.resize_layers[i](y)
            out.append(y)
            dump(os.path.join(out_dir, f"layer_{i + 1}.bin"), y)

        layer_1, layer_2, layer_3, layer_4 = out
        layer_1_rn = head.scratch.layer1_rn(layer_1)
        layer_2_rn = head.scratch.layer2_rn(layer_2)
        layer_3_rn = head.scratch.layer3_rn(layer_3)
        layer_4_rn = head.scratch.layer4_rn(layer_4)
        for i, t in enumerate([layer_1_rn, layer_2_rn, layer_3_rn, layer_4_rn]):
            dump(os.path.join(out_dir, f"layer_{i + 1}_rn.bin"), t)

        path_4 = head.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = head.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = head.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = head.scratch.refinenet1(path_2, layer_1_rn)
        dump(os.path.join(out_dir, "path_4.bin"), path_4)
        dump(os.path.join(out_dir, "path_3.bin"), path_3)
        dump(os.path.join(out_dir, "path_2.bin"), path_2)
        dump(os.path.join(out_dir, "path_1.bin"), path_1)

        o = head.scratch.output_conv1(path_1)
        o = torch.nn.functional.interpolate(o, (patch_h * 14, patch_w * 14), mode="bilinear", align_corners=True)
        o = head.scratch.output_conv2(o)
        depth = torch.nn.functional.relu(o)
        dump(os.path.join(out_dir, "depth.bin"), depth)

        # Cross-check: the full forward must equal the stepwise result.
        full = model(x)
        assert torch.allclose(full, depth.squeeze(1), atol=1e-5), "stepwise != full forward"

    # Preprocessor reference: official transform (cv2 INTER_CUBIC lower-bound-518 resize + normalize)
    # on a synthetic uint8 image.
    from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
    from torchvision.transforms import Compose
    import cv2
    src_h, src_w = 300, 451
    raw = rng.randint(0, 256, size=(src_h, src_w, 3), dtype=np.uint8)  # RGB HWC
    raw.tofile(os.path.join(out_dir, "preprocess_input.bin"))
    transform = Compose([
        Resize(width=518, height=518, resize_target=False, keep_aspect_ratio=True,
               ensure_multiple_of=14, resize_method="lower_bound",
               image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    pre = transform({"image": raw.astype(np.float64) / 255.0})["image"].astype(np.float32)
    pre.tofile(os.path.join(out_dir, "preprocess_ref.bin"))

    meta = {
        "encoder": args.encoder,
        "H": H, "W": W,
        "patch_h": patch_h, "patch_w": patch_w,
        "features": MODEL_CONFIGS[args.encoder]["features"],
        "out_channels": MODEL_CONFIGS[args.encoder]["out_channels"],
        "layer_indices": model.intermediate_layer_idx[args.encoder],
        "depth_mean": float(depth.mean()), "depth_min": float(depth.min()), "depth_max": float(depth.max()),
        "pre_src_h": src_h, "pre_src_w": src_w,
        "pre_out_h": pre.shape[1], "pre_out_w": pre.shape[2],
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {out_dir}: depth mean={meta['depth_mean']:.4f} min={meta['depth_min']:.4f} max={meta['depth_max']:.4f}")


if __name__ == "__main__":
    main()

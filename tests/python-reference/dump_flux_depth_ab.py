#!/usr/bin/env python3
"""Flux-Depth edge-fringe A/B reference: official Depth-Anything-V2 on a REAL image.

Runs the official DA-V2 (repo code at DEPTH_ANYTHING_V2_REPO + local .pth) on the input
image exactly like `infer_image` does, and additionally emulates the two downstream flows
the engine is compared against:
  - BFL FLUX.1-Depth (`DepthImageEncoder`): raw depth -> bicubic antialias upsample to gen
    res (no per-image normalization; /127.5-1 there, we keep raw).
  - ComfyUI comfyui_controlnet_aux DepthAnythingV2 node: input resized (INTER_AREA down /
    INTER_CUBIC up + pad to 64-multiple) to detect_resolution, infer_image, min-max ->
    uint8, remove pad; then upscaled to gen res like Comfy's ImageScale.

Dumps to --out (all F32 unless noted):
  net_input.bin      [1,3,netH,netW]   official transform output (compare vs engine Preprocess)
  net_depth.bin      [1,1,netH,netW]   raw model output at network res
  src_depth.bin      [srcH,srcW]       infer_image result (bilinear align_corners=True to src res), raw scale
  gen_bfl.bin        [genH,genW]       raw depth -> bicubic antialias to gen res, /max (engine flux scaling convention)
  gen_bilinear.bin   [genH,genW]       raw depth -> bilinear align_corners=True to gen res, /max (engine's resize kernel)
  comfy_map.bin      [genH,genW]       comfy-emulated uint8 pipeline (detect 512) upscaled to gen res, /255
  meta.json
Plus PNG previews of each map.

Usage:
  python dump_flux_depth_ab.py --image bus.png --ckpt depth_anything_v2_vitl.pth --encoder vitl \
      --gen-w 1024 --gen-h 1024 --out <dir>
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

REPO = os.environ.get("DEPTH_ANYTHING_V2_REPO", os.path.expanduser("~/.cache/depth_anything_v2_repo"))
sys.path.insert(0, REPO)

from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet  # noqa: E402
from torchvision.transforms import Compose  # noqa: E402

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def save_png(path, unit_map):
    cv2.imwrite(path, (np.clip(unit_map, 0, 1) * 255.0).round().astype(np.uint8))


def official_transform(rgb_float01, input_size=518):
    transform = Compose([
        Resize(width=input_size, height=input_size, resize_target=False, keep_aspect_ratio=True,
               ensure_multiple_of=14, resize_method="lower_bound", image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    return transform({"image": rgb_float01})["image"]


def resize_image_with_pad(img, resolution):
    """comfyui_controlnet_aux util: INTER_AREA down / INTER_CUBIC up to shorter-side=resolution, pad to 64-multiple."""
    H_raw, W_raw, _ = img.shape
    k = float(resolution) / float(min(H_raw, W_raw))
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    interp = cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA
    img = cv2.resize(img, (W_target, H_target), interpolation=interp)
    H_pad, W_pad = (-H_target) % 64, (-W_target) % 64
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode="edge")

    def remove_pad(x):
        return x[:H_target, :W_target, ...]

    return img_padded, remove_pad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--encoder", default="vitl", choices=list(MODEL_CONFIGS))
    ap.add_argument("--gen-w", type=int, default=1024)
    ap.add_argument("--gen-h", type=int, default=1024)
    ap.add_argument("--detect-res", type=int, default=512)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()

    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    src_h, src_w = bgr.shape[:2]
    rgb01 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # ── Official path at full source res ──────────────────────────────────────
    net_in = official_transform(rgb01)                       # [3,netH,netW]
    x = torch.from_numpy(net_in).unsqueeze(0)
    net_in.astype(np.float32).tofile(os.path.join(args.out, "net_input.bin"))
    with torch.no_grad():
        net_depth = model(x)                                 # [1,netH,netW]
    net_depth.unsqueeze(1).numpy().astype(np.float32).tofile(os.path.join(args.out, "net_depth.bin"))

    d = net_depth.unsqueeze(1)                               # [1,1,h,w]
    src_depth = torch.nn.functional.interpolate(d, (src_h, src_w), mode="bilinear", align_corners=True)[0, 0].numpy()
    src_depth.astype(np.float32).tofile(os.path.join(args.out, "src_depth.bin"))

    gen_hw = (args.gen_h, args.gen_w)
    gen_bfl = torch.nn.functional.interpolate(d, gen_hw, mode="bicubic", antialias=True)[0, 0].numpy()
    gen_bfl = gen_bfl / gen_bfl.max()
    gen_bfl.astype(np.float32).tofile(os.path.join(args.out, "gen_bfl.bin"))

    gen_bil = torch.nn.functional.interpolate(d, gen_hw, mode="bilinear", align_corners=True)[0, 0].numpy()
    gen_bil = gen_bil / gen_bil.max()
    gen_bil.astype(np.float32).tofile(os.path.join(args.out, "gen_bilinear.bin"))

    # ── Comfy aux-node emulation at detect_resolution ────────────────────────
    rgb8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    padded, remove_pad = resize_image_with_pad(rgb8, args.detect_res)
    pad_in = official_transform(padded.astype(np.float32) / 255.0)
    with torch.no_grad():
        pad_depth = model(torch.from_numpy(pad_in).unsqueeze(0))
    pd = torch.nn.functional.interpolate(pad_depth.unsqueeze(1), padded.shape[:2],
                                         mode="bilinear", align_corners=True)[0, 0].numpy()
    pd = (pd - pd.min()) / (pd.max() - pd.min()) * 255.0
    pd = remove_pad(pd.astype(np.uint8)).astype(np.float32) / 255.0
    comfy_map = cv2.resize(pd, (args.gen_w, args.gen_h), interpolation=cv2.INTER_LINEAR)
    comfy_map.astype(np.float32).tofile(os.path.join(args.out, "comfy_map.bin"))

    save_png(os.path.join(args.out, "src_depth.png"), src_depth / src_depth.max())
    save_png(os.path.join(args.out, "gen_bfl.png"), gen_bfl)
    save_png(os.path.join(args.out, "gen_bilinear.png"), gen_bil)
    save_png(os.path.join(args.out, "comfy_map.png"), comfy_map)

    meta = {
        "image": os.path.abspath(args.image), "src_w": src_w, "src_h": src_h,
        "net_h": int(x.shape[2]), "net_w": int(x.shape[3]),
        "gen_w": args.gen_w, "gen_h": args.gen_h, "detect_res": args.detect_res,
        "net_depth_min": float(net_depth.min()), "net_depth_max": float(net_depth.max()),
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

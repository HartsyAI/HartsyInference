#!/usr/bin/env python3
"""Dump ControlNetHED (softedge_hed / scribble_hed) reference tensors from controlnet_aux.

Runs the official ControlNetHED_Apache2 network (pip install controlnet_aux) on a fixed
deterministic image and dumps the side projections plus every post-processed form, for the C#
parity test (HedParityTests) and diff script (diff_hed.py).

Usage:
    python dump_hed.py --ckpt /path/ControlNetHED.pth

Outputs to tests/python-reference/hed_reference_tensors/:
    image.bin           HxWx3 uint8 source image
    input.bin           [1,3,H,W] F32 raw [0,255] network input
    proj_{1..5}.bin     the 5 side logits at their native scales
    edge.bin            fused soft edge [H,W] F32 in [0,1] (resize + mean + sigmoid)
    softedge_u8.bin     HEDdetector e2e output (uint8 grayscale, one channel)
    safe_u8.bin         e2e output with safe=True
    scribble_u8.bin     e2e output with scribble=True (binary 0/255)
    meta.json
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch
from einops import rearrange

from controlnet_aux.hed import ControlNetHED_Apache2, HEDdetector

# Multiples of 64 so HEDdetector's resize_image calls are identity and the e2e outputs
# align with the raw network forward.
H, W = 320, 256


def make_image():
    rng = np.random.RandomState(42)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.zeros((H, W, 3), np.float32)
    img[..., 0] = 127 + 90 * np.sin(xx / 17.0) * np.cos(yy / 23.0)
    img[..., 1] = 127 + 110 * np.cos(xx / 29.0 + yy / 13.0)
    img[..., 2] = 127 + 80 * np.sin((xx + yy) / 19.0)
    cv2.rectangle(img, (40, 60), (180, 200), (230, 40, 40), -1)
    cv2.circle(img, (140, 240), 55, (30, 220, 120), -1)
    cv2.line(img, (10, 300), (240, 30), (250, 250, 30), 5)
    img += rng.randn(H, W, 3) * 6.0
    return img.clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "hed_reference_tensors")
    os.makedirs(out_dir, exist_ok=True)

    net = ControlNetHED_Apache2()
    net.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    net.float().eval()

    img = make_image()
    img.tofile(os.path.join(out_dir, "image.bin"))

    with torch.no_grad():
        x = rearrange(torch.from_numpy(img.copy()).float(), "h w c -> 1 c h w")
        x.numpy().astype(np.float32).tofile(os.path.join(out_dir, "input.bin"))
        projs = net(x)
        for i, p in enumerate(projs):
            p.numpy().astype(np.float32).tofile(os.path.join(out_dir, f"proj_{i + 1}.bin"))
        edges = [e.numpy().astype(np.float32)[0, 0] for e in projs]
        edges = [cv2.resize(e, (W, H), interpolation=cv2.INTER_LINEAR) for e in edges]
        edge = 1 / (1 + np.exp(-np.mean(np.stack(edges, axis=2), axis=2).astype(np.float64)))
        edge.astype(np.float32).tofile(os.path.join(out_dir, "edge.bin"))

    det = HEDdetector(net)
    res = min(H, W)
    soft = det(img, detect_resolution=res, image_resolution=res, output_type="np")
    safe = det(img, detect_resolution=res, image_resolution=res, safe=True, output_type="np")
    scrib = det(img, detect_resolution=res, image_resolution=res, scribble=True, output_type="np")
    assert soft.shape == (H, W, 3), soft.shape
    soft[..., 0].tofile(os.path.join(out_dir, "softedge_u8.bin"))
    safe[..., 0].tofile(os.path.join(out_dir, "safe_u8.bin"))
    scrib[..., 0].tofile(os.path.join(out_dir, "scribble_u8.bin"))

    meta = {
        "H": H, "W": W,
        "proj_shapes": [list(p.shape) for p in projs],
        "edge_mean": float(edge.mean()),
        "softedge_mean": float(soft[..., 0].mean()),
        "scribble_white_frac": float((scrib[..., 0] == 255).mean()),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {out_dir}: edge mean={meta['edge_mean']:.4f} scribble white={meta['scribble_white_frac']:.4f}")


if __name__ == "__main__":
    main()

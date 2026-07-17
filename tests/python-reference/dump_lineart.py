#!/usr/bin/env python3
"""Dump lineart Generator reference tensors from controlnet_aux.

Runs the official lineart Generator (pip install controlnet_aux) on a fixed deterministic image
and dumps the stage intermediates plus the detector's e2e inverted conditioning map, for the C#
parity test (LineartParityTests) and diff script (diff_lineart.py).

Usage:
    python dump_lineart.py --variant realistic --ckpt /path/sk_model.pth
    python dump_lineart.py --variant coarse   --ckpt /path/sk_model2.pth

Outputs to tests/python-reference/lineart_reference_tensors/<variant>/:
    image.bin      HxWx3 uint8 source image
    input.bin      [1,3,H,W] F32 network input (/255)
    m0.bin .. m3.bin   stage outputs (stem / downsample / residual / upsample)
    line.bin       raw sigmoid output [1,1,H,W] F32 (white bg, dark lines)
    cond_u8.bin    LineartDetector e2e output (uint8, 255 - quantized line; white lines on black)
    meta.json
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch
from einops import rearrange

from controlnet_aux.lineart import Generator, LineartDetector

H, W = 320, 256


def make_image():
    rng = np.random.RandomState(7)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.zeros((H, W, 3), np.float32)
    img[..., 0] = 150 + 70 * np.sin(xx / 31.0) * np.sin(yy / 11.0)
    img[..., 1] = 127 + 100 * np.cos(xx / 23.0 - yy / 17.0)
    img[..., 2] = 110 + 90 * np.sin((2 * xx - yy) / 27.0)
    cv2.ellipse(img, (128, 120), (80, 50), 30, 0, 360, (240, 240, 240), -1)
    cv2.ellipse(img, (128, 120), (80, 50), 30, 0, 360, (20, 20, 20), 3)
    cv2.rectangle(img, (30, 200), (220, 290), (60, 60, 200), -1)
    cv2.line(img, (0, 40), (255, 100), (10, 10, 10), 2)
    img += rng.randn(H, W, 3) * 4.0
    return img.clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="realistic", choices=["realistic", "coarse"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "lineart_reference_tensors", args.variant)
    os.makedirs(out_dir, exist_ok=True)

    model = Generator(3, 1, 3)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()

    img = make_image()
    img.tofile(os.path.join(out_dir, "image.bin"))

    with torch.no_grad():
        x = rearrange(torch.from_numpy(img.copy()).float() / 255.0, "h w c -> 1 c h w")
        x.numpy().astype(np.float32).tofile(os.path.join(out_dir, "input.bin"))
        m0 = model.model0(x)
        m1 = model.model1(m0)
        m2 = model.model2(m1)
        m3 = model.model3(m2)
        line = model.model4(m3)
        for name, t in [("m0", m0), ("m1", m1), ("m2", m2), ("m3", m3), ("line", line)]:
            t.numpy().astype(np.float32).tofile(os.path.join(out_dir, name + ".bin"))
        full = model(x)
        assert torch.allclose(full, line), "stepwise != full forward"

    # e2e detector (identity resizes at multiples of 64): quantize + invert.
    det = LineartDetector(model, model)
    res = min(H, W)
    cond = det(img, coarse=False, detect_resolution=res, image_resolution=res, output_type="np")
    assert cond.shape == (H, W, 3), cond.shape
    cond[..., 0].tofile(os.path.join(out_dir, "cond_u8.bin"))

    meta = {
        "variant": args.variant, "H": H, "W": W,
        "line_mean": float(line.mean()), "cond_mean": float(cond[..., 0].mean()),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {out_dir}: line mean={meta['line_mean']:.4f} cond mean={meta['cond_mean']:.2f}")


if __name__ == "__main__":
    main()

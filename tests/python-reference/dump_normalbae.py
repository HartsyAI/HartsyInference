#!/usr/bin/env python3
"""Dump NormalBAE (surface-normal annotator) reference tensors from controlnet_aux.

Runs the official NNET (EfficientNet-B5 encoder + normal decoder, pip install controlnet_aux)
on a fixed deterministic image and dumps encoder taps, decoder stages, and the e2e conditioning
map, for the C# parity test (NormalBaeParityTests) and diff script (diff_normalbae.py).

Usage:
    python dump_normalbae.py --ckpt /path/scannet.pt

Outputs to tests/python-reference/normalbae_reference_tensors/:
    image.bin        HxWx3 uint8 source image
    input.bin        [1,3,H,W] F32 network input (/255 + ImageNet normalize)
    feat_{0..4}.bin  encoder taps (stage0 / stage1 / stage2 / stage4 / conv_head)
    xd_{0..4}.bin    decoder feature pyramid
    out_res{8,4,2,1}.bin   normalized [1,4,·,·] outputs per scale
    cond_u8.bin      e2e detector output HxWx3 uint8 ((normal+1)/2 as RGB)
    meta.json
"""
import argparse
import json
import os
import types

import numpy as np
import torch
from einops import rearrange

from controlnet_aux.normalbae import NormalBaeDetector, load_checkpoint
from controlnet_aux.normalbae.nets.NNET import NNET

H, W = 320, 256


def make_image():
    import cv2
    rng = np.random.RandomState(3)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.zeros((H, W, 3), np.float32)
    img[..., 0] = 127 + 80 * np.sin(xx / 41.0 + yy / 13.0)
    img[..., 1] = 127 + 90 * np.cos(xx / 19.0)
    img[..., 2] = 127 + 70 * np.sin(yy / 29.0)
    cv2.rectangle(img, (50, 50), (200, 180), (220, 200, 60), -1)
    cv2.circle(img, (100, 240), 60, (40, 60, 220), -1)
    img += rng.randn(H, W, 3) * 5.0
    return img.clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "normalbae_reference_tensors")
    os.makedirs(out_dir, exist_ok=True)

    margs = types.SimpleNamespace()
    margs.mode = "client"
    margs.architecture = "BN"
    margs.pretrained = "scannet"
    margs.sampling_ratio = 0.4
    margs.importance_ratio = 0.7
    model = NNET(margs)
    model = load_checkpoint(args.ckpt, model)
    model.eval()

    img = make_image()
    img.tofile(os.path.join(out_dir, "image.bin"))

    mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)

    with torch.no_grad():
        x = rearrange(torch.from_numpy(img.copy()).float() / 255.0, "h w c -> 1 c h w")
        x = (x - torch.from_numpy(mean)) / torch.from_numpy(std)
        x.numpy().astype(np.float32).tofile(os.path.join(out_dir, "input.bin"))

        features = model.encoder(x)
        for name, idx in [("feat_0", 4), ("feat_1", 5), ("feat_2", 6), ("feat_3", 8), ("feat_4", 11)]:
            features[idx].numpy().astype(np.float32).tofile(os.path.join(out_dir, name + ".bin"))

        dec = model.decoder
        xb0, xb1, xb2, xb3, xb4 = features[4], features[5], features[6], features[8], features[11]
        x_d0 = dec.conv2(xb4)
        x_d1 = dec.up1(x_d0, xb3)
        x_d2 = dec.up2(x_d1, xb2)
        x_d3 = dec.up3(x_d2, xb1)
        x_d4 = dec.up4(x_d3, xb0)
        for name, t in [("xd_0", x_d0), ("xd_1", x_d1), ("xd_2", x_d2), ("xd_3", x_d3), ("xd_4", x_d4)]:
            t.numpy().astype(np.float32).tofile(os.path.join(out_dir, name + ".bin"))

        outs, _, _ = dec(features, mode="test")
        for name, t in zip(["out_res8", "out_res4", "out_res2", "out_res1"], outs):
            t.numpy().astype(np.float32).tofile(os.path.join(out_dir, name + ".bin"))

    det = NormalBaeDetector(model)
    res = min(H, W)
    cond = det(img, detect_resolution=res, image_resolution=res, output_type="np")
    assert cond.shape == (H, W, 3), cond.shape
    cond.tofile(os.path.join(out_dir, "cond_u8.bin"))

    meta = {
        "H": H, "W": W,
        "feat_shapes": {n: list(features[i].shape) for n, i in
                        [("feat_0", 4), ("feat_1", 5), ("feat_2", 6), ("feat_3", 8), ("feat_4", 11)]},
        "out_res1_mean": float(outs[-1].mean()),
        "cond_mean": float(cond.mean()),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {out_dir}: out_res1 mean={meta['out_res1_mean']:.4f} cond mean={meta['cond_mean']:.2f}")


if __name__ == "__main__":
    main()

"""Real-ESRGAN reference: a float64 numpy transcription of BasicSR's RRDBNet plus RealESRGANer's pre/post-processing.

Transcribed from BasicSR basicsr/archs/rrdbnet_arch.py (ResidualDenseBlock, RRDB, RRDBNet), basicsr/archs/arch_util.py
(pixel_unshuffle) and Real-ESRGAN realesrgan/utils.py (RealESRGANer.pre_process / post_process / enhance) on their master
branches, 2026-11-03. Matches the official CLI defaults (inference_realesrgan.py): pre_pad 0, no tiling; the mod pad for a
2x model is F.pad(..., 'reflect') on the right and bottom. Precision is float64 throughout; the CLI's fp16 default is not
modelled. Needs only numpy (no torch, no pickle): checkpoints are read as safetensors.

Usage:
  python -I real_esrgan_reference.py fixture <out_dir>
      Seeded tiny RRDBNets (x4 and x2) with their weights as safetensors, inputs and expected outputs. Small enough to
      commit; RealEsrganParityTests reads them from TestData/RealEsrgan.
  python -I real_esrgan_reference.py real <checkpoint.safetensors> <out_dir> [--revision REV]
      A fixed synthetic 16x12 crop through a real checkpoint (scale inferred from conv_first, as the engine does).

Each case writes <case>.input.rgb (uint8 HWC), <case>.net_input.f32 (the network's [1,C,h,w] input after the mod pad and
unshuffle), <case>.net_output.f32 ([1,3,4h,4w]), <case>.output.rgb (uint8 HWC, cropped, clamped, rounded) and a
<case>.json with the dimensions, scale and checkpoint sha256. All .f32 files are little-endian float32.
"""
import argparse
import hashlib
import json
import os
import struct
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# ── safetensors I/O (header JSON + raw little-endian data) ─────────────────────────────────────────────

def read_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data = f.read()
    header.pop("__metadata__", None)
    out = {}
    for name, info in header.items():
        if info["dtype"] != "F32":
            raise ValueError(f"{name}: only F32 checkpoints are supported here, got {info['dtype']}")
        start, end = info["data_offsets"]
        out[name] = np.frombuffer(data[start:end], dtype="<f4").reshape(info["shape"])
    return out


def write_safetensors(path, tensors):
    header, blobs, offset = {}, [], 0
    for name in sorted(tensors):
        arr = np.ascontiguousarray(tensors[name], dtype="<f4")
        raw = arr.tobytes()
        header[name] = {"dtype": "F32", "shape": list(arr.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    blob += b" " * ((8 - len(blob) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for raw in blobs:
            f.write(raw)


# ── RRDBNet (BasicSR) ──────────────────────────────────────────────────────────────────────────────────

def conv3x3(x, w, b):
    """nn.Conv2d(k=3, stride 1, padding 1) on one image: x [C,H,W], w [O,C,3,3], b [O] -> [O,H,W]."""
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    windows = sliding_window_view(xp, (3, 3), axis=(1, 2))  # [C,H,W,3,3]
    return np.tensordot(w, windows, axes=([1, 2, 3], [0, 3, 4])) + b[:, None, None]


def lrelu(x):
    return np.where(x > 0, x, 0.2 * x)


def up2(x):
    """F.interpolate(scale_factor=2, mode='nearest')."""
    return x.repeat(2, axis=1).repeat(2, axis=2)


def pixel_unshuffle(x, scale):
    """arch_util.pixel_unshuffle on one image: view(c, h, s, w, s).permute(c, s_h, s_w, h, w)."""
    c, hh, hw = x.shape
    h, w = hh // scale, hw // scale
    return x.reshape(c, h, scale, w, scale).transpose(0, 2, 4, 1, 3).reshape(c * scale * scale, h, w)


class RrdbNet:
    def __init__(self, weights):
        self.w = {k: v.astype(np.float64) for k, v in weights.items()}
        self.scale = 2 if self.w["conv_first.weight"].shape[1] == 12 else 4
        self.num_block = 1 + max(int(k.split(".")[1]) for k in self.w if k.startswith("body."))

    def conv(self, name, x):
        return conv3x3(x, self.w[f"{name}.weight"], self.w[f"{name}.bias"])

    def rdb(self, p, x):
        x1 = lrelu(self.conv(f"{p}.conv1", x))
        x2 = lrelu(self.conv(f"{p}.conv2", np.concatenate((x, x1), 0)))
        x3 = lrelu(self.conv(f"{p}.conv3", np.concatenate((x, x1, x2), 0)))
        x4 = lrelu(self.conv(f"{p}.conv4", np.concatenate((x, x1, x2, x3), 0)))
        x5 = self.conv(f"{p}.conv5", np.concatenate((x, x1, x2, x3, x4), 0))
        return x5 * 0.2 + x

    def rrdb(self, p, x):
        out = self.rdb(f"{p}.rdb1", x)
        out = self.rdb(f"{p}.rdb2", out)
        out = self.rdb(f"{p}.rdb3", out)
        return out * 0.2 + x

    def forward(self, feat_in):
        """feat_in is the already-unshuffled input [C,h,w]; returns [3,4h,4w]."""
        feat = self.conv("conv_first", feat_in)
        body = feat
        for i in range(self.num_block):
            body = self.rrdb(f"body.{i}", body)
        feat = feat + self.conv("conv_body", body)
        feat = lrelu(self.conv("conv_up1", up2(feat)))
        feat = lrelu(self.conv("conv_up2", up2(feat)))
        return self.conv("conv_last", lrelu(self.conv("conv_hr", feat)))


# ── RealESRGANer pipeline (pre_pad 0, no tiling) ──────────────────────────────────────────────────────

def net_input(rgb_hwc, scale):
    """uint8 HWC -> the network's [C,h,w] input: /255, mod pad (reflect) to a multiple of the unshuffle, unshuffle."""
    img = rgb_hwc.astype(np.float64).transpose(2, 0, 1) / 255.0
    r = 2 if scale == 2 else 1
    if r > 1:
        _, h, w = img.shape
        pad_h, pad_w = (r - h % r) % r, (r - w % r) % r
        img = np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
        img = pixel_unshuffle(img, r)
    return img


def to_bytes(out_chw, width, height, scale):
    """Crop the mod pad off, clamp to [0,1], *255, round half to even (torch.round / np.round) -> uint8 HWC."""
    out = np.clip(out_chw[:, : height * scale, : width * scale], 0.0, 1.0)
    return np.round(out * 255.0).astype(np.uint8).transpose(1, 2, 0)


def run_case(out_dir, case, weights, rgb, extra):
    net = RrdbNet(weights)
    h, w, _ = rgb.shape
    x = net_input(rgb, net.scale)
    y = net.forward(x)
    rgb_out = to_bytes(y, w, h, net.scale)
    base = os.path.join(out_dir, case)
    rgb.tofile(base + ".input.rgb")
    x[None].astype("<f4").tofile(base + ".net_input.f32")
    y[None].astype("<f4").tofile(base + ".net_output.f32")
    rgb_out.tofile(base + ".output.rgb")
    meta = {
        "width": w, "height": h, "scale": net.scale, "num_block": net.num_block,
        "net_input_shape": [1, *x.shape], "net_output_shape": [1, *y.shape],
        "output_width": w * net.scale, "output_height": h * net.scale, **extra,
    }
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"{case}: in {w}x{h} -> out {w * net.scale}x{h * net.scale}, net {list(x.shape)} -> {list(y.shape)}, "
          f"out range [{y.min():.4f}, {y.max():.4f}]")


def tiny_weights(rng, scale, num_feat=8, num_grow=4, num_block=2):
    """Seeded RRDBNet weights at a tiny width. The 0.1 std keeps 2 blocks of residuals in a sane range."""
    w = {}

    def conv(name, cout, cin):
        w[f"{name}.weight"] = (rng.standard_normal((cout, cin, 3, 3)) * 0.1).astype(np.float32)
        w[f"{name}.bias"] = (rng.standard_normal(cout) * 0.05).astype(np.float32)

    conv("conv_first", num_feat, 12 if scale == 2 else 3)
    for i in range(num_block):
        for j in (1, 2, 3):
            p = f"body.{i}.rdb{j}"
            conv(f"{p}.conv1", num_grow, num_feat)
            conv(f"{p}.conv2", num_grow, num_feat + num_grow)
            conv(f"{p}.conv3", num_grow, num_feat + 2 * num_grow)
            conv(f"{p}.conv4", num_grow, num_feat + 3 * num_grow)
            conv(f"{p}.conv5", num_feat, num_feat + 4 * num_grow)
    for name in ("conv_body", "conv_up1", "conv_up2", "conv_hr"):
        conv(name, num_feat, num_feat)
    conv("conv_last", 3, num_feat)
    return w


def synthetic_rgb(rng, width, height):
    """A smooth gradient plus noise, so edges, flats and texture all reach the network."""
    yy, xx = np.mgrid[0:height, 0:width]
    base = np.stack([xx / max(width - 1, 1), yy / max(height - 1, 1), (xx + yy) / max(width + height - 2, 1)], -1)
    img = base * 200 + 25 + rng.normal(0, 18, (height, width, 3))
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def cmd_fixture(args):
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(20261103)
    # Odd and unequal sizes catch H/W swaps; the x2 case's odd width exercises the reflect mod pad.
    for case, scale, (w, h) in (("tiny_x4", 4, (7, 5)), ("tiny_x2", 2, (7, 6))):
        weights = tiny_weights(rng, scale)
        write_safetensors(os.path.join(args.out_dir, case + ".safetensors"), weights)
        run_case(args.out_dir, case, weights, synthetic_rgb(rng, w, h), {"seed": 20261103})


def cmd_real(args):
    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.checkpoint, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    weights = read_safetensors(args.checkpoint)
    rgb = synthetic_rgb(np.random.default_rng(7), 16, 12)
    run_case(args.out_dir, "real_x4plus", weights, rgb,
             {"checkpoint": os.path.basename(args.checkpoint), "checkpoint_sha256": sha, "revision": args.revision})


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fixture")
    p.add_argument("out_dir")
    p.set_defaults(func=cmd_fixture)
    p = sub.add_parser("real")
    p.add_argument("checkpoint")
    p.add_argument("out_dir")
    p.add_argument("--revision", default="")
    p.set_defaults(func=cmd_real)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

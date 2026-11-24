"""Reference dump for the Depth Pro 2 head and focal estimator (phase A2), fp32 on CPU.

Writes every stage the C# parity test (tests/HartsyInference.Vision.Tests/DepthPro2/Parity/DepthPro2ParityTests.cs) compares:
saved inputs, RoPE tables, embeddings, every block output, the four taps, the reassembled pyramid, every fusion output,
the decoder output, inverse depth, metric depth, and the focal estimator's resized input, taps, head stages, logits,
field of view and focal length. Each tensor is raw little-endian float32 with its shape in meta.json.

Reference: apple-aiml-research/ml-depth-pro-2 at the pinned commit below (fetched into --src), timm 1.0.27, torch >= 2.7.
Runs in the hartsy-ref:cpu Docker image, never on the Windows host:

  run.sh -v "<worktree>:/wt" -v "<models root>:/models:ro" -- \
      python /wt/tests/python-reference/dump_depth_pro2.py --out /wt/tests/python-reference/depth_pro2_reference_tensors
"""

from __future__ import annotations

import argparse
import json
import sys
import types
import urllib.request
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

COMMIT = "aa389cbb0b74341664ed663a0966c859905cd6c0"
RAW = f"https://raw.githubusercontent.com/apple-aiml-research/ml-depth-pro-2/{COMMIT}/src/depth_pro_2/"
SOURCES = ["__init__.py", "model.py", "utils.py", "streaming.py", "network/__init__.py", "network/decoder.py",
           "network/encoder.py", "network/fov.py"]

# (name, height, width, depth?) - 256x384 needs no pad; 250x370 is replicate-padded to 256x384 and resized to 224x320 by the
# estimator (the antialiased path); 256x512 is wider than 16:9, so the estimator crops it and converts the angles back.
CASES = [("a", 256, 384, True), ("b", 250, 370, True), ("c", 256, 512, False)]


def fetch_sources(root: Path) -> None:
    for name in SOURCES:
        target = root / "depth_pro_2" / name
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(RAW + name) as response:
            target.write_bytes(response.read())


def synthetic_rgb(height: int, width: int, seed: int) -> np.ndarray:
    """A smooth scene-like gradient with blobs and mild noise, uint8 (H, W, 3); saved, so no RNG has to match."""
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(0, 1, height), np.linspace(0, 1, width), indexing="ij")
    img = np.stack([0.3 + 0.5 * yy, 0.6 - 0.3 * xx + 0.2 * yy, 0.4 + 0.3 * np.sin(6 * xx) * np.cos(4 * yy)], axis=-1)
    for _ in range(6):
        cy, cx, r = rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0.05, 0.25)
        blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r))
        img += blob[..., None] * rng.uniform(-0.4, 0.4, size=3)
    img += rng.normal(0, 0.02, size=img.shape)
    return (np.clip(img, 0, 1) * 255).round().astype(np.uint8)


class Dump:
    def __init__(self, out: Path):
        self.out = out
        self.meta: dict[str, list[int]] = {}

    def __call__(self, name: str, t: torch.Tensor) -> None:
        a = t.detach().to(torch.float32).contiguous().cpu().numpy()
        a.astype("<f4").tofile(self.out / f"{name}.bin")
        self.meta[name] = list(a.shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--src", default="/work/ml-depth-pro-2")
    parser.add_argument("--depth", default="/models/Vision/DepthPro2-S/model.safetensors")
    parser.add_argument("--focal", default="/models/Vision/DepthPro2-focal/model.safetensors")
    args = parser.parse_args()

    src = Path(args.src)
    fetch_sources(src)
    sys.path.insert(0, str(src))
    # utils.py imports video and HEIF readers for file I/O only; the network never touches them, so empty stand-ins suffice.
    for module in ("av", "pillow_heif"):
        if module not in sys.modules:
            try:
                __import__(module)
            except ImportError:
                stub = types.ModuleType(module)
                stub.register_heif_opener = lambda: None
                sys.modules[module] = stub
    from depth_pro_2.model import PRESETS, DepthPro2, to_normalized_tensor
    from depth_pro_2.network.fov import FOVEstimator, sincos_2d

    torch.set_grad_enabled(False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dump = Dump(out)

    cfg = PRESETS["depthpro2-s"]
    model = DepthPro2(cfg.backbone_preset, cfg.reassembled_blocks, cfg.cross_frame_blocks, cfg.dims_reassemble,
                      cfg.dim_decoder, fov=False, precision=torch.float32)
    model.load_state_dict(load_file(args.depth), strict=True)
    model.eval()
    fov = FOVEstimator(torch.float32)
    fov.load_state_dict(load_file(args.focal), strict=True)
    fov.eval()
    enc = model.encoder
    npt = enc.num_prefix_tokens

    for seed, (name, height, width, with_depth) in enumerate(CASES):
        rgb = synthetic_rgb(height, width, 20261124 + seed)
        rgb.tofile(out / f"{name}_rgb_u8.bin")
        dump.meta[f"{name}_rgb_u8"] = [height, width, 3]
        x = to_normalized_tensor(rgb, torch.device("cpu")).unsqueeze(0)
        dump(f"{name}_pixels", x)

        if with_depth:
            pad_h, pad_w = -height % 32, -width % 32
            xp = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate") if (pad_h or pad_w) else x
            dump(f"{name}_padded", xp)
            grid = (xp.shape[-2] // 16, xp.shape[-1] // 16)
            rope = enc.rope.get_embed(shape=grid)
            dump(f"{name}_rope", rope)

            # MultiViewEncoder.encode, unrolled so every block's output is kept (T = 1: cross-frame blocks restore the prefix).
            tokens = enc.embed(xp)
            dump(f"{name}_embed", tokens)
            features = []
            for idx, block in enumerate(enc.blocks):
                if idx in enc.cross_frame_blocks:
                    tokens = enc.cross_frame_attention(block, tokens, rope)
                else:
                    tokens = block(tokens, rope=rope)
                dump(f"{name}_block{idx}", tokens)
                if idx in enc.reassembled_blocks:
                    features.append(tokens[:, npt:])
            ref_features, ref_grid = enc.encode(xp)
            assert ref_grid == grid
            for mine, theirs in zip(features, ref_features):
                assert torch.equal(mine, theirs), "unrolled encoder differs from MultiViewEncoder.encode"

            levels = enc.reassemble_tokens(features, grid)
            for i, level in enumerate(levels):
                dump(f"{name}_reassemble{i}", level)
            dec = model.decoder
            convs = [conv(level) for conv, level in zip(dec.convs, levels)]
            for i, c in enumerate(convs):
                dump(f"{name}_conv{i}", c)
            feat = dec.fusions[-1](convs[-1])
            dump(f"{name}_fusion3", feat)
            for i in range(len(levels) - 2, -1, -1):
                feat = dec.fusions[i](feat, convs[i])
                dump(f"{name}_fusion{i}", feat)
            assert torch.equal(feat, dec(levels))
            dump(f"{name}_decoded", feat)
            head = model.head(feat.float())
            dump(f"{name}_head", head)

            pred = model.infer(x[0], f_px=None)
            inverse = pred["inverse_depth"]
            assert torch.equal(inverse, head[0, 0, :height, :width])
            dump(f"{name}_inverse_depth", inverse)

        # Focal estimator on the unpadded image.
        y = fov.resize(x)
        dump(f"{name}_fov_input", y)
        tokens, fgrid = fov.encode(y)
        dump(f"{name}_fov_taps", tokens)
        h = fov.head
        taps = tokens.float().split(h.embed_dim, dim=-1)
        proj = h.proj(torch.cat([norm(t) for norm, t in zip(h.tap_norms, taps)], dim=-1))
        dump(f"{name}_fov_proj", proj)
        prefix, patches = proj[:, : h.num_prefix], proj[:, h.num_prefix:]
        grid_map = patches.transpose(1, 2).reshape(1, h.dim, *fgrid)
        pooled = F.avg_pool2d(grid_map, h.pool_stride, ceil_mode=True)
        dump(f"{name}_fov_pooled", pooled)
        code = sincos_2d(*pooled.shape[-2:], h.dim, pooled.device)
        dump(f"{name}_fov_sincos", code)
        seq = torch.cat([h.query.expand(1, -1, -1), prefix + h.prefix_embed, pooled.flatten(2).transpose(1, 2) + code], dim=1)
        dump(f"{name}_fov_seq", seq)
        for i, block in enumerate(h.blocks):
            seq = block(seq)
            dump(f"{name}_fov_block{i}", seq)
        logits = h.out(h.norm(seq[:, 0]))
        assert torch.equal(logits, h(tokens, fgrid))
        dump(f"{name}_fov_logits", logits)
        fov_deg = fov(x)
        dump(f"{name}_fov_deg", fov_deg)
        f_px = fov.focal_length_px(x)
        dump(f"{name}_f_px", f_px.reshape(1))
        if with_depth:
            depth = model.infer(x[0], f_px=f_px)["depth"]
            dump(f"{name}_depth", torch.where(torch.isinf(depth), torch.zeros_like(depth), depth))

    dump.meta["_versions"] = {"torch": torch.__version__, "timm": timm.__version__, "ml_depth_pro_2": COMMIT}
    (out / "meta.json").write_text(json.dumps(dump.meta, indent=1))
    print("wrote", len(dump.meta) - 1, "tensors to", out)


if __name__ == "__main__":
    main()

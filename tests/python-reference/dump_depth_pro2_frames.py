"""Reference dump for Depth Pro 2 cross-frame attention and video (phase A4), fp32 on CPU.

Writes what tests/HartsyInference.Vision.Tests/DepthPro2/Parity/DepthPro2FramesParityTests.cs compares, each tensor raw little-endian
float32 with its shape in meta.json:

- joint:     a T = 3 clip through ``infer`` (one joint forward), with every block's output of the unrolled encoder, the inverse depth,
             the estimated focal length and the metric depth.
- buffered:  ``infer_video`` with window 4 and overlap 2 on 6 frames (two clips, one cross-fade), on 7 frames (the flush pulls the last
             clip back over frames already returned) and on 3 frames (shorter than one window).
- unbuffered: ``infer_video(mode="unbuffered")`` with window 16 on 6 frames (partial caches only) and window 3 on 6 frames (the ring
             wraps and overwrites its oldest slot).

Frames are sliding crops of one saved synthetic image, so consecutive frames overlap the way a panning video's do. The 100 x 150 frames are
replicate-padded to 128 x 160. Reference: apple-aiml-research/ml-depth-pro-2 at the commit pinned in dump_depth_pro2.py, fetched into --src.
Runs in the hartsy-ref:cpu Docker image, never on the Windows host:

  run.sh -v "<worktree>:/wt" -v "<models root>:/models:ro" -- \
      python /wt/tests/python-reference/dump_depth_pro2_frames.py --out /wt/tests/python-reference/depth_pro2_frames_reference_tensors
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import timm
import torch
from safetensors.torch import load_file

from dump_depth_pro2 import COMMIT, Dump, fetch_sources, synthetic_rgb

HEIGHT, WIDTH, PAN = 100, 150, 9
CLIP = 7


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
    for module in ("av", "pillow_heif"):
        if module not in sys.modules:
            try:
                __import__(module)
            except ImportError:
                stub = types.ModuleType(module)
                stub.register_heif_opener = lambda: None
                sys.modules[module] = stub
    from depth_pro_2.model import PRESETS, DepthPro2, to_normalized_tensor
    from depth_pro_2.network.fov import FOVEstimator

    torch.set_grad_enabled(False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dump = Dump(out)

    cfg = PRESETS["depthpro2-s"]
    model = DepthPro2(cfg.backbone_preset, cfg.reassembled_blocks, cfg.cross_frame_blocks, cfg.dims_reassemble,
                      cfg.dim_decoder, fov=True, precision=torch.float32)
    state = load_file(args.depth)
    state.update({f"fov.{k}": v for k, v in load_file(args.focal).items()})
    model.load_state_dict(state, strict=True)
    model.eval()
    assert isinstance(model.fov, FOVEstimator)
    enc = model.encoder

    wide = synthetic_rgb(HEIGHT, WIDTH + PAN * (CLIP - 1), 20261125)
    frames_u8 = np.stack([wide[:, i * PAN: i * PAN + WIDTH] for i in range(CLIP)])
    frames_u8.tofile(out / "frames_rgb_u8.bin")
    dump.meta["frames_rgb_u8"] = list(frames_u8.shape)
    frames = [to_normalized_tensor(f, torch.device("cpu")) for f in frames_u8]

    # Joint forward of the first three frames, unrolled so every block's output is kept.
    x = torch.stack(frames[:3])
    pad_h, pad_w = -HEIGHT % 32, -WIDTH % 32
    xp = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    grid = (xp.shape[-2] // 16, xp.shape[-1] // 16)
    rope = enc.rope.get_embed(shape=grid)
    tokens = enc.embed(xp)
    for idx, block in enumerate(enc.blocks):
        tokens = enc.cross_frame_attention(block, tokens, rope) if idx in enc.cross_frame_blocks else block(tokens, rope=rope)
        dump(f"joint_block{idx}", tokens)
    pred = model.infer(x)
    dump("joint_inverse_depth", pred["inverse_depth"])
    dump("joint_f_px", pred["focallength_px"].reshape(1))
    dump("joint_depth", torch.where(torch.isinf(pred["depth"]), torch.zeros_like(pred["depth"]), pred["depth"]))
    # Each frame alone, to show the joint forward is not per-frame independent.
    alone = torch.stack([model.infer(f)["inverse_depth"] for f in frames[:3]])
    dump.meta["_joint_vs_alone_relrms"] = float((pred["inverse_depth"] - alone).norm() / alone.norm())

    def video(name: str, count: int, window: int, overlap: int, mode: str) -> None:
        preds = list(model.infer_video(iter(frames[:count]), window=window, overlap=overlap, mode=mode))
        assert len(preds) == count
        dump(f"{name}_inverse_depth", torch.stack([p["inverse_depth"] for p in preds]))
        depth = torch.stack([p["depth"] for p in preds])
        dump(f"{name}_depth", torch.where(torch.isinf(depth), torch.zeros_like(depth), depth))
        dump(f"{name}_f_px", preds[0]["focallength_px"].reshape(1))

    video("buffered6", 6, 4, 2, "buffered")
    video("buffered7", 7, 4, 2, "buffered")
    video("buffered3", 3, 4, 2, "buffered")
    video("unbuffered16", 6, 16, 0, "unbuffered")
    video("unbuffered3", 6, 3, 0, "unbuffered")

    dump.meta["_versions"] = {"torch": torch.__version__, "timm": timm.__version__, "ml_depth_pro_2": COMMIT}
    (out / "meta.json").write_text(json.dumps(dump.meta, indent=1))
    print("wrote", len(dump.meta) - 2, "tensors to", out, "joint vs alone relRMS", dump.meta["_joint_vs_alone_relrms"])


if __name__ == "__main__":
    main()

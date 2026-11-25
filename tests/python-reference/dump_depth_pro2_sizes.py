"""Reference run of Depth Pro 2 B, L and H+ on saved real-photo inputs (phase A5), fp32 on CPU.

For one size it writes what the C# CLI is compared against: the metric depth, inverse depth and focal length of
- image:      ``infer`` on one PNG,
- views:      ``infer`` on a folder of PNGs of one size as one joint clip,
- buffered:   ``infer_video`` (window 4, overlap 2) on the frames folder,
- unbuffered: ``infer_video(mode="unbuffered")`` with a 3-frame ring on the same frames.
Everything is .npy (float32; infinite depth stays +inf). The inputs are PNG files saved by the caller, not seeds.

Reference: apple-aiml-research/ml-depth-pro-2 at the pinned commit below (fetched into --src), timm 1.0.27, torch >= 2.7.
Runs in the hartsy-ref:cpu Docker image, never on the Windows host:

  run.sh -v "<worktree>:/wt" -v "<models root>:/models:ro" -v "<inputs>:/in:ro" -- \
      python /wt/tests/python-reference/dump_depth_pro2_sizes.py --size b --out /work/ref-b
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
import urllib.request
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from safetensors.torch import load_file

COMMIT = "aa389cbb0b74341664ed663a0966c859905cd6c0"
RAW = f"https://raw.githubusercontent.com/apple-aiml-research/ml-depth-pro-2/{COMMIT}/src/depth_pro_2/"
SOURCES = ["__init__.py", "model.py", "utils.py", "streaming.py", "network/__init__.py", "network/decoder.py",
           "network/encoder.py", "network/fov.py"]
REPO_DIR = {"s": "DepthPro2-S", "b": "DepthPro2-B", "l": "DepthPro2-L", "h": "DepthPro2-H"}


def fetch_sources(root: Path) -> None:
    for name in SOURCES:
        target = root / "depth_pro_2" / name
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(RAW + name) as response:
            target.write_bytes(response.read())


def load_png(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", required=True, choices=sorted(REPO_DIR))
    parser.add_argument("--out", required=True)
    parser.add_argument("--src", default="/work/ml-depth-pro-2")
    parser.add_argument("--models", default="/models/Vision")
    parser.add_argument("--image", default="/in/room.png")
    parser.add_argument("--views", default="/in/views")
    parser.add_argument("--frames", default="/in/pan")
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

    torch.set_grad_enabled(False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = PRESETS[f"depthpro2-{args.size}"]
    model = DepthPro2(cfg.backbone_preset, cfg.reassembled_blocks, cfg.cross_frame_blocks, cfg.dims_reassemble,
                      cfg.dim_decoder, fov=True, precision=torch.float32)
    state = load_file(f"{args.models}/{REPO_DIR[args.size]}/model.safetensors")
    state.update({f"fov.{k}": v for k, v in load_file(f"{args.models}/DepthPro2-focal/model.safetensors").items()})
    model.load_state_dict(state, strict=True)
    model.eval()
    times: dict[str, float] = {}

    def keep(name: str, pred: dict, count: int) -> None:
        depth = pred["depth"].to(torch.float32).reshape(count, *pred["depth"].shape[-2:]).numpy()
        inverse = pred["inverse_depth"].to(torch.float32).reshape(count, *pred["inverse_depth"].shape[-2:]).numpy()
        np.save(out / f"{name}_depth.npy", depth)
        np.save(out / f"{name}_inverse_depth.npy", inverse)
        np.save(out / f"{name}_f_px.npy", np.asarray(pred["focallength_px"], dtype=np.float32).reshape(-1)[:1])

    start = time.time()
    x = to_normalized_tensor(load_png(Path(args.image)), torch.device("cpu")).unsqueeze(0)
    keep("image", model.infer(x), 1)
    times["image"] = time.time() - start

    start = time.time()
    views = [to_normalized_tensor(load_png(p), torch.device("cpu")) for p in sorted(Path(args.views).glob("*.png"))]
    keep("views", model.infer(torch.stack(views)), len(views))
    times["views"] = time.time() - start

    frames = [to_normalized_tensor(load_png(p), torch.device("cpu")) for p in sorted(Path(args.frames).glob("*.png"))]
    for name, window, overlap, mode in (("buffered", 4, 2, "buffered"), ("unbuffered", 3, 0, "unbuffered")):
        start = time.time()
        preds = list(model.infer_video(iter(frames), window=window, overlap=overlap, mode=mode))
        assert len(preds) == len(frames)
        keep(name, {"depth": torch.stack([p["depth"] for p in preds]),
                    "inverse_depth": torch.stack([p["inverse_depth"] for p in preds]),
                    "focallength_px": preds[0]["focallength_px"]}, len(frames))
        times[name] = time.time() - start

    meta = {"size": args.size, "frames": len(frames), "views": len(views), "seconds": times,
            "versions": {"torch": torch.__version__, "timm": timm.__version__, "ml_depth_pro_2": COMMIT}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()

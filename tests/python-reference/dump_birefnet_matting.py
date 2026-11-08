"""Offline reference dump for the C# BiRefNet-matting port (never imported by engine code).

Runs the upstream PyTorch model at the pinned revision and writes one safetensors file holding the input pixels,
the normalized network input, the logits, the 8-bit matte the upstream usage snippet produces, and a few stage taps.

Run with isolated mode so nothing in the working directory or the user site shadows the reference:

    python -I dump_birefnet_matting.py --code-dir <dir with birefnet_pkg/> --weights <model.safetensors> \
        --sha256 a9875de5b1e6c8eb5fdaa8c727a82927ce442cdc87ba3abee6a77e6fa46c25bb --out <dir>

<code-dir>/birefnet_pkg/ must hold the pinned ``birefnet.py`` and ``BiRefNet_config.py`` (plus an empty __init__.py),
fetched from huggingface.co/ZhengPeng7/BiRefNet-matting at revision eccde0a8cbdce7ac5fecfeb06340fe7b949e85d9.
"""

import argparse
import hashlib
import os
import sys

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from torchvision import transforms


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def synthetic_scene(width: int, height: int) -> np.ndarray:
    """A deterministic non-square RGB scene: a smooth background with an elliptical foreground."""
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    background = np.stack([
        60 + 120 * xs / width,
        90 + 80 * ys / height,
        200 - 100 * xs / width,
    ], axis=-1)
    cx, cy, rx, ry = width * 0.55, height * 0.5, width * 0.22, height * 0.36
    inside = ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2 <= 1.0
    foreground = np.stack([230 - 60 * ys / height, 140 + 40 * xs / width, 70 + 0 * xs], axis=-1)
    scene = np.where(inside[..., None], foreground, background)
    return np.clip(np.round(scene), 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    actual = sha256_of(args.weights)
    if actual != args.sha256:
        sys.exit(f"weights sha256 {actual} does not match the pinned {args.sha256}")

    sys.path.insert(0, args.code_dir)
    from birefnet_pkg.birefnet import BiRefNet  # the pinned upstream module

    torch.set_grad_enabled(False)
    model = BiRefNet(bb_pretrained=False)
    state = load_file(args.weights)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        sys.exit(f"state dict mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")
    model = model.float().eval()

    rgb = synthetic_scene(args.width, args.height)
    image = Image.fromarray(rgb)

    taps = {}
    bb_outputs = []
    model.bb.register_forward_hook(lambda m, i, o: bb_outputs.append([t.detach().clone() for t in o]))
    model.squeeze_module.register_forward_hook(lambda m, i, o: taps.__setitem__("squeeze_out", o.detach().clone()))
    model.decoder.decoder_block4.register_forward_hook(
        lambda m, i, o: taps.__setitem__("decoder_block4_out", o.detach().clone()))
    model.decoder.decoder_block1.register_forward_hook(
        lambda m, i, o: taps.__setitem__("decoder_block1_out", o.detach().clone()))

    transform_image = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    pixels = transform_image(image).unsqueeze(0)
    outputs = model(pixels)
    logits = outputs[-1] if isinstance(outputs, list) else outputs
    preds = logits.sigmoid()
    mask = transforms.ToPILImage()(preds[0].squeeze())
    mask = mask.resize(image.size)
    mask_u8 = np.asarray(mask, dtype=np.float32)

    tensors = {
        "input_rgb": torch.from_numpy(rgb.astype(np.float32)).contiguous(),
        "pixels": pixels.contiguous(),
        "logits": logits.contiguous(),
        "mask_u8": torch.from_numpy(mask_u8).contiguous(),
    }
    # The bb hook fires once for the full-resolution pass and then once for the half-resolution pass.
    for pass_name, outputs in zip(["full", "half"], bb_outputs[:2]):
        for stage, feature in enumerate(outputs):
            tensors[f"bb_{pass_name}_{stage}"] = feature.contiguous()
    for name, value in taps.items():
        tensors[name] = value.contiguous()

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "reference.safetensors")
    save_file(tensors, out_path)
    print(f"wrote {out_path} with {len(tensors)} tensors; logits range "
          f"[{float(logits.min()):.4f}, {float(logits.max()):.4f}]")


if __name__ == "__main__":
    main()

"""Reference dump for MOSS-VL-Realtime image preprocessing (pixels to patch rows).

MOSS-VL's processor is Qwen2-VL's fast image processor with MOSS's settings (patch 16, merge 2, temporal patch 1,
mean/std 0.5, bicubic) and a per-batch pixel budget. This script runs the processor that ships with the installed
`transformers` library, not the model repository's processing_moss_vl.py: no checkpoint code is executed. The batch
budget rule is not in that class, so each case passes the per-image max_pixels the rule would give; the C# unit tests
cover the rule itself.

Inputs are deterministic synthetic images. Outputs (committed as fixtures, all small):
  image_<i>.rgb          raw HWC uint8 RGB
  shapes_256x256.rgb/png a red disc and a blue square on white, for the pixels-to-reply test
  pixel_values_<i>.npy   float32 [grid_t*grid_h*grid_w, 768], merge-block order
  meta.json              per case: width, height, max_pixels, grid [t, h, w]

Usage: python dump_moss_vl_preprocess.py <out_dir>
"""

import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast

SEED = 20261108
MIN_PIXELS = 4096          # preprocessor_config.json size.shortest_edge
MAX_PIXELS = 16777216      # size.longest_edge, the per-image ceiling

# (width, height, max_pixels): a downscale to the 32 grid, a tall upscale-free case, and one where max_pixels binds.
CASES = [
    (100, 75, MAX_PIXELS),
    (64, 200, MAX_PIXELS),
    (320, 180, 8192),
    (40, 30, MAX_PIXELS),   # below min_pixels: smart_resize scales it up
]


def make_image(width, height, rng):
    y, x = np.mgrid[0:height, 0:width]
    r = x * 255.0 / max(width - 1, 1)
    g = y * 255.0 / max(height - 1, 1)
    b = ((x + y) % 32) * 8.0
    img = np.stack([r, g, b], axis=-1) + rng.integers(-24, 25, size=(height, width, 3))
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    meta = []
    for i, (width, height, max_pixels) in enumerate(CASES):
        img = make_image(width, height, rng)
        processor = Qwen2VLImageProcessorFast(
            size={"shortest_edge": MIN_PIXELS, "longest_edge": max_pixels},
            patch_size=16, temporal_patch_size=1, merge_size=2,
            image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5])
        result = processor(images=[Image.fromarray(img)], return_tensors="pt")
        pixels = result["pixel_values"].to(torch.float32).numpy()
        grid = [int(v) for v in result["image_grid_thw"][0].tolist()]
        img.tofile(os.path.join(out, f"image_{i}.rgb"))
        np.save(os.path.join(out, f"pixel_values_{i}.npy"), np.ascontiguousarray(pixels))
        meta.append({"width": width, "height": height, "max_pixels": max_pixels, "grid": grid,
                     "resample": str(processor.resample)})
        print(f"case {i}: {width}x{height} max {max_pixels} -> grid {grid}, pixel_values {pixels.shape}")
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump({"seed": SEED, "min_pixels": MIN_PIXELS, "cases": meta}, f, indent=2)
    write_shapes_fixture(out)


def write_shapes_fixture(out):
    """A 256x256 scene with unambiguous content for the pixels-to-reply test: a red disc and a blue square on white."""
    from PIL import ImageDraw
    img = Image.new("RGB", (256, 256), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse((24, 64, 124, 164), fill=(220, 20, 20))
    draw.rectangle((150, 80, 230, 160), fill=(20, 40, 210))
    np.asarray(img, dtype=np.uint8).tofile(os.path.join(out, "shapes_256x256.rgb"))
    img.save(os.path.join(out, "shapes_256x256.png"))


if __name__ == "__main__":
    main()

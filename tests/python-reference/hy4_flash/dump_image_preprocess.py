"""Reference for the Hy4 vision image preprocessing: the Hugging Face HYV4VLImageProcessor at transformers eb1d5db, on fixed uint8 images.

Run in the reference container, with this folder mounted at /work:

    docker run --rm -v <repo>/tests/python-reference/hy4_flash:/work hartsy-ref:cpu python /work/dump_image_preprocess.py /work/data/image_preprocess

Writes, per case <name>:
  <name>__rgb.u8               the input image, raw uint8 HWC, row-major (the C# test reads these exact pixels; height and width are in __grid.json)
  <name>__pixel_values.npy     the processor's pixel_values, float32, [patches, 768]
  <name>__grid.json            image_grid_thw as [t, h, w] in patches, and the smart_resize size it chose
  smart_resize table           smart_resize.json: (height, width) -> (h_bar, w_bar) for the released min and max pixels

Aligned cases (height and width multiples of 32 inside the pixel bounds) have no resampling, so their pixel values pin the
normalisation and patch order exactly. Unaligned cases pin the grid and the resize size; their pixel values depend on the
resampler and are written for the record.
"""

import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from transformers.models.hy_v4_vl.image_processing_hy_v4_vl import HYV4VLImageProcessor, smart_resize

HERE = os.path.dirname(os.path.abspath(__file__))
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "data", "image_preprocess")
os.makedirs(out, exist_ok=True)

released = json.load(open(os.path.join(HERE, "released_processor_config.json")))["image_processor"]
processor = HYV4VLImageProcessor(
    do_resize=True,
    size={"shortest_edge": released["min_pixels"], "longest_edge": released["max_pixels"]},
    image_mean=released["image_mean"],
    image_std=released["image_std"],
    patch_size=released["patch_size"],
    merge_size=released["merge_size"],
    temporal_patch_size=released["temporal_patch_size"],
)
FACTOR = released["patch_size"] * released["merge_size"]
MIN_PIXELS = released["min_pixels"]
MAX_PIXELS = released["max_pixels"]

# (name, height, width). Aligned: multiples of 32 with h*w inside [4096, 4194304]. Unaligned: anything else.
CASES = [
    ("aligned_96x128", 96, 128),
    ("aligned_256x160", 256, 160),
    ("unaligned_37x53", 37, 53),
    ("unaligned_300x200", 300, 200),
    ("unaligned_17x1000", 17, 1000),
]

for index, (name, height, width) in enumerate(CASES):
    rng = np.random.default_rng(1000 + index)
    rgb = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    rgb.tofile(os.path.join(out, f"{name}__rgb.u8"))
    result = processor(images=[Image.fromarray(rgb)], return_tensors="pt")
    pixel_values = result["pixel_values"].to(torch.float32).numpy()
    grid = result["image_grid_thw"].tolist()[0]
    h_bar, w_bar = smart_resize(height, width, factor=FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    np.save(os.path.join(out, f"{name}__pixel_values.npy"), pixel_values)
    with open(os.path.join(out, f"{name}__grid.json"), "w") as f:
        json.dump({"image_grid_thw": grid, "resized": [h_bar, w_bar], "input": [height, width]}, f, indent=1)
    print(name, "grid", grid, "resized", (h_bar, w_bar), "pixel_values", pixel_values.shape)

# The smart_resize table: integer maths only, so the C# side must match it exactly.
table = []
for h in range(1, 1200, 37):
    for w in range(1, 1200, 41):
        if max(h, w) / min(h, w) > 200:
            continue
        table.append({"in": [h, w], "out": list(smart_resize(h, w, factor=FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS))})
with open(os.path.join(out, "smart_resize.json"), "w") as f:
    json.dump({"factor": FACTOR, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS, "cases": table}, f)
print("smart_resize cases", len(table))

"""
Diffs C# Z-Image transformer layer dumps against the Python (diffusers) reference.

Usage:
  tests/python-reference/.venv/bin/python tests/python-reference/diff_zimage_layers.py [csharp_dir] [reference_dir]

Defaults:
  csharp_dir   = Output/zimage_csharp_dump
  reference_dir = tests/python-reference/zimage_reference_tensors/full_forward

For each layer in the reference index, computes |ref - cs| stats. The first layer
where avg_err jumps from ~1e-7 to >1e-3 is THE bug location.
"""
import json
import sys
import os
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ref_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    REPO_ROOT, "tests/python-reference/zimage_reference_tensors/full_forward"
)
cs_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO_ROOT, "Output/zimage_csharp_dump")

with open(os.path.join(ref_dir, "index.json")) as f:
    idx = json.load(f)

print(f"Reference dir: {ref_dir}")
print(f"C# dump dir:   {cs_dir}")
print()
print(f"{'layer':<28} {'shape (ref)':<22} {'avg_err':>10} {'max_err':>10} {'note'}")
print("-" * 95)

img_padded_len = None  # for slicing final_layer (C# is image-only)

for layer in idx["layers"]:
    name = layer["name"]
    shape = layer["shape"]
    safe = name.replace(".", "_")
    cs_path = os.path.join(cs_dir, "layers", f"{safe}.bin")
    ref_path = os.path.join(ref_dir, layer["file"])

    if not os.path.exists(cs_path):
        print(f"{name:<28} {str(shape):<22} {'-':>10} {'-':>10} <missing in C#>")
        continue
    if not os.path.exists(ref_path):
        print(f"{name:<28} {str(shape):<22} {'-':>10} {'-':>10} <missing in ref>")
        continue

    ref = np.fromfile(ref_path, dtype=np.float32)
    cs = np.fromfile(cs_path, dtype=np.float32)

    note = ""
    # Special case: x_embedder has [imgRealLen, hidden] (unbatched). C# is [B, imgRealLen, hidden] — same data.
    # cap_embedder similarly. Just compare flat.
    # final_layer: Python is [B, totalSeq, patchDim]; C# is [B, imgRealLen, patchDim] — slice ref to first imgRealLen.
    # layers.* / noise_refiner / context_refiner: same shape both sides.
    if name == "final_layer" and ref.size != cs.size:
        # Slice python ref to first cs.size elements (C# applies final_layer to image-only).
        # python is [1, total, 64] = 1 * totalSeq * 64 floats. C# is [1, imgPaddedLen, 64].
        if cs.size <= ref.size and ref.size % shape[2] == 0 and cs.size % shape[2] == 0:
            ref = ref[: cs.size]
            note = f"sliced ref to first {cs.size // shape[2]} tokens (C# image-only)"
        else:
            print(f"{name:<28} {str(shape):<22} {'-':>10} {'-':>10} <size mismatch ref={ref.size} cs={cs.size}>")
            continue

    if ref.size != cs.size:
        print(f"{name:<28} {str(shape):<22} {'-':>10} {'-':>10} <size mismatch ref={ref.size} cs={cs.size}>")
        continue

    diff = np.abs(ref.astype(np.float64) - cs.astype(np.float64))
    avg = diff.mean()
    mx = diff.max()
    flag = "  <-- BUG" if avg > 1e-3 else ""
    print(f"{name:<28} {str(shape):<22} {avg:>10.3e} {mx:>10.3e} {flag} {note}")

# Also diff the final output velocity if present.
print()
ref_out = os.path.join(ref_dir, "output_velocity.bin")
cs_out = os.path.join(cs_dir, "output_velocity.bin")
if os.path.exists(cs_out) and os.path.exists(ref_out):
    ref = np.fromfile(ref_out, dtype=np.float32)
    cs = np.fromfile(cs_out, dtype=np.float32)
    if ref.size == cs.size:
        diff = np.abs(ref.astype(np.float64) - cs.astype(np.float64))
        flag = "  <-- BUG" if diff.mean() > 1e-3 else ""
        print(f"{'output_velocity':<28} {'[1, 16, 32, 32]':<22} {diff.mean():>10.3e} {diff.max():>10.3e} {flag}")
    else:
        print(f"output_velocity size mismatch: ref={ref.size} cs={cs.size}")

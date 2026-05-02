"""
Diffs C# SD3.5 transformer layer dumps against the Python (diffusers) reference.

Usage:
  tests/python-reference/.venv/bin/python tests/python-reference/diff_sd35_layers.py [csharp_dir] [reference_dir]

Defaults:
  csharp_dir   = Output/sd35_csharp_dump
  reference_dir = tests/python-reference/sd35_reference_tensors/full_forward

For each layer in the reference index, prints |ref - cs| stats. The first layer
where avg_err jumps from ~1e-7 to ~1e-3+ is THE bug location.
"""
import json
import sys
import os
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

cs_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO_ROOT, "Output/sd35_csharp_dump")
ref_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO_ROOT, "tests/python-reference/sd35_reference_tensors/full_forward")

with open(os.path.join(ref_dir, "index.json")) as f:
    idx = json.load(f)

print(f"Reference dir: {ref_dir}")
print(f"C# dump dir:   {cs_dir}")
print()
print(f"{'layer':<28} {'shape':<25} {'avg_err':>11} {'max_err':>11} {'note'}")
print("-" * 100)

# Order entries: structural inputs first, then layers, then output
def sort_key(e):
    name = e["name"]
    if name.startswith("input_"): return (0, name)
    if name == "patch_embed": return (1, name)
    if name == "time_text_embed": return (2, name)
    if name.startswith("block_"):
        # extract block index and side
        parts = name.split("_")
        block_idx = int(parts[1])
        side = 0 if "image" in name else 1
        return (3, block_idx, side)
    if name == "norm_out": return (4, name)
    if name == "proj_out": return (5, name)
    if name == "output_velocity": return (6, name)
    return (7, name)

entries = sorted(idx, key=sort_key)

for entry in entries:
    name = entry["name"]
    shape = entry["shape"]
    if name.startswith("input_"):
        continue  # don't diff inputs (they're identical by construction)

    safe = name.replace(".", "_")
    cs_path = os.path.join(cs_dir, "layers", f"{safe}.bin")
    ref_path = os.path.join(ref_dir, entry["file"])

    # Special case: output_velocity is at root, not in layers/
    if name == "output_velocity":
        cs_path = os.path.join(cs_dir, "output_velocity.bin")

    if not os.path.exists(cs_path):
        print(f"{name:<28} {str(shape):<25} {'-':>11} {'-':>11} <missing in C#: {cs_path}>")
        continue
    if not os.path.exists(ref_path):
        print(f"{name:<28} {str(shape):<25} {'-':>11} {'-':>11} <missing in ref>")
        continue

    ref = np.fromfile(ref_path, dtype=np.float32)
    cs = np.fromfile(cs_path, dtype=np.float32)

    note = ""
    if ref.size != cs.size:
        # Sometimes shape transposed (e.g. [B,C,H,W] vs [B,H*W,C]). Try comparing element-wise sorted abs sums.
        print(f"{name:<28} {str(shape):<25} {'-':>11} {'-':>11} <size mismatch ref={ref.size} cs={cs.size}>")
        continue

    diff = np.abs(ref.astype(np.float64) - cs.astype(np.float64))
    avg = diff.mean()
    mx = diff.max()
    flag = "  <-- BUG" if avg > 1e-3 else ""
    print(f"{name:<28} {str(shape):<25} {avg:>11.3e} {mx:>11.3e} {flag} {note}")

print()

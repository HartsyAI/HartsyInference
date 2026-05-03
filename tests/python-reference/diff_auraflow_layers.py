"""
Diffs C# AuraFlow transformer layer dumps against the Python (diffusers) reference.

Usage:
  tests/python-reference/.venv/bin/python tests/python-reference/diff_auraflow_layers.py [csharp_dir] [reference_dir]

Defaults:
  csharp_dir   = Output/auraflow_csharp_dump  (set AURAFLOW_DEBUG_DIR to this in the C# diff test)
  reference_dir = tests/python-reference/auraflow_reference_tensors/full_forward

The first layer where avg_err jumps from ~1e-7 to ~1e-3+ is THE bug location.
"""
import json
import sys
import os
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

cs_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO_ROOT, "Output/auraflow_csharp_dump")
ref_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO_ROOT, "tests/python-reference/auraflow_reference_tensors/full_forward")

with open(os.path.join(ref_dir, "index.json")) as f:
    idx = json.load(f)

print(f"Reference dir: {ref_dir}")
print(f"C# dump dir:   {cs_dir}")
print()
print(f"{'layer':<32} {'shape':<25} {'avg_err':>11} {'max_err':>11} {'flag'}")
print("-" * 100)

# Sort entries
def sort_key(e):
    name = e["name"]
    if name.startswith("input_"): return (0, name)
    if name == "pos_embed": return (1, name)
    if name.startswith("joint_block_"):
        parts = name.split("_")
        idx_n = int(parts[2])
        side = 0 if "image" in name else 1
        return (2, idx_n, side)
    if name.startswith("single_block_"):
        return (3, int(name.split("_")[2]))
    if name == "norm_out": return (4, name)
    if name == "proj_out": return (5, name)
    if name == "output_velocity": return (6, name)
    return (7, name)

for entry in sorted(idx, key=sort_key):
    name = entry["name"]
    shape = entry["shape"]
    if name.startswith("input_"):
        continue

    safe = name.replace(".", "_")
    cs_path = os.path.join(cs_dir, "layers", f"{safe}.bin")
    ref_path = os.path.join(ref_dir, entry["file"])
    if name == "output_velocity":
        cs_path = os.path.join(cs_dir, "output_velocity.bin")

    if not os.path.exists(cs_path):
        print(f"{name:<32} {str(shape):<25} {'-':>11} {'-':>11} <missing in C#>")
        continue
    if not os.path.exists(ref_path):
        continue

    ref = np.fromfile(ref_path, dtype=np.float32)
    cs = np.fromfile(cs_path, dtype=np.float32)
    if ref.size != cs.size:
        print(f"{name:<32} {str(shape):<25} {'-':>11} {'-':>11} <size ref={ref.size} cs={cs.size}>")
        continue

    diff = np.abs(ref.astype(np.float64) - cs.astype(np.float64))
    avg = diff.mean()
    mx = diff.max()
    flag = "  <-- BUG" if avg > 1e-3 else ""
    print(f"{name:<32} {str(shape):<25} {avg:>11.3e} {mx:>11.3e} {flag}")

print()

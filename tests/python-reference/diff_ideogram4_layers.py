"""
Diffs the C# Ideogram4Transformer layer dumps against the Python reference.

Usage:
  python3 tests/python-reference/diff_ideogram4_layers.py [csharp_dir] [reference_dir]

Defaults:
  csharp_dir    = Output/ideogram4_csharp_dump        (written by Ideogram4DiffTests)
  reference_dir = tests/python-reference/ideogram4_reference_tensors/full_forward

For each tap in the reference index, prints |ref - cs| stats. The first tap where
avg_err jumps from ~1e-6 to ~1e-3+ is THE bug location. Tap order follows the forward
pass: mrope -> input_proj -> hidden_in -> adaln_input -> layers.0..N -> output_velocity.
"""
import json
import os
import sys
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

cs_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO_ROOT, "Output/ideogram4_csharp_dump")
ref_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    REPO_ROOT, "tests/python-reference/ideogram4_reference_tensors/full_forward")

with open(os.path.join(ref_dir, "index.json")) as f:
    idx = json.load(f)


def order(name):
    if name.startswith("mrope"): return (0, name)
    if name == "input_proj": return (1, name)
    if name == "hidden_in": return (2, name)
    if name == "adaln_input": return (3, name)
    if name.startswith("layers."): return (4, int(name.split(".")[1]))
    if name == "output_velocity": return (5, name)
    return (6, name)


entries = sorted(idx, key=lambda e: order(e["name"]))

print(f"Reference dir: {ref_dir}")
print(f"C# dump dir:   {cs_dir}")
print()
print(f"{'tap':<20} {'shape':<22} {'avg_err':>11} {'max_err':>11} {'note'}")
print("-" * 92)

worst = None
for entry in entries:
    name = entry["name"]
    shape = entry["shape"]
    safe = name.replace(".", "_")
    # output_velocity is written at the C# dump root; everything else under layers/.
    cs_path = os.path.join(cs_dir, "output_velocity.bin") if name == "output_velocity" \
        else os.path.join(cs_dir, "layers", f"{safe}.bin")
    ref_path = os.path.join(ref_dir, entry["file"])

    if not os.path.exists(cs_path):
        print(f"{name:<20} {str(shape):<22} {'-':>11} {'-':>11} <missing in C#: {cs_path}>")
        continue
    if not os.path.exists(ref_path):
        print(f"{name:<20} {str(shape):<22} {'-':>11} {'-':>11} <missing in ref>")
        continue

    ref = np.fromfile(ref_path, dtype=np.float32)
    cs = np.fromfile(cs_path, dtype=np.float32)
    if ref.size != cs.size:
        print(f"{name:<20} {str(shape):<22} {'-':>11} {'-':>11} <size mismatch ref={ref.size} cs={cs.size}>")
        continue

    diff = np.abs(ref.astype(np.float64) - cs.astype(np.float64))
    avg, mx = diff.mean(), diff.max()
    flag = "  <-- DIVERGENCE" if avg > 1e-3 else ""
    if worst is None and avg > 1e-3:
        worst = name
    print(f"{name:<20} {str(shape):<22} {avg:>11.3e} {mx:>11.3e} {flag}")

print()
if worst is None:
    print("PASS: all taps within 1e-3. C# matches the Python reference.")
else:
    print(f"FIRST DIVERGENCE at: {worst}")
    print("  Inspect that step in Ideogram4Transformer/Ideogram4Block, or flip the matching")
    print("  SPEC_* toggle in dump_ideogram4_full_forward.py and re-run if it's the final layer.")

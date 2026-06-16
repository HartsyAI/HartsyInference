"""
Diffs the C# ACE-Step v1 DiT layer dumps against the Python reference.

Usage (after running AceStepDitDiffTests then dump_ace_step_dit.py):
  python3 tests/python-reference/diff_ace_step_layers.py

Reference taps live in Output/ace_step_dit_parity/ref (index.json); C# dumps in .../cs.
First tap where avg_err jumps to ~1e-3+ is THE bug. Tap order: patch_embed -> tblock -> layers.* -> velocity.
"""
import json
import os
import sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "Output", "ace_step_dit_parity")
ref_dir = os.path.join(ROOT, "ref")
cs_dir = os.path.join(ROOT, "cs")

with open(os.path.join(ref_dir, "index.json")) as f:
    idx = json.load(f)

def order(name):
    if name == "patch_embed": return (0, "")
    if name == "tblock": return (1, "")
    if name.startswith("layers."): return (2, int(name.split(".")[1]))
    if name == "output_velocity": return (3, "")
    return (4, name)

print(f"Reference: {ref_dir}\nC# dump:   {cs_dir}\n")
print(f"{'tap':<18} {'shape':<18} {'avg_err':>11} {'max_err':>11} {'note'}")
print("-" * 78)

worst = None
for e in sorted(idx, key=lambda e: order(e["name"])):
    name = e["name"]; safe = name.replace(".", "_")
    cs_path = os.path.join(cs_dir, "output_velocity.bin") if name == "output_velocity" \
        else os.path.join(cs_dir, "layers", f"{safe}.bin")
    ref_path = os.path.join(ref_dir, e["file"])
    if not os.path.exists(cs_path):
        print(f"{name:<18} {str(e['shape']):<18} {'-':>11} {'-':>11} <missing C#: {cs_path}>"); continue
    ref = np.fromfile(ref_path, np.float32); cs = np.fromfile(cs_path, np.float32)
    if ref.size != cs.size:
        print(f"{name:<18} {str(e['shape']):<18} {'-':>11} {'-':>11} <size ref={ref.size} cs={cs.size}>"); continue
    d = np.abs(ref.astype(np.float64) - cs.astype(np.float64))
    avg, mx = d.mean(), d.max()
    flag = "  <-- DIVERGENCE" if avg > 1e-3 else ""
    if worst is None and avg > 1e-3: worst = name
    print(f"{name:<18} {str(e['shape']):<18} {avg:>11.3e} {mx:>11.3e} {flag}")

print()
print("PASS: C# matches the Python reference." if worst is None else f"FIRST DIVERGENCE at: {worst}")

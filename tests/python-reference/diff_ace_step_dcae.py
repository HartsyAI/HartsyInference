"""
Diffs the C# ACE-Step Music-DCAE decoder layer dumps against the Python reference.

Usage (after running AceStepDcaeDiffTests then dump_ace_step_dcae.py):
  python3 tests/python-reference/diff_ace_step_dcae.py

Reference taps live in Output/ace_step_dcae_parity/ref (index.json); C# dumps in .../cs/layers.
First tap where corr drops below 0.999 / max_err jumps to ~1e-3+ is THE bug.
Tap order: conv_in -> stage.3 -> upsample.2 -> stage.2 -> ... -> stage.0 -> norm_out -> conv_out.
"""
import json
import os
import sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "Output", "ace_step_dcae_parity")
ref_dir = os.path.join(ROOT, "ref")
cs_dir = os.path.join(ROOT, "cs", "layers")

with open(os.path.join(ref_dir, "index.json")) as f:
    idx = json.load(f)

def order(name):
    if name == "dcae.conv_in": return (0, 0)
    if name.startswith("dcae.stage."): return (1, -int(name.split(".")[-1]))
    if name.startswith("dcae.upsample."): return (1, -int(name.split(".")[-1]) - 0.5)
    if name == "dcae.norm_out": return (2, 0)
    if name == "dcae.conv_out": return (3, 0)
    return (4, name)

def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    da = np.sqrt((a * a).sum()); db = np.sqrt((b * b).sum())
    if da == 0 or db == 0:
        return 1.0 if da == db else 0.0
    return float((a * b).sum() / (da * db))

print(f"Reference: {ref_dir}\nC# dump:   {cs_dir}\n")
print(f"{'tap':<20} {'shape':<18} {'corr':>10} {'avg_err':>11} {'max_err':>11} {'note'}")
print("-" * 86)

worst = None
for e in sorted(idx, key=lambda e: order(e["name"])):
    name = e["name"]; safe = name.replace(".", "_")
    cs_path = os.path.join(cs_dir, f"{safe}.bin")
    ref_path = os.path.join(ref_dir, e["file"])
    if not os.path.exists(cs_path):
        print(f"{name:<20} {str(e['shape']):<18} {'-':>10} {'-':>11} {'-':>11} <missing C#: {cs_path}>"); continue
    ref = np.fromfile(ref_path, np.float32); cs = np.fromfile(cs_path, np.float32)
    if ref.size != cs.size:
        print(f"{name:<20} {str(e['shape']):<18} {'-':>10} {'-':>11} {'-':>11} <size ref={ref.size} cs={cs.size}>"); continue
    rd = ref.astype(np.float64); cd = cs.astype(np.float64)
    c = corr(rd, cd)
    d = np.abs(rd - cd); avg, mx = d.mean(), d.max()
    bad = c < 0.999 or mx > 1e-3
    flag = "  <-- DIVERGENCE" if bad else ""
    if worst is None and bad: worst = name
    print(f"{name:<20} {str(e['shape']):<18} {c:>10.6f} {avg:>11.3e} {mx:>11.3e} {flag}")

print()
print("PASS: C# matches the Python reference (corr > 0.999, max_err < 1e-3)." if worst is None
      else f"FIRST DIVERGENCE at: {worst}")

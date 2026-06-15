"""
Diffs C# Hunyuan3D layer dumps against the Python reference (dump_hunyuan3d_full_forward.py).

The C# side writes its dumps via Hunyuan3DDebugDump when HARTSYINFERENCE_HUNYUAN3D_DEBUG_DIR is set, one
flat `<tag>.bin` per tag (tags match the Python reference layer names: cond_tokens, dit_block_<i>,
vae_occupancy, output_velocity, ...). Run the C# generation with that env var pointed at `csharp_dir`,
then run this script to find the first layer that diverges.

Usage:
  tests/python-reference/.venv/bin/python tests/python-reference/diff_hunyuan3d_layers.py [csharp_dir] [reference_dir]
"""
import json, sys, os, numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
cs_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO_ROOT, "Output/hunyuan3d_csharp_dump")
ref_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO_ROOT, "tests/python-reference/hunyuan3d_reference_tensors/full_forward")

index_path = os.path.join(ref_dir, "index.json")
if not os.path.exists(index_path):
    raise SystemExit(f"missing reference index.json at {index_path} — run dump_hunyuan3d_full_forward.py first")

with open(index_path) as f:
    idx = json.load(f)

print(f"Reference: {ref_dir}\nC#: {cs_dir}\n")
print(f"{'layer':<28} {'shape':<22} {'avg_err':>11} {'max_err':>11} flag")
print("-" * 86)

first_bug = None
for entry in idx:
    name = entry["name"]
    if name.startswith("input_") or entry["file"].startswith("inputs/"):
        continue  # inputs are fed identically; not a divergence point
    cs_path = os.path.join(cs_dir, f"{name}.bin")
    ref_path = os.path.join(ref_dir, entry["file"])
    if not os.path.exists(cs_path):
        print(f"{name:<28} {str(entry['shape']):<22} {'-':>11} {'-':>11} <missing C#>")
        continue
    if not os.path.exists(ref_path):
        continue
    ref = np.fromfile(ref_path, dtype=np.float32)
    cs = np.fromfile(cs_path, dtype=np.float32)
    if ref.size != cs.size:
        print(f"{name:<28} {str(entry['shape']):<22} {'-':>11} {'-':>11} <size {ref.size}/{cs.size}>")
        continue
    d = np.abs(ref.astype(np.float64) - cs.astype(np.float64))
    is_bug = d.mean() > 1e-3
    flag = "  <-- BUG" if is_bug else ""
    if is_bug and first_bug is None:
        first_bug = name
    print(f"{name:<28} {str(entry['shape']):<22} {d.mean():>11.3e} {d.max():>11.3e} {flag}")

print("-" * 86)
print(f"first divergence: {first_bug}" if first_bug else "all compared layers within tolerance (1e-3)")

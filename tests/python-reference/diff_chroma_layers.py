"""
Diffs C# Chroma transformer layer dumps against the Python (diffusers) reference.

Usage:
  tests/python-reference/.venv/bin/python tests/python-reference/diff_chroma_layers.py [csharp_dir] [reference_dir]
"""
import json, sys, os, numpy as np
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
cs_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO_ROOT, "Output/chroma_csharp_dump")
ref_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO_ROOT, "tests/python-reference/chroma_reference_tensors/full_forward")

with open(os.path.join(ref_dir, "index.json")) as f: idx = json.load(f)
print(f"Reference: {ref_dir}\nC#: {cs_dir}\n")
print(f"{'layer':<32} {'shape':<25} {'avg_err':>11} {'max_err':>11} flag")
print("-" * 95)
def cs_filename_for(name: str) -> str:
    """Translate Python ref layer name → C# dump filename.
    Python uses tuple-hook indices (double_N_0 = txt out, double_N_1 = img out — diffusers'
    Chroma block returns `(encoder_hidden_states, hidden_states)` in that order). C# uses
    explicit `double_N_txt_out` / `double_N_img_out` names.
    """
    if name == "output_velocity": return "output_velocity.bin"
    if name.startswith("double_"):
        parts = name.split("_")
        if len(parts) == 3 and parts[2] in ("0", "1"):
            suffix = "txt_out" if parts[2] == "0" else "img_out"
            return f"layers/double_{parts[1]}_{suffix}.bin"
    return f"layers/{name.replace('.', '_')}.bin"

for entry in idx:
    name = entry["name"]
    if name.startswith("input_"): continue
    cs_rel = cs_filename_for(name)
    cs_path = os.path.join(cs_dir, cs_rel)
    ref_path = os.path.join(ref_dir, entry["file"])
    if not os.path.exists(cs_path):
        print(f"{name:<32} {str(entry['shape']):<25} {'-':>11} {'-':>11} <missing C#>"); continue
    if not os.path.exists(ref_path): continue
    ref = np.fromfile(ref_path, dtype=np.float32); cs = np.fromfile(cs_path, dtype=np.float32)
    if ref.size != cs.size:
        print(f"{name:<32} {str(entry['shape']):<25} {'-':>11} {'-':>11} <size {ref.size}/{cs.size}>"); continue
    d = np.abs(ref.astype(np.float64) - cs.astype(np.float64))
    flag = "  <-- BUG" if d.mean() > 1e-3 else ""
    print(f"{name:<32} {str(entry['shape']):<25} {d.mean():>11.3e} {d.max():>11.3e} {flag}")

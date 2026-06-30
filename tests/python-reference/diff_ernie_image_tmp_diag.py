"""TEMPORARY DIAGNOSTIC diff (ERNIE-Image haze). Safe to delete.
maxAbs + correlation + std-ratio per component, in forward order."""
import json, sys, os, numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
cs_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "Output/ernie_image_csharp_dump")
ref_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "tests/python-reference/ernie_image_reference_tensors/full_forward")
idx = json.load(open(os.path.join(ref_dir, "index.json")))
order = ["patch_embed","text_proj","time_embedding","modulation_shift_msa","modulation_scale_msa",
         "modulation_gate_msa","modulation_shift_mlp","modulation_scale_mlp","modulation_gate_mlp",
         "rope_freqs"] + [f"layers.{i}" for i in range(36)] + ["final_linear","output_velocity"]
by = {e["name"]: e for e in idx}
print(f"Ref: {ref_dir}\nC#:  {cs_dir}\n")
print(f"{'component':<24}{'shape':<20}{'maxAbs':>11}{'avgAbs':>11}{'corr':>9}{'stdRef':>9}{'stdCS':>9} flag")
print("-"*112)
for name in order:
    if name not in by: continue
    e = by[name]; safe = name.replace(".", "_")
    cs_path = os.path.join(cs_dir, "output_velocity.bin" if name=="output_velocity" else f"layers/{safe}.bin")
    ref_path = os.path.join(ref_dir, e["file"])
    if not os.path.exists(cs_path): print(f"{name:<24}{str(e['shape']):<20}{'<no C#>':>11}"); continue
    ref = np.fromfile(ref_path, np.float32).astype(np.float64)
    cs  = np.fromfile(cs_path,  np.float32).astype(np.float64)
    if ref.size != cs.size: print(f"{name:<24}{str(e['shape']):<20} size {ref.size}/{cs.size}"); continue
    d = np.abs(ref-cs)
    corr = np.corrcoef(ref, cs)[0,1] if ref.std()>0 and cs.std()>0 else float('nan')
    flag = "  <-- FIRST DIVERGE" if (d.max()>1e-2 and corr<0.999) else ("  <-- err" if d.max()>1e-2 else "")
    print(f"{name:<24}{str(e['shape']):<20}{d.max():>11.3e}{d.mean():>11.3e}{corr:>9.5f}{ref.std():>9.3f}{cs.std():>9.3f}{flag}")

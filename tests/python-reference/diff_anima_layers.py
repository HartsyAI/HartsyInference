"""
Diffs C# Anima transformer layer dumps against the Python (diffusers) reference.

Usage:
  tests/python-reference/.venv/bin/python tests/python-reference/diff_anima_layers.py [csharp_dir] [reference_dir]

Defaults:
  csharp_dir   = Output/anima_csharp_dump            (produced by AnimaDiffTests.cs)
  reference_dir = tests/python-reference/anima_reference_tensors/full_forward

For each layer in the reference index, reads both binaries (flat F32) and computes
|ref - cs| stats. The first layer where avg_err jumps from ~1e-4 (numerical noise)
to >1e-2 (real divergence) is the bug location.

Name mapping (C# uses different naming than the diffusers reference):

    Python (reference)      C# (AnimaDebugDump)         Notes
    ──────────────────────  ──────────────────────────  ──────────────────────────────────
    patch_embed             patch_embed                 [1,256,2048] (Python flattens 5D
                                                        →3D internally so we just compare
                                                        flat data — element count matches)
    time_embed              temb                        time_embed returns (temb, embT);
                                                        Python's hook captured the first
                                                        (temb). [1, 6144]
    block_00..27            block_0..27                 Python pads index, C# doesn't
    norm_out                — (no C# equivalent)        C# fuses norm_out + proj_out into
                                                        a single ApplyFinalLayer step
                                                        and dumps the final result as
                                                        `final_proj`
    proj_out                final_proj                  Final per-token projection output
    output_velocity         output_velocity             Final unpatchified velocity

Note that Python captures shape may have extra dims compared to C#, but the FLAT
element data is what matters and matches.
"""
import json
import sys
import os
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ref_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    REPO_ROOT, "tests/python-reference/anima_reference_tensors/full_forward"
)
cs_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO_ROOT, "Output/anima_csharp_dump")

with open(os.path.join(ref_dir, "index.json")) as f:
    idx = json.load(f)

print(f"Reference dir: {ref_dir}")
print(f"C# dump dir:   {cs_dir}")
print()
print(f"{'layer (python name)':<24} {'C# name':<20} {'shape (ref)':<28} {'avg_err':>11} {'max_err':>11} {'note'}")
print("-" * 120)


def cs_name_for(python_name: str) -> str:
    """Map a Python reference layer name to the corresponding C# dump filename (without .bin)."""
    if python_name == "time_embed":
        return "temb"
    if python_name == "proj_out":
        return "final_proj"
    if python_name.startswith("block_"):
        # Python pads to 2 digits (block_00..27); C# does not (block_0..27).
        idx_str = python_name[len("block_"):]
        return f"block_{int(idx_str)}"
    return python_name


def safe_filename(name: str) -> str:
    """AnimaDebugDump replaces '.' and '/' with '_'. Mirror that."""
    return name.replace(".", "_").replace("/", "_")


# Sort layers in pipeline order: patch_embed → time_embed → block_00..27 → norm_out → proj_out
def sort_key(layer):
    name = layer["name"]
    if name == "patch_embed":
        return (0, 0)
    if name == "time_embed":
        return (1, 0)
    if name.startswith("block_"):
        return (2, int(name[len("block_"):]))
    if name == "norm_out":
        return (3, 0)
    if name == "proj_out":
        return (4, 0)
    return (5, 0)


for layer in sorted(idx["layers"], key=sort_key):
    py_name = layer["name"]
    shape = layer["shape"]
    cs_layer_name = cs_name_for(py_name)
    cs_path = os.path.join(cs_dir, "layers", f"{safe_filename(cs_layer_name)}.bin")
    ref_path = os.path.join(ref_dir, layer["file"])

    note = ""
    if py_name == "norm_out":
        note = "(no C# equivalent — see proj_out/final_proj)"
        print(f"{py_name:<24} {'—':<20} {str(shape):<28} {'—':>11} {'—':>11} {note}")
        continue

    if not os.path.exists(cs_path):
        print(f"{py_name:<24} {cs_layer_name:<20} {str(shape):<28} {'—':>11} {'—':>11} <missing C# file: {cs_path}>")
        continue
    if not os.path.exists(ref_path):
        print(f"{py_name:<24} {cs_layer_name:<20} {str(shape):<28} {'—':>11} {'—':>11} <missing reference>")
        continue

    ref = np.fromfile(ref_path, dtype=np.float32)
    cs = np.fromfile(cs_path, dtype=np.float32)

    if ref.size != cs.size:
        print(f"{py_name:<24} {cs_layer_name:<20} {str(shape):<28} {'—':>11} {'—':>11} <size mismatch: ref={ref.size}, cs={cs.size}>")
        continue

    diff = np.abs(ref - cs)
    avg_err = float(diff.mean())
    max_err = float(diff.max())

    # Highlight the first significant divergence
    marker = ""
    if avg_err > 1e-2:
        marker = " ← DIVERGENT"
    elif avg_err > 1e-3:
        marker = " ← drift"

    print(f"{py_name:<24} {cs_layer_name:<20} {str(shape):<28} {avg_err:>11.3e} {max_err:>11.3e}{marker}")


# ── Final output_velocity ──
ref_out_path = os.path.join(ref_dir, "output_velocity.bin")
cs_out_path = os.path.join(cs_dir, "output_velocity.bin")
print("-" * 120)
if os.path.exists(ref_out_path) and os.path.exists(cs_out_path):
    ref = np.fromfile(ref_out_path, dtype=np.float32)
    cs = np.fromfile(cs_out_path, dtype=np.float32)
    if ref.size == cs.size:
        diff = np.abs(ref - cs)
        avg_err = float(diff.mean())
        max_err = float(diff.max())
        marker = " ← DIVERGENT" if avg_err > 1e-2 else (" ← drift" if avg_err > 1e-3 else "")
        py_shape = idx["output_shape"]
        print(f"{'output_velocity':<24} {'output_velocity':<20} {str(py_shape):<28} {avg_err:>11.3e} {max_err:>11.3e}{marker}")
        print(f"  ref stats:  mean={ref.mean():+.4f}  std={ref.std():.4f}  abs_mean={np.abs(ref).mean():.4f}")
        print(f"  cs  stats:  mean={cs.mean():+.4f}  std={cs.std():.4f}  abs_mean={np.abs(cs).mean():.4f}")
    else:
        print(f"output_velocity:        size mismatch ref={ref.size}, cs={cs.size}")
else:
    print(f"output_velocity:        missing — ref_exists={os.path.exists(ref_out_path)}, cs_exists={os.path.exists(cs_out_path)}")

print()
print("Interpretation:")
print("  avg_err < 1e-4  : F32 numerical noise — layer matches reference")
print("  avg_err ~ 1e-3  : minor drift (acceptable for ~28-block accumulation)")
print("  avg_err > 1e-2  : real divergence — bug in or before this layer")
print()
print("Find the FIRST 'DIVERGENT' or 'drift' row — that's where the bug starts.")

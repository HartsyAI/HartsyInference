"""
Diffs C# AnimaLlmAdapter layer dumps against the canonical diffusion-pipe Python reference.

Usage:
  tests/python-reference/.venv/bin/python tests/python-reference/diff_anima_llm_adapter_layers.py [csharp_dir] [reference_dir]

Defaults:
  csharp_dir    = Output/anima_llm_adapter_csharp_dump   (AnimaLlmAdapterDiffTests.cs writes this)
  reference_dir = tests/python-reference/anima_llm_adapter_reference

For each layer in the reference index, reads both binaries (flat F32) and computes |ref - cs| stats.
First layer where avg_err jumps above the F32 noise floor (~1e-5) into the 1e-3+ range is the bug location.
"""
import json
import sys
import os
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

cs_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO_ROOT, "Output/anima_llm_adapter_csharp_dump")
ref_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO_ROOT, "tests/python-reference/anima_llm_adapter_reference")

with open(os.path.join(ref_dir, "index.json")) as f:
    idx = json.load(f)

print(f"Reference dir: {ref_dir}")
print(f"C# dump dir:   {cs_dir}")
print(f"Prompt:        {idx.get('prompt', '?')!r}")
print(f"T5 seq_len:    {idx.get('t5_seq_len', '?')}")
print(f"Qwen3 seq_len: {idx.get('qwen3_seq_len', '?')}")
print()
print(f"{'layer (python name)':<24} {'C# file':<24} {'shape (ref)':<22} {'avg_err':>11} {'max_err':>11} {'note'}")
print("-" * 120)


def safe_filename(name: str) -> str:
    return name.replace(".", "_").replace("/", "_")


def sort_key(layer):
    name = layer["name"]
    order = {"x_post_embed": 0, "post_out_proj": 100, "final_output": 101}
    if name in order:
        return order[name]
    if name.startswith("block_"):
        return 1 + int(name[len("block_"):])
    return 999


for layer in sorted(idx["layers"], key=sort_key):
    py_name = layer["name"]
    shape = layer["shape"]
    cs_file = safe_filename(py_name) + ".bin"
    cs_path = os.path.join(cs_dir, "layers", cs_file)
    ref_path = os.path.join(ref_dir, layer["file"])

    if not os.path.exists(cs_path):
        print(f"{py_name:<24} {cs_file:<24} {str(shape):<22} {'—':>11} {'—':>11} <missing C# file>")
        continue
    if not os.path.exists(ref_path):
        print(f"{py_name:<24} {cs_file:<24} {str(shape):<22} {'—':>11} {'—':>11} <missing reference>")
        continue

    ref = np.fromfile(ref_path, dtype=np.float32)
    cs = np.fromfile(cs_path, dtype=np.float32)

    if ref.size != cs.size:
        print(f"{py_name:<24} {cs_file:<24} {str(shape):<22} {'—':>11} {'—':>11} <size mismatch: ref={ref.size}, cs={cs.size}>")
        continue

    diff = np.abs(ref - cs)
    avg_err = float(diff.mean())
    max_err = float(diff.max())

    marker = ""
    if avg_err > 1e-2:
        marker = " ← DIVERGENT"
    elif avg_err > 1e-3:
        marker = " ← drift"

    print(f"{py_name:<24} {cs_file:<24} {str(shape):<22} {avg_err:>11.3e} {max_err:>11.3e}{marker}")

# ── Final output comparison ──
ref_out_path = os.path.join(ref_dir, "final_output.bin")
cs_out_path = os.path.join(cs_dir, "layers", "final_output.bin")
print("-" * 120)
if os.path.exists(ref_out_path) and os.path.exists(cs_out_path):
    ref = np.fromfile(ref_out_path, dtype=np.float32)
    cs = np.fromfile(cs_out_path, dtype=np.float32)
    if ref.size == cs.size:
        diff = np.abs(ref - cs)
        marker = " ← DIVERGENT" if diff.mean() > 1e-2 else (" ← drift" if diff.mean() > 1e-3 else "")
        print(f"{'final_output':<24} {'final_output.bin':<24} {'(see ref index)':<22} {diff.mean():>11.3e} {diff.max():>11.3e}{marker}")
        print(f"  ref stats: mean={ref.mean():+.4f}  std={ref.std():.4f}  abs_mean={np.abs(ref).mean():.4f}")
        print(f"  cs  stats: mean={cs.mean():+.4f}  std={cs.std():.4f}  abs_mean={np.abs(cs).mean():.4f}")
    else:
        print(f"final_output:           size mismatch ref={ref.size}, cs={cs.size}")
else:
    print(f"final_output:           missing — ref_exists={os.path.exists(ref_out_path)}, cs_exists={os.path.exists(cs_out_path)}")

print()
print("Interpretation:")
print("  avg_err < 1e-4 : F32 numerical noise — layer matches reference")
print("  avg_err ~ 1e-3 : minor drift")
print("  avg_err > 1e-2 : real divergence — bug in or before this layer")
print()
print("Common suspects (in order of pipeline):")
print("  x_post_embed diverges  → embed.weight not loaded, or T5 ids translated wrong.")
print("  block_0 diverges       → RoPE convention, QK-norm order, cross-attn K/V source.")
print("  later blocks drift     → small accumulating numerical error (usually acceptable).")
print("  post_out_proj diverges → out_proj weight/bias not loaded.")
print("  final_output diverges  → final-RMSNorm/out_proj order is wrong.")

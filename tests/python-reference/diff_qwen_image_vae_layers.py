"""
Diffs C# Qwen-Image VAE decoder layer dumps against the Python (diffusers) reference.

Usage:
  tests/python-reference/.venv/bin/python tests/python-reference/diff_qwen_image_vae_layers.py [csharp_dir] [reference_dir]

Defaults:
  csharp_dir    = Output/qwen_image_vae_csharp_dump        (produced by QwenImageVaeDiffTests.cs)
  reference_dir = tests/python-reference/qwen_image_vae_reference

For each layer in the reference index, reads both binaries (flat F32) and computes
|ref - cs| stats. The first layer where avg_err jumps from ~1e-4 (numerical noise)
to >1e-2 (real divergence) is the bug location.

Layer name mapping (Python reference → C# AnimaDebugDump-style file name):
    Python (reference)            C# dump file (under layers/)             Notes
    ────────────────────────────  ───────────────────────────────────────  ─────────────────────────
    post_quant_conv               post_quant_conv.bin                       After top-level conv2
    decoder.conv_in               decoder_conv_in.bin                       After 16→384 conv
    decoder.mid_block             decoder_mid_block.bin                     After middle (res→attn→res)
    decoder.up_blocks.0..3        decoder_up_blocks_0..3.bin                After each up_block group
    decoder.conv_out              decoder_conv_out.bin                      Final 3-ch RGB

The Python hooks capture 5D tensors [B, C, 1, H, W] (T=1 for image mode); C# captures 4D
[B, C, H, W]. Element counts match — we diff the flat F32 buffers directly.
"""
import json
import sys
import os
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

cs_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    REPO_ROOT, "Output/qwen_image_vae_csharp_dump"
)
ref_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    REPO_ROOT, "tests/python-reference/qwen_image_vae_reference"
)

with open(os.path.join(ref_dir, "index.json")) as f:
    idx = json.load(f)

print(f"Reference dir: {ref_dir}")
print(f"C# dump dir:   {cs_dir}")
print()
print(f"{'layer (python name)':<26} {'C# file':<26} {'shape (ref)':<28} {'avg_err':>11} {'max_err':>11} {'note'}")
print("-" * 120)


def safe_filename(name: str) -> str:
    """C# QwenImageVaeDebugDump replaces '.' and '/' with '_'. Mirror that."""
    return name.replace(".", "_").replace("/", "_")


# Pipeline order: post_quant_conv → conv_in → mid_block → up_blocks.0..3 → conv_out
def sort_key(layer):
    name = layer["name"]
    order = {
        "post_quant_conv":     0,
        "decoder.conv_in":     1,
        "decoder.mid_block":   2,
        "decoder.up_blocks.0": 3,
        "decoder.up_blocks.1": 4,
        "decoder.up_blocks.2": 5,
        "decoder.up_blocks.3": 6,
        "decoder.conv_out":    7,
    }
    return order.get(name, 99)


for layer in sorted(idx["layers"], key=sort_key):
    py_name = layer["name"]
    shape = layer["shape"]
    cs_file = safe_filename(py_name) + ".bin"
    cs_path = os.path.join(cs_dir, "layers", cs_file)
    ref_path = os.path.join(ref_dir, layer["file"])

    if not os.path.exists(cs_path):
        print(f"{py_name:<26} {cs_file:<26} {str(shape):<28} {'—':>11} {'—':>11} <missing C# file: {cs_path}>")
        continue
    if not os.path.exists(ref_path):
        print(f"{py_name:<26} {cs_file:<26} {str(shape):<28} {'—':>11} {'—':>11} <missing reference>")
        continue

    ref = np.fromfile(ref_path, dtype=np.float32)
    cs = np.fromfile(cs_path, dtype=np.float32)

    if ref.size != cs.size:
        print(f"{py_name:<26} {cs_file:<26} {str(shape):<28} {'—':>11} {'—':>11} <size mismatch: ref={ref.size}, cs={cs.size}>")
        continue

    diff = np.abs(ref - cs)
    avg_err = float(diff.mean())
    max_err = float(diff.max())

    marker = ""
    if avg_err > 1e-2:
        marker = " ← DIVERGENT"
    elif avg_err > 1e-3:
        marker = " ← drift"

    print(f"{py_name:<26} {cs_file:<26} {str(shape):<28} {avg_err:>11.3e} {max_err:>11.3e}{marker}")


# ── Final output_image ──
ref_out_path = os.path.join(ref_dir, "output_image.bin")
cs_out_path = os.path.join(cs_dir, "output_image.bin")
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
        print(f"{'output_image':<26} {'output_image.bin':<26} {str(py_shape):<28} {avg_err:>11.3e} {max_err:>11.3e}{marker}")
        print(f"  ref stats:  mean={ref.mean():+.4f}  std={ref.std():.4f}  min={ref.min():+.4f}  max={ref.max():+.4f}")
        print(f"  cs  stats:  mean={cs.mean():+.4f}  std={cs.std():.4f}  min={cs.min():+.4f}  max={cs.max():+.4f}")
    else:
        print(f"output_image:              size mismatch ref={ref.size}, cs={cs.size}")
else:
    print(f"output_image:              missing — ref_exists={os.path.exists(ref_out_path)}, cs_exists={os.path.exists(cs_out_path)}")

print()
print("Interpretation:")
print("  avg_err < 1e-4  : F32 numerical noise — layer matches reference")
print("  avg_err ~ 1e-3  : minor drift (acceptable for deep stacks)")
print("  avg_err > 1e-2  : real divergence — bug in or before this layer")
print()
print("Find the FIRST 'DIVERGENT' or 'drift' row — that's where the bug starts.")
print()
print("Common suspects:")
print("  post_quant_conv diverges → per-channel rescale (latents_mean/std) wrong, or conv2 weight not loaded.")
print("  decoder.conv_in diverges → 16→384 conv (causal conv kT=3 → temporalSlot=-1 slice) wrong.")
print("  decoder.mid_block diverges → ResidualBlock or AttentionBlock impl.")
print("  decoder.up_blocks.N diverges → Resample (2× spatial upsample) or shortcut conv at the boundary.")
print("  decoder.conv_out diverges → RMSNorm gamma flatten or final 3-ch conv (kT=3 slice).")

# Z-Image Layer-by-Layer Debug Plan

> Pipeline runs end-to-end and produces a recognizable humanoid silhouette matching the prompt, but with heavy high-frequency banding artifacts. ~10 bugs already fixed (see `PHASE_3_DEVIATIONS.md` #25–27). The remaining bug(s) live somewhere in the 30 main DiT layers / 4 refiners / patchify / RoPE / attention plumbing.
>
> **Methodology**: same one that nailed Flux down to single-layer precision in deviation #3 — feed BOTH runtimes the same synthetic inputs, hook every layer in Python, dump tensors to disk, run the same inputs through C# with a debug-dump path, diff layer-by-layer, find the FIRST divergence (jumps from `~1e-7` to `>1e-3`), that's the bug.

## Inputs to fix everything to

Synthetic, deterministic, fully reproducible. No real prompts. No real noise. Just:

| Input | Spec | Reason |
|---|---|---|
| `latent` | `[1, 16, 32, 32]` F32, deterministic seed 42, mean 0 std 1 | Small enough for fast iteration; same shape Z-Image expects |
| `caption_embeddings` | `[1, 64, 2560]` F32, deterministic seed 42 | Skips Qwen3 entirely so we isolate the transformer |
| `sigma` | scalar 0.5 | Mid-schedule timestep |

Saved to `tests/python-reference/zimage_reference_tensors/inputs/{latent.bin, caption.bin}` so the C# side reads the **exact same bits**.

---

## Phase 1 — Python reference (estimated 1 session)

### 1.1 Verify venv works with Z-Image

```bash
cd /home/kalebbroo/Desktop/Projects/HartsyInference
python3 -m venv tests/python-reference/.venv
tests/python-reference/.venv/bin/pip install \
    "torch>=2.1" --index-url https://download.pytorch.org/whl/cu121
tests/python-reference/.venv/bin/pip install \
    "diffusers>=0.37" "transformers>=4.45" safetensors sentencepiece accelerate
tests/python-reference/.venv/bin/python -c \
    "from diffusers import ZImageTransformer2DModel; print('OK')"
```

If `ZImageTransformer2DModel` import fails, install diffusers from main:
```bash
tests/python-reference/.venv/bin/pip install \
    git+https://github.com/huggingface/diffusers.git
```

### 1.2 Pull the Tongyi-MAI/Z-Image-Turbo diffusers shards

We need the **transformer subfolder only** (the 6 GB diffusers safetensors split, NOT the SwarmUI single-file). The diffusers loader populates the model from the `transformer/` subfolder + `transformer/config.json`.

```bash
tests/python-reference/.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('Tongyi-MAI/Z-Image-Turbo', allow_patterns=['transformer/*'],
    local_dir='/home/kalebbroo/Desktop/Projects/HartsyInference/tests/test-models/zimage-turbo')
"
```

This is BF16 ~12 GB (twice the FP8Mix size). Disk has plenty of room.

### 1.3 Write `dump_zimage_full_forward.py`

Based on `dump_flux_full_forward.py`. Hook every relevant module and save its output. Roughly:

```python
# tests/python-reference/dump_zimage_full_forward.py

import sys, os, json
import torch
import numpy as np
from diffusers import ZImageTransformer2DModel

MODEL_DIR = "tests/test-models/zimage-turbo"
OUT_DIR   = "tests/python-reference/zimage_reference_tensors/full_forward"

def save(t, path):
    t.float().cpu().contiguous().numpy().tofile(path)

def stats(name, t):
    f = t.float().flatten()
    return {"name": name, "shape": list(t.shape), "dtype": str(t.dtype),
            "mean": float(f.mean()), "std": float(f.std()),
            "min": float(f.min()), "max": float(f.max()),
            "first_8": [float(x) for x in f[:8]]}

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f"{OUT_DIR}/inputs", exist_ok=True)
os.makedirs(f"{OUT_DIR}/layers", exist_ok=True)

# ----- Deterministic synthetic inputs -----
torch.manual_seed(42)
B, C, H, W = 1, 16, 32, 32
latent  = torch.randn(B, C, H, W, dtype=torch.float32)
caption = torch.randn(B, 64, 2560, dtype=torch.float32)
sigma   = 0.5
save(latent,  f"{OUT_DIR}/inputs/latent.bin")
save(caption, f"{OUT_DIR}/inputs/caption.bin")

# ----- Load transformer in F32 (so reference matches FP8 path's F32-cast loaded weights byte-for-byte) -----
xfm = ZImageTransformer2DModel.from_pretrained(
    MODEL_DIR, subfolder="transformer", torch_dtype=torch.float32
).eval()

# ----- Register hooks on every named submodule we care about -----
captures = {}
def hook(name):
    def _h(mod, _in, out):
        if isinstance(out, tuple): out = out[0]
        captures[name] = out.detach().clone()
    return _h

# Embedders
xfm.t_embedder.register_forward_hook(hook("t_embedder"))
xfm.cap_embedder.register_forward_hook(hook("cap_embedder"))
xfm.x_embedder.register_forward_hook(hook("x_embedder"))

# Refiners (2 each)
for i, blk in enumerate(xfm.context_refiner):
    blk.register_forward_hook(hook(f"context_refiner.{i}"))
for i, blk in enumerate(xfm.noise_refiner):
    blk.register_forward_hook(hook(f"noise_refiner.{i}"))

# 30 main layers
for i, blk in enumerate(xfm.layers):
    blk.register_forward_hook(hook(f"layers.{i}"))

# Final layer
xfm.final_layer.register_forward_hook(hook("final_layer"))

# ----- Forward -----
# The diffusers transformer signature varies; check the actual __call__.
# Likely: forward(hidden_states, encoder_hidden_states, timestep, ...)
# Match the EXACT call the diffusers pipeline makes (pipeline_z_image.py:520-555).
# Pre-multiply timestep by 1000 inside the call as the pipeline does.
with torch.no_grad():
    output = xfm(
        hidden_states=latent.unsqueeze(2),                        # [1, 16, 1, 32, 32] – check unsqueeze axis
        encoder_hidden_states=caption,
        timestep=torch.tensor([(1.0 - sigma) * 1000.0]),          # diffusers timestep convention
        return_dict=False,
    )[0]

save(output, f"{OUT_DIR}/output_velocity.bin")

# ----- Persist captures + stats index -----
index = []
for name, t in captures.items():
    safe = name.replace(".", "_")
    save(t, f"{OUT_DIR}/layers/{safe}.bin")
    index.append(stats(name, t) | {"file": f"layers/{safe}.bin"})

with open(f"{OUT_DIR}/index.json", "w") as f:
    json.dump({
        "config": dict(xfm.config),
        "inputs": {"latent": stats("latent", latent), "caption": stats("caption", caption), "sigma": sigma},
        "output": stats("output_velocity", output),
        "layers": index
    }, f, indent=2)

print(f"Dumped {len(captures)} layers + 1 final output to {OUT_DIR}")
```

**Estimated runtime**: ~30 sec on RTX 3060, ~3 min on CPU.

### 1.4 Smoke-test the dump

After running:
```bash
ls tests/python-reference/zimage_reference_tensors/full_forward/layers/ | wc -l
# Expect: 1 (t_embedder) + 1 (cap_embedder) + 1 (x_embedder)
#       + 2 (context_refiner) + 2 (noise_refiner)
#       + 30 (layers) + 1 (final_layer) = 38
cat tests/python-reference/zimage_reference_tensors/full_forward/index.json | python3 -m json.tool | head -30
```

---

## Phase 2 — C# diagnostic dump (estimated half a session)

### 2.1 Add a debug-dump mode to `ZImageTransformer.Forward`

Pattern: env var `Z_IMAGE_DEBUG_DIR=path` → if set, write each layer's output to `path/layers/<safe_name>.bin` after every captured site, exact same naming as the Python script. No-op overhead when the env var is unset.

Sites to dump (mirror Phase 1.3):
- `t_embedder` — output of `ComputeTimestepEmbedding`
- `cap_embedder` — output of `EmbedCaption`
- `x_embedder` — output of the patch Linear
- `context_refiner.{0,1}` — return value of each block
- `noise_refiner.{0,1}` — return value of each block
- `layers.{0..29}` — return value of each main block
- `final_layer` — output of `ApplyFinalLayer`
- `output_velocity` — final return value of `Forward`

### 2.2 Add a C# test that reads the SAME synthetic inputs and runs the transformer

```csharp
[Fact]
public void Transformer_Matches_PythonReference_LayerByLayer()
{
    string refDir = Environment.GetEnvironmentVariable("Z_IMAGE_REFERENCE_DIR")
        ?? Path.Combine(LinuxRepoRoot, "tests/python-reference/zimage_reference_tensors/full_forward");
    if (!File.Exists(Path.Combine(refDir, "inputs/latent.bin"))) { _output.WriteLine("SKIPPED"); return; }

    // Load synthetic inputs from disk (same bytes Python saw).
    Tensor latent  = LoadF32Bin(Path.Combine(refDir, "inputs/latent.bin"),  new TensorShape(1, 16, 32, 32));
    Tensor caption = LoadF32Bin(Path.Combine(refDir, "inputs/caption.bin"), new TensorShape(1, 64, 2560));

    // Set debug dump dir BEFORE Forward.
    string dumpDir = Path.Combine(LinuxRepoRoot, "Output/zimage_csharp_dump");
    Environment.SetEnvironmentVariable("Z_IMAGE_DEBUG_DIR", dumpDir);

    // ... load transformer (FP8Mix or F32 — must match what Python loaded) ...
    Tensor output = transformer.Forward(backend, latent, caption, sigma: 0.5f);

    // Diff every layer.
    string idxPath = Path.Combine(refDir, "index.json");
    var idx = JsonDocument.Parse(File.ReadAllText(idxPath));
    foreach (var layer in idx.RootElement.GetProperty("layers").EnumerateArray())
    {
        string name = layer.GetProperty("name").GetString()!;
        string safe = name.Replace(".", "_");
        string pyPath = Path.Combine(refDir, $"layers/{safe}.bin");
        string csPath = Path.Combine(dumpDir, $"layers/{safe}.bin");

        (float avgErr, float maxErr) = DiffF32Files(pyPath, csPath);
        _output.WriteLine($"  {name,-30} avg_err={avgErr:E3} max_err={maxErr:E3}");
        // First layer with avg_err > 1e-3 is THE BUG.
    }
}
```

### 2.3 Mind the dtype caveat

The Python reference dumps F32 weights → F32 forward. To make the C# diff meaningful **without an FP8 dequant noise floor**, the C# side must also load Z-Image in F32 (via `CastWeightsToF32` on the transformer dict). Yes that's ~28 GB peak, so use the `Phase A → dispose Qwen3 → Phase B` pattern from the existing test. Or just don't load Qwen3 at all in this test — synthetic caption_embeddings are pre-generated.

---

## Phase 3 — Triage the first divergence

Once you have a clean `avg_err` per layer, the first jump from `~1e-7` to `>1e-3` is the bug location. Likely culprits and where to look in each:

| First-diverging layer | Most likely bug |
|---|---|
| `t_embedder` | Sinusoidal sin/cos order, mlp shape mismatch, time scaling |
| `cap_embedder` | RMSNorm eps, projection bias presence, weight transposition |
| `x_embedder` | **Patchify channel order** (`[pH, pW, C]` vs `[C, pH, pW]` etc) — currently #1 suspect |
| `context_refiner.0` | Attention mask not enforced for caption pad slots; QKV split order; RoPE on caption (caption refiner has NO rope per diffusers) |
| `noise_refiner.0` | AdaLN modulation not yet applied at this stage? Image-only RoPE with frame=1 vs something else; pad token application |
| `layers.0` | **Concat order [image, caption] vs [caption, image]**; full-sequence RoPE with frame=capRealLen+1 for image, 1..capRealLen for caption; attention mask combining |
| `layers.{1..29}` | Residual / gate sign flips that compound; would show as monotonic err growth |
| `final_layer` | LayerNorm-no-affine eps, scale-only modulation, `[hidden, ADALN_EMBED_DIM]` vs `[2*hidden, ...]` |

If `t_embedder` already diverges, the bug is in `ComputeTimestepEmbedding` and we never had a chance.
If everything is `~1e-7` until `layers.0` and then goes wild, the bug is in the per-block forward (most likely RoPE or attention mask).
If `context_refiner` is fine but `noise_refiner` is wrong, the bug is in AdaLN or image-only position IDs.

### Sub-component drill-down (when block-level shows divergence)

Once we know which block fails first, write a sub-component dump similar to `dump_attn_sublayers.py`:
- norm1 output
- modulated (after `(1+scale_msa)`)
- q, k, v post-projection
- q_norm, k_norm output
- post-RoPE q, k
- attention output (post-SDPA)
- to_out output
- norm2 output
- post-residual

That tells us **which sub-op** is wrong. From there, the bug is usually a 1–3 line code change.

---

## Phase 4 — Fix and verify

Apply the fix, re-run Phase 1 + 2, confirm `avg_err < 1e-4` for every layer. Then re-run the end-to-end pipeline test and the image should be coherent.

If FP8 noise floor adds `~1e-3` of error (vs F32 ref), that's expected and benign — at FP8 the reference comparison should be done at F16 precision throughout (cast everything once, run, dump).

---

## Common Z-Image-specific gotchas to verify

These are things Flux *didn't* have but Z-Image does, and any one could be the remaining bug:

1. **Attention mask for pad tokens**: diffusers builds `[B, seq]` bool mask, broadcast to `[B,1,1,seq]` → SDPA. C# currently passes `null` mask → pad tokens participate fully. Could absolutely cause banding.
2. **`tanh()` on gates after `chunk(4)`** — added, but verify the indexing: scale_msa(0), gate_msa(1), scale_mlp(2), gate_mlp(3). The diffusers `mod.unsqueeze(1).chunk(4, dim=2)` chunks on dim=2 (the feature dim), our chunk takes the same.
3. **Pad token application order**: diffusers does `where(mask, pad_token, feats)` BEFORE RoPE; C# applies pad tokens after embedding but before refiner — order matters if pad position IDs differ from real ones (they do — pads are (0,0,0)).
4. **Sequence concat order [image, caption]** — fixed; verify slicing direction matches.
5. **Static shift 3.0 vs dynamic shift**: Z-Image scheduler config says `use_dynamic_shifting=False, shift=3.0`. Diffusers Z-Image pipeline reportedly computes dynamic shift Flux-style anyway — find out which is actually used.
6. **Timestep input format**: diffusers takes `(1 - sigma) * 1000` and the transformer multiplies by `t_scale=1000` internally → final sinusoidal input is `(1 - sigma) * 1000 * 1000 = 1e6 * (1 - sigma)`?? Or just `(1 - sigma) * 1000`?? Verify.
7. **Caption refiner has NO RoPE** — confirm; my context_refiner doesn't apply RoPE which matches.
8. **Image refiner uses frame=1 image-only RoPE** vs main layers use frame=capRealLen+1 — verify against diffusers pipeline.

Pick whichever bug shows up first in the per-layer diff. Don't fix more than one thing at a time.

---

## Pseudo-code: minimum-viable diff utility

```python
# tests/python-reference/diff_zimage_layers.py
import json, numpy as np, sys

ref_dir = "tests/python-reference/zimage_reference_tensors/full_forward"
cs_dir  = sys.argv[1] if len(sys.argv) > 1 else "Output/zimage_csharp_dump"

idx = json.load(open(f"{ref_dir}/index.json"))
print(f"{'layer':<32} {'avg_err':>12} {'max_err':>12} {'shape':>20}")

for layer in idx["layers"]:
    name, shape = layer["name"], layer["shape"]
    safe = name.replace(".", "_")
    ref = np.fromfile(f"{ref_dir}/layers/{safe}.bin", dtype=np.float32).reshape(shape)
    try:
        cs = np.fromfile(f"{cs_dir}/layers/{safe}.bin", dtype=np.float32).reshape(shape)
    except FileNotFoundError:
        print(f"{name:<32} <missing in C# dump>")
        continue
    diff = (ref - cs).astype(np.float64)
    print(f"{name:<32} {float(np.abs(diff).mean()):>12.3e} {float(np.abs(diff).max()):>12.3e} {str(shape):>20}")
```

Run order:
```bash
# 1. Generate Python reference (slow — ~30 sec on GPU)
tests/python-reference/.venv/bin/python tests/python-reference/dump_zimage_full_forward.py

# 2. Run C# diff test (fast — synthetic inputs, no Qwen3)
LD_LIBRARY_PATH=/tmp/cuda-libs dotnet test tests/HartsyInference.Diffusion.Tests/HartsyInference.Diffusion.Tests.csproj \
    --filter "FullyQualifiedName~Transformer_Matches_PythonReference_LayerByLayer"

# 3. Diff
tests/python-reference/.venv/bin/python tests/python-reference/diff_zimage_layers.py
```

Expected output (when broken):
```
layer                              avg_err      max_err          shape
t_embedder                       1.234e-07    8.940e-07           [1, 256]
cap_embedder                     2.110e-07    1.234e-06        [1, 64, 3840]
x_embedder                       6.700e-02    9.110e-01        [1, 256, 3840]   ← BUG
context_refiner.0                ... (anything after the bug compounds, ignore)
```

Expected output (when fixed):
```
layer                              avg_err      max_err          shape
t_embedder                       1.2e-07      8.9e-07
cap_embedder                     2.1e-07      1.2e-06
x_embedder                       1.5e-07      9.0e-07
context_refiner.0                3.2e-06      8.4e-05
... (all layers < 1e-3) ...
final_layer                      4.0e-04      6.7e-03          ← FP32 accumulation tolerance
```

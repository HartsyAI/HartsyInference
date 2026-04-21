# Phase 3 — Deviations from Design Plan

This document tracks every case where the C# implementation diverged from the reference Python (diffusers/PyTorch) behavior, how the bug was found, and how it was fixed. It serves as a debugging journal and a guide for future model ports.

---

## 1. BatchedMatMul — 2D Right Operand Silently Produced Zeros

**Design assumption**: `BatchedMatMul(a[B,M,K], b[K,N])` would correctly handle a 2D weight matrix broadcast across the batch dimension.

**Deviation**: `MatMulKernels.BatchedMatMul` read `N = b.Shape[2]`, but for a 2D tensor `[K, N]`, `Shape[2]` returns 0 (uninitialized `_dim2` in `TensorShape`). This made N=0, causing every matmul to produce an all-zeros output. The issue affected:
- **CLIP text encoder**: All 12 transformer layers were no-ops (residual passthrough only). The UNet received raw token+position embeddings instead of contextual text representations — the text prompt was effectively ignored.
- **VAE mid-block attention**: Attention projections produced zeros, making attention a residual passthrough.

**Fix**: Added 2D detection: `bool bIs2D = b.Shape.Rank == 2; long N = bIs2D ? b.Shape[1] : b.Shape[2];` and set `bSliceSize = 0` for 2D to reuse the same weight pointer across all batch slices.

**Impact**: This was the primary cause of the "brownish blob" output — without functional text encoding, the UNet had no semantic conditioning and produced essentially random noise predictions.

## 2. UNet Self-Attention — K/V Projected from Un-Normed Hidden

**Design assumption**: `TransformerSubBlock` would correctly apply LayerNorm before Q, K, and V projections for self-attention.

**Deviation**: The `TransformerSubBlock.Forward` method applied LayerNorm to produce `normed`, then projected Q from `normed` but K and V from the raw `context` parameter. For self-attention (where `context == hidden`), this meant K/V came from un-normed input while Q came from normed input. In diffusers, `attn1(norm_hidden_states)` passes the normed tensor for all of Q, K, V.

**Fix**: Added `ReferenceEquals(hidden, context)` check to detect self-attention and route K/V through the normed tensor: `Tensor kvSource = ReferenceEquals(hidden, context) ? normed : context;`

## 3. UNet Attention Head Count — Inverted Head/Dim Interpretation

**Design assumption**: `UNetConfig.AttentionHeadDim` values (8 for SD1.5) represent the per-head dimension, so `numHeads = channels / headDim`.

**Deviation (original)**: Originally passed 8 directly as numHeads — which was actually correct for diffusers semantics.

**Deviation (incorrect fix)**: Changed to `numHeads = channels / AttentionHeadDim[i]` = 320/8 = 40 heads with headDim=8. This was wrong — diffusers uses `attention_head_dim=8` to mean **8 attention heads** (when `num_attention_heads` is not specified). The confusing naming in diffusers led to the misinterpretation.

**How it was found**: Layer-by-layer binary comparison against Python reference tensors. The error was invisible at the pipeline level (images were "plausible but bad") but obvious when comparing per-layer outputs:
- `down_blocks.0.resnets.0`: avg_err=3.5e-7 (perfect)
- `down_blocks.0.attentions.0`: avg_err=0.127 (first divergence!)
- Errors compounded through all subsequent layers to avg_err=1.808 at mid_block

Running `dump_attn_sublayers.py` confirmed: Python uses 8 heads with head_dim=40, while C# used 40 heads with head_dim=8. This caused:
- Wrong attention scale: `1/sqrt(8)=0.354` vs correct `1/sqrt(40)=0.158` (2.24x too large)
- Wrong multi-head split pattern: 40 tiny 8-dim heads instead of 8 larger 40-dim heads
- Completely different attention distributions, causing ~56% signal dampening over 20 denoising steps

**Fix**: Renamed config property `AttentionHeadDim` → `NumAttentionHeads` for clarity. Changed UNet constructor to use the value directly: `numHeads = config.NumAttentionHeads[i]` instead of `outCh / config.AttentionHeadDim[i]`. Applied at all three sites (down blocks, mid block, up blocks).

**Result**: First attention block error dropped from avg_err=0.127 to avg_err=4.3e-5 (2,940x improvement). All layers now match Python within float32 accumulation tolerance. Pipeline produces coherent images.

**Lesson**: Diffusers' `attention_head_dim` parameter is confusingly named — it specifies **head count**, not head dimension, when `num_attention_heads` is not provided. Always verify multi-head attention shapes against the reference by printing `attn.heads` and checking the actual Q/K/V reshape dimensions. See the troubleshooting guide below.

## 4. VAE Attention — 3D Tensors Passed to 4D SDPA Kernel

**Design assumption**: The VAE attention layer could pass tensors directly to `ScaledDotProductAttention`.

**Deviation**: `VaeAttention` passed 3D tensors `[B, seqLen, C]` to the SDPA kernel which expects 4D `[B, H, S, D]`. With a 3D tensor, `Shape[3]` returned 0 (uninitialized `_dim3`), making the head dimension D=0. The attention kernel's inner loops iterated zero times, producing all-zeros output.

**Fix**: Added reshape to 4D before SDPA: `[B, seqLen, C]` → `[B, 1, seqLen, C]` (single-head attention), then reshape back after.

## 5. Timestep Embedding — Sin/Cos Order Swapped

**Design assumption**: Sinusoidal timestep embedding would use `[sin, cos]` layout.

**Deviation**: SD1.5 diffusers uses `flip_sin_to_cos=True` (default), producing `[cos, sin]` layout. Our code had `[sin, cos]`.

**Fix**: Swapped to `embPtr[i] = cos(angle); embPtr[halfDim + i] = sin(angle);`

## 6. Euler Scheduler — Missing scale_model_input

**Design assumption**: The Euler scheduler would not need to scale model input before each UNet forward pass.

**Deviation**: Diffusers' `EulerDiscreteScheduler.scale_model_input` divides the latent by `sqrt(sigma^2 + 1)` before each UNet call. Without this, the UNet receives inputs at the wrong scale.

**Fix**: Added `ScaleModelInput(stepIndex)` to `IScheduler` interface and implemented in `EulerDiscreteScheduler`. Applied scaling in both `Generate` and `GenerateFromTokens` denoising loops.

## 7. Euler Step — Division by Zero at Final Timestep

**Design assumption**: The Euler step formula `derivative = (sample - pred_x0) / sigma` would work for all timesteps.

**Deviation**: At the final timestep (t=0), sigma approaches 0, causing division by zero. The algebraic simplification for epsilon prediction eliminates this: since `derivative = model_output` (the division cancels), the step reduces to `prev_sample = sample + model_output * (sigma_next - sigma)` with no division needed.

**Fix**: Simplified the epsilon-prediction path to avoid division. Added sigma guard for v-prediction path.

---

## Troubleshooting Methodology

The following approach was developed during SD1.5 debugging and should be reused for all future model ports.

### Step 1: Build a Python Reference Pipeline

Create a Python script (`tests/python-reference/dump_reference_stats.py`) that runs the full pipeline with known inputs and saves:
- Initial noise tensor (binary)
- Text embeddings (binary)
- Per-step latent tensors (binary + JSON stats: mean, std, min, max)
- Final latent tensor (binary)

Use a **venv** to avoid system Python conflicts: `python -m venv tests/python-reference/.venv`

### Step 2: Run C# Pipeline with Python's Noise

Write a test (`CrossRuntimeValidationTests.PipelineWithPythonNoiseMatchesReference`) that loads Python's saved initial noise and text embeddings, runs the C# pipeline, and compares per-step latent statistics. This eliminates RNG differences and isolates model/scheduler bugs.

### Step 3: Single Forward Pass Comparison

If per-step stats diverge, isolate a single UNet forward pass (`CrossRuntimeValidationTests.SingleUNetPassMatchesPythonReference`). Save Python's step-0 inputs and outputs, feed the same inputs to C#, compare the output tensor element-wise. If this diverges, the bug is in the UNet (not the scheduler).

### Step 4: Layer-by-Layer Binary Comparison

Create a Python script (`tests/python-reference/dump_layer_outputs.py`) that hooks every UNet layer and saves each layer's output as a binary tensor. Then write a C# test (`UNetDiagnosticTests.LayerByLayerComparisonWithPythonInputs`) that manually steps through the UNet one layer at a time, comparing each output against the Python reference. This pinpoints the **first divergent layer**.

Key pattern:
```
time_embedding:              avg_err=3.4e-8  (PERFECT — no bug here)
conv_in:                     avg_err=0.0     (PERFECT)
down_blocks.0.resnets.0:    avg_err=3.5e-7  (PERFECT)
down_blocks.0.attentions.0: avg_err=0.127   ← FIRST DIVERGENCE — bug is here
```

### Step 5: Sub-Layer Decomposition

Once the divergent layer is identified, create a Python script (`tests/python-reference/dump_attn_sublayers.py`) that manually executes each sub-operation within that layer and saves intermediate tensors. For a CrossAttentionBlock, this means ~20 sub-steps: GroupNorm, reshape, proj_in, LayerNorm, Q/K/V projections, multi-head reshape, attention logits, softmax probs, attention output, merge, output projection, residuals, cross-attention intermediates, FFN intermediates, proj_out, final output.

Compare the C# sub-operations against these references to find the exact operation that introduces error.

### Step 6: Fix and Verify

After fixing the bug, re-run the layer-by-layer comparison to confirm all layers now match within float32 tolerance (avg_err < 1e-3, ideally < 1e-4). Then run the full pipeline comparison to verify end-to-end correctness.

---

## Advice for Future Model Ports

### Attention Configuration is the #1 Trap

Every model framework names attention parameters differently. The same parameter name can mean different things across models:

| Framework/Model | Parameter | Meaning |
|---|---|---|
| diffusers SD1.5 | `attention_head_dim=8` | **8 heads** (confusing! not head dim) |
| diffusers SDXL | `attention_head_dim=[5,10,20]` | Per-block head counts |
| Some configs | `num_heads=8` | 8 heads (clear) |
| Some configs | `head_dim=64` | 64-dim per head (clear) |

**Always verify** by printing `model.attn.heads` and checking Q/K/V reshape shapes in the Python reference before writing C# config code. Never trust parameter names alone.

### Weight Shape vs Usage Mismatches

Safetensor weights carry their original shapes. A `proj_in.weight` with shape `[320, 320, 1, 1]` is a Conv2d, but applying it as a linear projection on reshaped data is mathematically equivalent for 1x1 kernels. However, for non-1x1 kernels, the im2col/GEMM path must be used. Always check the weight shape before assuming Linear vs Conv.

### GELU Variant Differences

- C# currently uses **tanh-approximated GELU**: `x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
- PyTorch default `F.gelu` uses **exact erf-based GELU**: `x * 0.5 * (1 + erf(x / sqrt(2)))`
- Some models explicitly use `approximate="tanh"` which matches our C# version

For SD1.5, the GEGLU FFN uses the default (erf-based). The difference is small (~1e-4 relative error for typical values) and acceptable for inference. If a future model shows unexplained FFN divergence, check which GELU variant the reference uses.

### RNG Differences are Expected

C# uses Box-Muller for Gaussian noise; PyTorch uses a different algorithm. Same seed produces different noise. For validation, always compare using **shared noise tensors** (saved from Python, loaded in C#), never by matching seeds.

### Float32 Accumulation Tolerance

Different operation ordering and SIMD reduction patterns cause small float32 differences. Expected tolerances for correctly-implemented layers:

| Layer type | Expected avg_err | Notes |
|---|---|---|
| Element-wise (Add, SiLU) | < 1e-7 | Nearly exact |
| GroupNorm, LayerNorm | < 1e-6 | Reduction order matters |
| Linear/Conv (GEMM) | < 1e-5 | Accumulation order |
| Full attention block | < 1e-4 | Compounds through softmax |
| Full UNet pass | < 1e-3 | Accumulated through all layers |

If a layer exceeds these by 10x+, there's a real bug — not just floating-point noise.

### Diagnostic Script Inventory

All scripts live in `tests/python-reference/` and use the venv at `tests/python-reference/.venv/`:

| Script | Purpose |
|---|---|
| `dump_reference_stats.py` | Full pipeline: saves noise, embeddings, per-step latents, final output |
| `dump_layer_outputs.py` | Per-layer UNet outputs with index.json |
| `dump_attn_sublayers.py` | Sub-operation breakdown of first CrossAttentionBlock |
| `compare_layers.py` | Utility for comparing binary tensor files |

To run: `"tests/python-reference/.venv/Scripts/python" "tests/python-reference/<script>.py"`

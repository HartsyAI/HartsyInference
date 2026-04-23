# Phase 3 — Deviations from Design Plan

This document tracks every case where the C# implementation diverged from the reference Python (diffusers/PyTorch) behavior, how the bug was found, and how it was fixed. It serves as a debugging journal and a guide for future model ports.

---

## CPU Pipeline Deviations

### 1. BatchedMatMul — 2D Right Operand Silently Produced Zeros

**Design assumption**: `BatchedMatMul(a[B,M,K], b[K,N])` would correctly handle a 2D weight matrix broadcast across the batch dimension.

**Deviation**: `MatMulKernels.BatchedMatMul` read `N = b.Shape[2]`, but for a 2D tensor `[K, N]`, `Shape[2]` returns 0 (uninitialized `_dim2` in `TensorShape`). This made N=0, causing every matmul to produce an all-zeros output. The issue affected:
- **CLIP text encoder**: All 12 transformer layers were no-ops (residual passthrough only). The UNet received raw token+position embeddings instead of contextual text representations — the text prompt was effectively ignored.
- **VAE mid-block attention**: Attention projections produced zeros, making attention a residual passthrough.

**Fix**: Added 2D detection: `bool bIs2D = b.Shape.Rank == 2; long N = bIs2D ? b.Shape[1] : b.Shape[2];` and set `bSliceSize = 0` for 2D to reuse the same weight pointer across all batch slices.

**Impact**: This was the primary cause of the "brownish blob" output — without functional text encoding, the UNet had no semantic conditioning and produced essentially random noise predictions.

### 2. UNet Self-Attention — K/V Projected from Un-Normed Hidden

**Design assumption**: `TransformerSubBlock` would correctly apply LayerNorm before Q, K, and V projections for self-attention.

**Deviation**: The `TransformerSubBlock.Forward` method applied LayerNorm to produce `normed`, then projected Q from `normed` but K and V from the raw `context` parameter. For self-attention (where `context == hidden`), this meant K/V came from un-normed input while Q came from normed input. In diffusers, `attn1(norm_hidden_states)` passes the normed tensor for all of Q, K, V.

**Fix**: Added `ReferenceEquals(hidden, context)` check to detect self-attention and route K/V through the normed tensor: `Tensor kvSource = ReferenceEquals(hidden, context) ? normed : context;`

### 3. UNet Attention Head Count — Inverted Head/Dim Interpretation

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

**Lesson**: Diffusers' `attention_head_dim` parameter is confusingly named — it specifies **head count**, not head dimension, when `num_attention_heads` is not provided. Always verify multi-head attention shapes against the reference by printing `attn.heads` and checking the actual Q/K/V reshape dimensions.

### 4. VAE Attention — 3D Tensors Passed to 4D SDPA Kernel

**Design assumption**: The VAE attention layer could pass tensors directly to `ScaledDotProductAttention`.

**Deviation**: `VaeAttention` passed 3D tensors `[B, seqLen, C]` to the SDPA kernel which expects 4D `[B, H, S, D]`. With a 3D tensor, `Shape[3]` returned 0 (uninitialized `_dim3`), making the head dimension D=0. The attention kernel's inner loops iterated zero times, producing all-zeros output.

**Fix**: Added reshape to 4D before SDPA: `[B, seqLen, C]` → `[B, 1, seqLen, C]` (single-head attention), then reshape back after.

### 5. Timestep Embedding — Sin/Cos Order and Frequency Divisor

**Deviation (sin/cos order)**: SD1.5 diffusers uses `flip_sin_to_cos=True` (default), producing `[cos, sin]` layout. Our code had `[sin, cos]`. Since every ResNet block conditions on the timestep embedding, this corrupted all noise predictions.

**Deviation (frequency divisor)**: Diffusers uses `/ (half_dim - 1)` in `get_timestep_embedding()`. Our code used `/ halfDim`, causing the highest frequency component to be ~6% off. This ensures the frequency range spans exactly `[1, 1/10000]`.

**Fix**: Swapped to `[cos, sin]` order and changed divisor to `(halfDim - 1)`.

### 6. Euler Scheduler — Missing scale_model_input

**Deviation**: Diffusers' `EulerDiscreteScheduler.scale_model_input` divides the latent by `sqrt(sigma^2 + 1)` before each UNet call. Without this, the UNet receives inputs at the wrong scale.

**Fix**: Added `ScaleModelInput(stepIndex)` to `IScheduler` interface and implemented in `EulerDiscreteScheduler`.

### 7. Euler Step — Division by Zero at Final Timestep

**Deviation**: At the final timestep (t=0), sigma approaches 0, causing division by zero in `derivative = (sample - pred_x0) / sigma`. The algebraic simplification for epsilon prediction eliminates this: `derivative = model_output` (the division cancels).

**Fix**: Simplified the epsilon-prediction path to avoid division. Added sigma guard for v-prediction path.

---

## CUDA Backend Deviations

### 8. CLIP Text Encoder — Missing Final LayerNorm

**Before**: `Encode()` returned raw last transformer layer output without applying `final_layer_norm`.
**After**: `Encode()` applies `final_layer_norm` matching HuggingFace `CLIPTextTransformer.forward()`.

**Impact**: Without this, text embeddings had std ~5 instead of ~1, causing 5x amplified conditioning signals that produced abstract patterns instead of coherent images.

### 9. CUDA SDPA Softmax — PTX Kernel

**Previous deviation**: The softmax step used a CPU roundtrip (download scores → host softmax → upload). Replaced with pure-PTX numerically stable per-row softmax using shared memory reductions (3-pass: max → exp+sum → normalize). One block per row, blockDim=256. Uses `ex2.approx.f32` for exp and `rcp.approx.f32` for 1/sum.

### 10. CUDA Conv2D — Im2Col + cuBLAS SGEMM (No cuDNN)

Conv2D is implemented via im2col (PTX kernel) + cuBLAS SGEMM, rather than cuDNN. Temporary column buffer allocated per forward pass. For 1x1 convolutions, im2col is skipped and input is used directly. Avoids cuDNN dependency, keeping the project pure C# + CUDA Driver API + cuBLAS.

### 11. CUDA GroupNorm/LayerNorm — Three-Pass Kernels

Both normalization kernels use a three-pass approach (mean → variance → normalize+affine) with shared memory reductions. Simpler to implement correctly than online Welford single-pass. Performance impact minimal since not bottleneck vs GEMM.

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

Write a test that loads Python's saved initial noise and text embeddings, runs the C# pipeline, and compares per-step latent statistics. This eliminates RNG differences and isolates model/scheduler bugs.

### Step 3: Single Forward Pass Comparison

If per-step stats diverge, isolate a single UNet forward pass. Save Python's step-0 inputs and outputs, feed the same inputs to C#, compare element-wise. If this diverges, the bug is in the model (not the scheduler).

### Step 4: Layer-by-Layer Binary Comparison

Hook every layer in Python and save outputs. Step through C# one layer at a time, comparing each output. This pinpoints the **first divergent layer**:
```
time_embedding:              avg_err=3.4e-8  (PERFECT)
conv_in:                     avg_err=0.0     (PERFECT)
down_blocks.0.resnets.0:    avg_err=3.5e-7  (PERFECT)
down_blocks.0.attentions.0: avg_err=0.127   ← FIRST DIVERGENCE
```

### Step 5: Sub-Layer Decomposition

Once the divergent layer is identified, manually execute each sub-operation in Python and save intermediates. For a CrossAttentionBlock: GroupNorm, reshape, proj_in, LayerNorm, Q/K/V projections, multi-head reshape, attention logits, softmax, output projection, residuals, FFN, etc. Compare C# sub-operations against these to find the exact bug.

### Step 6: Fix and Verify

Re-run layer-by-layer comparison to confirm all layers match (avg_err < 1e-3, ideally < 1e-4). Then run full pipeline comparison for end-to-end correctness.

---

## Lessons for Future Model Ports

### Attention Configuration is the #1 Trap

Every framework names attention parameters differently:

| Framework/Model | Parameter | Meaning |
|---|---|---|
| diffusers SD1.5 | `attention_head_dim=8` | **8 heads** (confusing! not head dim) |
| diffusers SDXL | `attention_head_dim=[5,10,20]` | Per-block head counts |
| Some configs | `num_heads=8` | 8 heads (clear) |
| Some configs | `head_dim=64` | 64-dim per head (clear) |

**Always verify** by printing `model.attn.heads` and checking Q/K/V reshape shapes in Python before writing C# config code.

### Weight Shape vs Usage Mismatches

A `proj_in.weight` with shape `[320, 320, 1, 1]` is a Conv2d but equivalent to linear for 1x1 kernels. For non-1x1 kernels, im2col/GEMM must be used. Always check weight shape.

### GELU Variant Differences

C# uses tanh-approximated GELU. PyTorch default `F.gelu` uses exact erf-based GELU. Difference is ~1e-4 and acceptable, but check which variant the reference uses if FFN diverges.

### RNG Differences

C# Box-Muller vs PyTorch algorithm. Same seed = different noise. Always compare with **shared noise tensors**, never by matching seeds.

### Expected FP32 Tolerances

| Layer type | Expected avg_err |
|---|---|
| Element-wise (Add, SiLU) | < 1e-7 |
| GroupNorm, LayerNorm | < 1e-6 |
| Linear/Conv (GEMM) | < 1e-5 |
| Full attention block | < 1e-4 |
| Full UNet/DiT pass | < 1e-3 |

If a layer exceeds these by 10x+, there's a real bug — not FP noise.

### Diagnostic Script Inventory

All scripts in `tests/python-reference/` using venv at `tests/python-reference/.venv/`:

| Script | Purpose |
|---|---|
| `dump_reference_stats.py` | Full pipeline: noise, embeddings, per-step latents, final output |
| `dump_layer_outputs.py` | Per-layer model outputs with index.json |
| `dump_attn_sublayers.py` | Sub-operation breakdown of first CrossAttentionBlock |
| `compare_layers.py` | Utility for comparing binary tensor files |

# Phase 3 — Deviations from Design Plan

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

## 3. UNet Attention Head Count — headDim Passed as numHeads

**Design assumption**: `UNetConfig.AttentionHeadDim` values would be passed as the head dimension when constructing attention blocks.

**Deviation**: `AttentionHeadDim[i]` (value: 8 for all SD1.5 blocks) was passed directly as `numHeads` instead of computing `numHeads = channels / headDim`. This gave 8 heads everywhere instead of 40/80/160/160 heads for the 320/640/1280/1280 channel blocks.

**Fix**: Computed `int numHeads = outCh / config.AttentionHeadDim[i]` for down blocks, mid block, and up blocks.

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

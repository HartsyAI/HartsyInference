# Lens + Lance generation-performance pass — 2026-07-16

Hardware: RTX 4090 24 GB (CUDA device 0, driver 580.159.03, CUDA 13.0), .NET 10.0.9, Release build.
Branch: `lens-lance-perf-0716` (base `landing-0716` @ fc2034d). All runs seed 42 via the existing
`LensGenerationTests` / `LanceGenerationTests` harnesses.

Both pipelines were brought up 07-15/16 with reference parity corr ~1.0 but ran **host-bound**: the
per-step path was full of host glue (host RoPE loops, host gather/scatter role routing, host head
permutes / modulation / gated-residual loops), each `DataPointer` read a full-stream D2H drain. This
pass ports the per-step path to existing `IBackend` GPU ops (no new kernels), following the proven
QwenImageBlock / ChromaDoubleStreamBlock recipe.

## Results

| Model | Config | Metric | Before | After | Speedup |
|---|---|---|---|---|---|
| Lance 3B T2I | 768², 20 steps, cfg 4 (interval-gated) | s/step (CFG, 2 forwards) | 12.8 s | 0.33 s | **39×** |
| Lance 3B T2I | 768², 20 steps | s/step (cond-only tail steps) | 7.1 s | 0.17 s | 42× |
| Lance 3B T2I | 768², 20 steps | generate total (incl. preload + VAE) | 259.8 s | 21.9 s | **12×** |
| Lens Turbo | 1024², 8 steps, cfg 1 | s/step (single forward) | 14.4 s | 0.55 s | **26×** |
| Lens Turbo | 1024², 8 steps | denoise+decode wall (excl. TE encode) | ~124 s | ~6.8 s | 18× |
| Lens Turbo | 1024², 8 steps | full gen incl. GPT-OSS TE encode | 201.6 s | 84.7 s | 2.4× |

Parity gates (fixed seed 42 vs pre-change baseline, baseline rebuilt from `fc2034d` in a scratch
worktree):

| Model | Gate | Corr | Notes |
|---|---|---|---|
| Lens Turbo | single-forward velocity DebugDump, real inputs (512², step 1 — identical inputs both sides) | **0.999973** | passes ≥0.9999 |
| Lens Turbo | 8-step final image (1024²) | 0.999848 | accumulated TF32-GEMM re-rounding (ulp-changed inputs into 10-bit-mantissa GEMMs); neighbor-corr identical to baseline (0.9930), visually equivalent |
| Lance 3B | 20-step final image (768²) | **0.999950** | passes ≥0.9999 end-to-end; residual = F16 fused SDPA flip + reduction order |

All 73 Lens/Lance unit tests + 5 Lance video tests pass unchanged (the synthetic-weights backbone
tests exercise the rewritten blocks through the CPU backend's reference op implementations). A new
env-gated `LensForwardProbeTests` (GpuIntegration) reproduces the single-forward layer dump used
for the before/after diff.

## Root causes (before) and fixes

### Lens (`LensTransformerBlock` / `LensTransformer` / `LensPipeline`)
1. **Host RoPE per block** — `LensRope.ApplyRotationBatched` rotated Q and K via `DataPointer` host
   loops, twice per block × 48 blocks × steps, rebuilding the trig tables every call.
   → `LensRope.GetOrBuildJointTables` (cached `[S, headDim]` duplicated-pair tables, `[img, txt]`
   joint order) + `IBackend.WanRopeInterleaved` device kernel on the pre-permute layout.
2. **Host reshape/concat/split/modulation/gated-residual glue** — `DiTUtils.ReshapeToMultiHead` /
   `ConcatAlongSeqDimMultiHead` / `SplitAlongSeqDimMultiHead`, `QkNorm.Forward`,
   `AdaLNModulation.ApplyModulation/ApplyGatedResidual` were all host loops.
   → `backend.RmsNorm` (QK-norm on `[B,S,H,D]` views), `backend.Concat`/`SliceRows` (joint
   seq concat/split), `backend.Permute0213`, `DiTUtils.Modulate` (AddScalar +
   AffineBroadcastLastDim), `backend.GatedResidualLastDim` — the QwenImageBlock recipe.
3. **Text stack recomputed every forward** — per-layer `txt_norm` RMSNorm + host channel-concat +
   `txt_in` Linear ran per step on step-invariant encoder captures.
   → computed once per prompt (2-entry ref-keyed cache in `LensTransformer`), block loop gets a
   device copy.
4. **Host final layer** — `DiTUtils.LayerNormNoAffine` + scale/shift modulation host loop per
   forward → `backend.LayerNormNoAffine` + `SliceRows` chunking + `DiTUtils.Modulate`.
5. **No DiT/VAE preload** — added `PreloadWeights(transformer)` before the loop and
   `PreloadWeights(vaeDecoder)` before decode (PreloadWeights/FreeWeights symmetry).
6. SDPA stays **F32** for Lens: the F16 fused path was tried and NaN'd from block 36 on — Q/K are
   QK-RMSNormed (bounded) but **V is not**, and Lens's undamped residual stream runs hot enough
   (rms up to ~1e8-1e9) that an F16-cast V overflows 65504. Probe-verified (layer dumps went
   full-NaN at block_36 with `allowF16: true`, clean without).

### Lance (`LanceMoTBlock` / `LanceTransformer`)
1. **Host gather/scatter role routing** — und/gen token subsets were gathered and scattered via
   host memcpys ~16× per block × 36 blocks (each a stream drain). The packed layout is always
   und-prefix / gen-run / und-tail (validated once per forward), so routing is now contiguous
   `SliceRows` / `Concat` device ops per role segment.
2. **Host RoPE / head permutes / GQA expand / residual adds** — `Multimodal3DRope.ApplyRotary`,
   `PermuteToBhsd`, `ExpandKvToBhsd`, `PermuteFromBhsd`, `AddRows` host loops →
   `backend.ApplyRopeSingle` (rotate-half kernel; per-tensor so the 16:2 GQA head mismatch is
   safe), `backend.Permute0213`, `backend.RepeatKvHeads`, `backend.Add`.
3. **Step-invariant state rebuilt per forward** — M-RoPE cos/sin (`BuildCosSin`), the text-segment
   embeds, and the gathered `latent_pos_embed` rows now build once per prompt (2-entry ref-keyed
   caches keyed on the caller's arrays/tensors).
4. **Packed-sequence build on host** — text-embed writes + host add of pos-embed/timestep → device
   `Linear` + `Add` + `AffineBroadcastLastDim` (ones-scale broadcast of the timestep embed) +
   3-segment `Concat`.
5. SDPA `allowF16: true` with the additive sparse mask riding the cuDNN fused engine as an fp32
   bias score-modifier (mask value −1e9 is finite, F16-safe by construction).

## Remaining / deferred
- Lance ~0.165 s/forward is near the F32-activation GEMM floor for a 3B dense-equivalent forward
  over ~2360 tokens on the 4090; further wins would need an F16 activation path or CUDA-graph step
  capture (deferred — out of scope for this pass).
- Lens Turbo total is now dominated by the GPT-OSS-20B MoE text encode (~78 s, once per prompt) and
  pipeline wiring/load, not the denoise loop. TE perf was explicitly out of scope (encode-once).
- Lens SDPA is the F32 path (materialized/tiled GEMM); an F16-safe fused path would need V
  normalization awareness or a per-block V-scale — deferred.
- Host-side per-step leftovers (one D2H drain per step each): scheduler step / Euler step / CFG
  renorm read the velocity on host. Same pattern as every other pipeline; not measurable at
  current step times.
- Lens end-of-pipeline host unpack/BN/unpatchify loops run once per generation (~tens of ms).

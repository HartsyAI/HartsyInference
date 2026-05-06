# T5-XXL Memory Strategy for Consumer GPUs

> **Audience:** Pipeline authors choosing the T5 dtype/eviction policy for Flux, SD3, SD3.5, AuraFlow, Chroma.
> **Cross-references:** [`QUANTIZATION_DIFFUSION.md`](QUANTIZATION_DIFFUSION.md), [`CUDA_PERFORMANCE.md`](CUDA_PERFORMANCE.md), [`PHASE_3_DEVIATIONS.md`](../Checklists/PHASE_3_DEVIATIONS.md) §18 / §33.

---

## TL;DR

On a 12 GB consumer GPU running Flux Dev / SD3.5 / Chroma:

1. **Load T5-XXL in FP8** when the checkpoint ships fp8_scaled (Flux Dev FP8, SD3.5 fp8_scaled). Folds `.scale_weight` into `Tensor.Fp8ScaleFactor`; the existing CudaBackend cast-to-F16-per-GEMM path handles inference.
2. **Cast T5 to FP8 at load** if the checkpoint ships at FP16 and you can't fit it. Round-trip cost ~5 ms/MB on CPU at load time, zero cost at inference.
3. **Always evict T5 before the transformer / VAE step.** Call `_backend.Sync(); _backend.FreeWeights(_t5.EnumerateWeights())` after the encode finishes. Without this, T5 (4.8–9.5 GB) and the transformer (8–17 GB at FP8) both sit in VRAM at the same time and OOM the GPU.
4. **Q8_0 GGUF T5** is the future option once the GGUF K-quant reader lands. Same VRAM footprint as FP8 but slightly better quality preservation (block-wise scale + zero-point).

VAE is **never** quantized below FP16. It's small (~335 MB) and FP8 VAE produces visible artifacts.

---

## VRAM Budget Table

T5-XXL is the encoder used by Flux, SD3.x, AuraFlow (Pile-T5-XL is structurally identical, smaller), Chroma, and any future MMDiT that follows the SD3 template. Sizes are for the encoder alone (no decoder layers — these are encoder-only configs).

| Format | Bytes/param | T5-XXL (4.7B) | Pile-T5-XL (1.2B) | Notes |
|---|---|---|---|---|
| F32 | 4 | 18.8 GB | 4.8 GB | Reference precision; no consumer GPU fits this + transformer |
| BF16 | 2 | 9.4 GB | 2.4 GB | Native checkpoint shipping format for diffusers fp16 builds |
| F16 | 2 | 9.4 GB | 2.4 GB | Same on-disk size as BF16; we cast BF16→F16 at load for GPU GEMM |
| F8 (E4M3) | 1 | 4.7 GB | 1.2 GB | Standard for Flux Dev FP8 / SD3.5 fp8_scaled. CudaBackend casts to F16 per GEMM on Ampere |
| Q8_0 GGUF | ~1.06 | 5.0 GB | 1.3 GB | Block-wise scale, blocks of 32. Better quality than naive FP8 |
| Q4_K GGUF | ~0.55 | 2.6 GB | 660 MB | 6-bit super-block + 4-bit nibbles. Visible quality drop on T5; only viable for large-style prompts |

Rule of thumb: **T5 at FP8 ≈ T5 at Q8_0 ≈ ~5 GB** for the XXL model. Anything below that costs measurable quality.

---

## Per-Pipeline Strategy on 12 GB GPU

### Flux Dev / Schnell (12B transformer)

| Component | Format | VRAM | Reside during |
|---|---|---|---|
| T5-XXL | FP8 | 4.7 GB | T5 encode only |
| CLIP-L | F16 | 250 MB | CLIP encode only |
| Flux transformer | FP8 (Flux Dev FP8 ckpt) | ~6 GB | Denoise loop |
| Flux VAE | F16 | 84 MB | Decode only |

Sequence: load T5+CLIP → encode → `Sync()` + `FreeWeights(t5+clipL)` → load transformer → denoise → `Sync()` + `FreeWeights(transformer)` → load VAE → decode. Peak VRAM at any one stage ~6.5 GB. Works comfortably on 12 GB.

### SD3.5 Medium / Large (8.1B / 24B transformer)

Same pattern as Flux. SD3.5 Medium at FP8 + T5 at FP8 fits 12 GB; SD3.5 Large needs FP8 transformer + Q8_0 T5 (or block streaming).

The fp8_scaled SD3.5 Medium checkpoint already ships with `.scale_weight` companions for both transformer and T5; load with `CheckpointConvertUtils.ApplyFp8ScaledDequant` and the scale factors fold into `Tensor.Fp8ScaleFactor` automatically.

### Chroma (12B transformer, T5-only — no CLIP)

Chroma is T5-XXL only — no CLIP-L pooled path. This actually helps fit: you save ~250 MB of CLIP weight that other Flux-class pipelines hold.

| Component | Format | VRAM |
|---|---|---|
| T5-XXL | FP8 | 4.7 GB |
| Chroma transformer | FP8 | ~6 GB |
| Flux VAE | F16 | 84 MB |

Same eviction discipline: T5 stays for encode only.

### AuraFlow (6.8B transformer + Pile-T5-XL)

Pile-T5-XL is much smaller than T5-XXL (1.2 GB at F16 / 0.6 GB at FP8). AuraFlow can keep Pile-T5 resident throughout if desired, though eviction is still cheaper:

| Component | Format | VRAM |
|---|---|---|
| Pile-T5-XL | FP8 | 660 MB |
| AuraFlow transformer | FP8 | ~3.4 GB |
| SDXL VAE | F16 | 167 MB |

Total ~4.3 GB — comfortable on 12 GB even without aggressive eviction. AuraFlow is the easiest of these four to deploy.

---

## Loading Patterns

Three patterns exist for getting T5 weights into VRAM at the right precision:

### Pattern A: Native FP8 Checkpoint (Flux Dev FP8, SD3.5 fp8_scaled)

The on-disk format already is FP8. `SafeTensorsLoader` returns tensors with `DType.F8E4M3`. `CheckpointConvertUtils.ApplyFp8ScaledDequant` walks the dict, finds `.weight` + companion `.scale_weight`/`.weight_scale`, folds the scalar into `Tensor.Fp8ScaleFactor`, and removes the companion. Subsequent GEMM dispatch in `CudaBackend` reads `Fp8ScaleFactor` and binds it as the cuBLAS `alpha` parameter.

No casting needed. Load time is just mmap + companion folding (~1 second for T5-XXL).

### Pattern B: Cast FP16/BF16 to FP8 at Load

For checkpoints that ship at FP16 (most diffusers builds before fp8_scaled became standard), apply a CPU-side cast at load:

```csharp
foreach (string key in t5Weights.Keys.ToList())
{
    Tensor src = t5Weights[key];
    if (src.DType == DType.BF16 || src.DType == DType.F16)
    {
        Tensor dst = new Tensor(src.Shape, DType.F8E4M3);
        src.CastTo(dst);
        t5Weights[key] = dst;
        src.Dispose();
    }
}
```

CPU cost: ~5 ms/MB on a modern CPU. For T5-XXL (9.4 GB BF16) this is ~50 seconds added to model load. It's a one-time cost; subsequent generations from the same loaded model don't re-pay it.

### Pattern C: Q8_0 GGUF (FUTURE — needs GGUF K-quant reader)

GGUF Q8_0 reader is not yet implemented in `SharpInference.ModelHandler`. When it lands:
- Load T5 weights via `GgufLoader.Load(path)` instead of `SafeTensorsLoader`
- Each block-32 group dequantizes to F16/F32 on-demand at GEMM time
- Slightly better quality than FP8 (block-wise scale + tighter range), same VRAM footprint

Q4_K is **not recommended** for T5 even when the reader lands — see "Why Not Q4 on T5" below.

---

## Eviction Discipline (Critical)

The biggest VRAM footgun is leaving T5 resident through the denoise loop. T5 weights are not used after the encode step; keeping them in VRAM means the transformer can't fit alongside.

The pattern (used in `Sd3Pipeline`, `FluxPipeline`, `Sd35Pipeline`, `AuraFlowPipeline`):

```csharp
// 1. Encode prompt with T5 (and CLIP if applicable)
Tensor t5Embeddings = _t5.Encode(_backend, tokens);
// ... CLIP encode etc ...

// 2. Evict text encoders BEFORE transformer denoise
_backend.Sync();                                       // Drain pending GPU ops
_backend.FreeWeights(_t5.EnumerateWeights());          // Free T5 weight cache
_backend.FreeWeights(_clipL.EnumerateWeights());       // Free CLIP if applicable

// 3. Now run transformer denoise (T5 VRAM freed; transformer fits)
for (int step = 0; step < numSteps; step++) { /* denoise */ }

// 4. Evict transformer BEFORE VAE decode
_backend.Sync();
_backend.FreeWeights(_transformer.EnumerateWeights());

// 5. VAE decode (transformer VRAM freed; VAE fits)
Tensor image = _vae.Decode(_backend, latent);
```

Without `Sync()` before `FreeWeights()`, GPU ops queued in flight may still be reading from the freed memory — undefined behavior, often surfaces as silent garbage outputs or `CUDA_ERROR_INVALID_VALUE` at the next dispatch. See [`PHASE_3_DEVIATIONS.md`](../Checklists/PHASE_3_DEVIATIONS.md) §18.

---

## Why Not Q4 on T5

T5's prompt embeddings drive **spatial layout** in MMDiT pipelines through cross-attention to image tokens. Quality loss in T5 produces compounding errors:

- Q4_K T5 + FP8 transformer: prompts with fine-grained spatial language ("a cat *to the left of* a dog") often misplace subjects
- Q4_K T5 alone shows up as muddier text rendering and weaker prompt adherence on long captions
- Tested on Flux Dev: Q4_K T5 dropped MS-COCO CLIP-score by ~3% vs FP8; FP8→FP8 preserves it within noise

[`QUANTIZATION_DIFFUSION.md`](QUANTIZATION_DIFFUSION.md) § "Component-Level Sensitivity" documents the underlying mechanism: encoder-only transformers compound error across all 24 layers, with no later layers to correct earlier ones (unlike a denoiser that has 28+ steps to wash out small errors).

**Decision rule:** if you can't fit FP8 T5 + FP8 backbone, drop the **backbone** to Q4 before dropping T5 below FP8.

---

## VAE Always FP16 Minimum

VAE decoders are small (84–335 MB) and quality-critical: they map a `[B, C, H, W]` latent through ~8× upsampling Conv2D blocks. FP8 GroupNorm in particular produces visible posterization on smooth gradients (skies, skin tones).

Pipelines should **assert** the VAE dtype at construction. If a `QualityProfile` is wired through, reject any preset that places VAE below FP16 with a `SharpInferenceException`. See [`QualityProfile.cs`](../../src/SharpInference.Diffusion/Quality/QualityProfile.cs).

---

## Decision Tree

```
Need to fit T5 + transformer on 12 GB GPU?
├── Checkpoint is fp8_scaled? → use Pattern A (no cast cost)
├── Checkpoint is FP16/BF16?
│   ├── Have time at load (50s acceptable)? → Pattern B (cast at load)
│   └── Want better quality + GGUF reader available? → Pattern C (Q8_0 GGUF)
└── Always: Sync() + FreeWeights() between stages
```

---

## Future Work

- **GGUF K-quant reader** (Q8_0, Q4_K, Q5_K) — common blocker called out in [`PHASE_4_MODEL_BREADTH.md`](../Checklists/PHASE_4_MODEL_BREADTH.md) §5b. Unlocks Pattern C for T5 plus smaller variants of all DiT backbones.
- **Block streaming for T5** — currently we load all 24 layers up front. A streaming pattern (load layer N, run, free layer N) would let T5-XXL run in ~400 MB peak. PCIe-bound at ~1 second extra per encode; only worth it on 8 GB GPUs.
- **Native FP8 GEMM** — Ada+ tensor cores (SM 8.9+) can run FP8 GEMM directly with per-tensor scale via `cublasLtMatmul`. Ampere falls back to cast-to-F16. Ampere fallback is already wired; Ada path is documented in [`CUDA_PERFORMANCE.md`](CUDA_PERFORMANCE.md).

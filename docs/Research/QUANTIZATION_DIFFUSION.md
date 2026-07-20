# Quantization for Diffusion — Research Notes

## Summary

Quantization enables large diffusion models to run on consumer GPUs with limited VRAM. Flux.1-dev at FP16 requires ~24 GB VRAM; Q8_0 reduces this to ~12 GB while maintaining 98-99% of visual quality. The key finding is that **mixed-precision is essential**: the UNet/DiT backbone tolerates Q8_0 (and even Q4_K in transformer architectures) with minimal degradation, but the VAE decoder and text encoders are significantly more sensitive. The recommended strategy for HartsyInference is: **Q8_0 for the denoising backbone + FP16 for VAE + FP16 or FP8 for text encoders**, with GGUF as the on-disk format and on-demand dequantization during forward passes. Q4 quantization is usable for DiT-based models (Flux, SD3) but causes visible artifacts in UNet-based models (SD1.5, SDXL) and should be offered as an option with appropriate warnings. Q2 is not viable for any diffusion component.

## Detailed Findings

### 1. Component-Level Quantization Sensitivity

Diffusion pipelines consist of multiple components with very different sensitivity to quantization:

**Denoising Backbone (UNet or DiT)**
- Primary target for quantization — consumes >95% of inference compute and the vast majority of parameters.
- DiT/transformer-based models (Flux, SD3, SD3.5) tolerate quantization significantly better than UNet-based models (SD1.5, SDXL). This is because transformers use linear + attention layers (amenable to scale-based quantization), while UNets rely heavily on Conv2D (where spatial correlations make quantization harder).
- Q8_0 is "basically indistinguishable from the original FP16" for both UNet and DiT architectures ([stduhpf SD3.5 GGUF](https://huggingface.co/stduhpf/SD3.5-Large-GGUF-mixed-sdcpp), [city96 FLUX.1-dev-gguf](https://huggingface.co/city96/FLUX.1-dev-gguf/discussions/15)).
- Q5_K_S for DiT models shows "barely noticeable quality loss from Q8" — recommended for low-VRAM scenarios.
- Q4_K_S for DiT is acceptable; Q3_K_S is "amazing that it maintains its quality" but shows slight roughness.
- Q2_K is "awful" — severe degradation, nobody should use it ([city96 discussion](https://huggingface.co/city96/FLUX.1-dev-gguf/discussions/15)).
- For UNet models (SDXL), Conv2D-heavy architectures are harder to quantize. sd.cpp originally noted "quantization wasn't feasible for regular UNET models (conv2d)" — this has improved but Conv2D layers remain more sensitive than linear layers.

**VAE Decoder**
- The VAE is extremely sensitive to quantization. It converts latent space back to pixel space, and any quantization error directly manifests as visible artifacts (color shifts, blocking, detail loss).
- **No GGUF quantized VAE models are readily available** — the community keeps VAEs in FP32, FP16, or BF16 ([ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF), [city96 GGUF issue #406](https://github.com/city96/ComfyUI-GGUF/issues/406)).
- The VAE is small (168 MB for Flux, ~160 MB for SDXL) so quantizing it yields negligible memory savings while risking visible quality loss.
- **Recommendation: Always keep VAE in FP16 or FP32.** Never quantize.

**Text Encoders (CLIP, T5)**
- CLIP-L is small (~246 MB for Flux) — keep in FP16; quantization saves little and risks prompt adherence degradation.
- CLIP-G (used by SDXL, SD3) — also relatively small, keep in FP16 or at most FP8.
- T5-XXL (used by Flux, SD3) is large (~9.5 GB in FP16) and is a legitimate quantization target. FP8 is commonly used with minimal quality impact. Q4_K has been used successfully in community mixed-quant models ([stduhpf SD3.5 GGUF](https://huggingface.co/stduhpf/SD3.5-Large-GGUF-mixed-sdcpp)).
- For Stable Diffusion 3 with 3 text encoders: quantize TE1 (CLIP-L) and TE3 (T5-XXL) but do NOT quantize TE2 (CLIP-G OpenCLIP) — causes quality issues ([Quanto diffusers blog](https://huggingface.co/blog/quanto-diffusers)).

### 2. SDXL vs Flux Quantization Response

| Aspect | SDXL (UNet) | Flux (DiT) |
|--------|-------------|------------|
| Architecture | UNet with Conv2D + ResNet + Attention | MMDiT + SingleDiT (pure transformer) |
| Parameters | ~2.6B (UNet) | ~12B (transformer) |
| FP16 VRAM | ~6.5 GB | ~24 GB |
| Q8_0 quality | ~99% of FP16 | ~99% of FP16 |
| Q4 quality | Noticeable degradation, artifacts in fine detail | Acceptable, 90%+ quality retention |
| Quantization compatibility | Conv2D tensors resist K-quants | Linear layers work well with all quant types |
| K-quant support | Full (tensor shapes match 256-block superblocks) | SD3.5 Large: ~90% of tensors incompatible with K-quants due to shape mismatch |
| Best low-VRAM quant | Q8_0 (Q4 risky) | Q5_K_S or Q4_K_S |

Key difference: DiT models like Flux use predominantly linear layers and attention, which have well-behaved weight distributions amenable to symmetric quantization. UNet models have Conv2D kernels with spatial structure that makes quantization harder — the spatial correlations in convolutional filters mean more information is lost per quantization step.

### 3. Post-Load vs Pre-Quantized Loading Strategies

**Strategy A: Pre-quantized GGUF (Recommended for HartsyInference)**
- Weights stored on disk in quantized GGUF format (Q8_0, Q4_K_M, etc.).
- During inference, each tensor is dequantized on-demand to the compute dtype (FP16 or BF16) for each forward pass.
- Advantages: Fast loading, small disk footprint, no quantization step at startup.
- Disadvantage: Repeated dequantization adds overhead per forward pass (each tensor dequantizes to bfloat16 for each use).
- This is what sd.cpp and ComfyUI-GGUF use.
- sd.cpp conversion: `./bin/sd-cli -M convert -m model.safetensors -o model.q8_0.gguf -v --type q8_0`

**Strategy B: Load FP16 then quantize in-memory (Quanto/torchao approach)**
- Load full FP16 weights from safetensors, then quantize at runtime.
- Used by diffusers with Quanto: `quantize(pipeline.transformer, weights=qfloat8); freeze(pipeline.transformer)`
- Advantages: Works with any safetensors model, no conversion step needed.
- Disadvantage: Requires full FP16 model to fit in RAM during load, slow startup.
- Can save quantized checkpoints for reuse.

**Strategy C: Mixed-format loading**
- Load different components from different files at different precisions.
- sd.cpp supports separate `--diffusion-model`, `--clip_l`, `--clip_g`, `--t5xxl`, `--vae` flags.
- Example: Q4_K DiT + FP16 VAE + Q4_K T5 + FP16 CLIP.
- **This is the recommended approach for HartsyInference** — maximum flexibility.

### 4. Layer-Level Sensitivity Analysis

Research from Qua2SeDiMo (2024) and ViDiT-Q (ICLR 2025) provides detailed layer sensitivity data:

**Most Sensitive (keep higher precision):**
- Time embedding layers (t-Embed) — identified as crucial bottleneck; large activation ranges and outliers.
- Adaptive normalization inputs in Flux — require 16-bit precision ([SVDQuant](https://arxiv.org/html/2411.05007v1)).
- Cross-attention key-value projections — sensitive to quantization, critical for text-image alignment.
- Final output projection layers (`proj_out`) — excluding from INT4 quantization significantly improves quality.

**Moderately Sensitive:**
- Self-attention Q/K/V projections — moderate sensitivity, benefits from Q8_0 over Q4.
- Spatial attention layers — visual quality is "primarily influenced by spatial attention and FFN layers."

**Least Sensitive (can quantize aggressively):**
- FFN/MLP intermediate layers — tolerate Q4_K well, especially the second MLP layer in MMDiT blocks.
- Condition embedding layers — "less crucial" for maintaining fidelity in PixArt models.
- Skip connections / residual paths — quantizable but need careful handling of accumulation.

**Architecture-dependent preferences:**
- UNet models prefer Uniform Affine Quantization (UAQ) — scale-based methods.
- DiT models show "a slight preference for K-Means-based quantization" (clustering methods).

### 5. Q4 Quantization Assessment

Q4 quantization for diffusion models is **not universally "too lossy"** but requires careful application:

- **DiT models (Flux, SD3):** Q4_K_S is viable. SVDQuant achieves W4A4 on FLUX.1-dev with FID 19.9 vs FP16 baseline FID 20.3 — actually slightly better ([SVDQuant paper](https://arxiv.org/html/2411.05007v1)). Community reports Q4_1 delivers "outstanding quality" on Flux even on a GTX 1070ti.
- **UNet models (SDXL):** Q4 shows more degradation. SDXL W4A4 FID increases from 16.6 (FP16) to 19.0-20.7 (quantized). Usable but noticeably different.
- **SD1.5:** Q4 is marginal. Research suggests 3-bit uniform quantization causes "extremely challenging" FID degradation (jumping from ~100 to 170+ FID).
- **For HartsyInference:** Offer Q4_K as an option for DiT models with a note that quality is reduced. For UNet models, Q8_0 should be the minimum recommended.

### 6. Quality Metrics

Formal FID/CLIP score comparisons from published research:

| Model | Precision | FID | Notes |
|-------|-----------|-----|-------|
| FLUX.1-dev | FP16 | 20.3 | Baseline (50 steps) |
| FLUX.1-dev | W4A4 INT | 19.9 | SVDQuant — slightly better than baseline |
| FLUX.1-dev | W4A4 FP | 21.0 | SVDQuant |
| SDXL-Turbo | FP16 | 16.6 | Baseline (30 steps) |
| SDXL-Turbo | W4A4 INT | 20.7 | SVDQuant |
| SDXL-Turbo | W4A4 FP | 19.0 | SVDQuant |
| PixArt-alpha | FP16 baseline | 99.67 FID | Reference |
| PixArt-alpha | W4 (various) | 97-98 FID | Qua2SeDiMo — minimal degradation |
| PixArt-alpha | W3 UAQ | 172.51 FID | Severe degradation at 3-bit |
| LDM-4 (ImageNet) | W8A8 | Lossless | PTQ4DiT |
| LDM-4 (ImageNet) | W4A8 | Lossless | PTQ4DiT |

Community consensus on visual quality retention:
- Q8_0: 98-99% quality (essentially imperceptible)
- Q6_K: ~97% quality
- Q5_K_S: ~95% quality (barely noticeable)
- Q4_K_S: ~90% quality (slight roughness, acceptable)
- Q3_K_S: ~85% quality (noticeable on detailed prompts)
- Q2_K: ~70% quality (unacceptable)

## Key Numbers/Constants

| Constant | Value | Context |
|----------|-------|---------|
| Q8_0 block size | 32 elements | GGML standard |
| Q8_0 block struct size | 34 bytes | 2 (fp16 scale) + 32 (int8 quants) |
| Q4_0 block size | 32 elements | GGML standard |
| Q4_0 block struct size | 18 bytes | 2 (fp16 scale) + 16 (4-bit packed) |
| K-quant superblock size | 256 elements | Required for Q4_K, Q5_K, Q6_K etc. |
| Flux.1-dev FP16 VRAM | ~24 GB | Full model |
| Flux.1-dev Q8_0 VRAM | ~12 GB | ~50% reduction |
| Flux.1-dev Q4 VRAM | ~6-8 GB | ~67-75% reduction |
| SDXL FP16 VRAM | ~6.5 GB | Full model |
| SD1.5 FP16 VRAM | ~2.3 GB | With flash attention: ~1.9 GB |
| SD1.5 Q8_0 VRAM | ~2.1 GB | With flash attention: ~1.6 GB |
| Flux transformer params | ~12B | MMDiT + SingleDiT |
| Flux T5-XXL size | ~9.5 GB FP16 | Dominant text encoder |
| Flux CLIP-L size | ~246 MB | Small — keep FP16 |
| Flux VAE size | ~168 MB | Small — always keep FP16 |
| SDXL UNet params | ~2.6B | |
| Q8_0 perplexity increase | ~0.01 points | vs FP16 baseline |
| ViDiT-Q W8A8 memory saving | 2-2.5x | vs FP16 |
| ViDiT-Q W8A8 speedup | 1.4-1.7x | End-to-end latency |
| SVDQuant memory reduction | 3.5x | For 12B Flux on laptop 4090 |
| SVDQuant speedup vs NF4 | 3.0x | Weight-only quantization baseline |

## Data Layouts/Formats

### Q8_0 Block Layout (GGML)

```
Block size: 32 weights
Total bytes per block: 34

struct block_q8_0 {
    ggml_fp16_t d;       // 2 bytes: scale factor (float16)
    int8_t      qs[32];  // 32 bytes: quantized values (-128 to 127)
};
// Total: 34 bytes per 32 weights = 8.5 bits/weight
```

Dequantization: `weight[i] = qs[i] * d` (symmetric, no zero-point offset).

### Q4_0 Block Layout (GGML)

```
Block size: 32 weights
Total bytes per block: 18

struct block_q4_0 {
    ggml_fp16_t d;       // 2 bytes: scale factor (float16)
    uint8_t     qs[16];  // 16 bytes: 32 x 4-bit values packed in pairs
};
// Total: 18 bytes per 32 weights = 4.5 bits/weight
```

Dequantization: Extract 4-bit nibbles (low and high), subtract 8 (zero-point offset of 8), multiply by scale `d`.

### K-Quant Superblock Layout

K-quants use 256-element superblocks for better accuracy. This is critical for diffusion models:
- SD3.5 Large: ~90% of tensors have shapes incompatible with the 256-element superblock requirement.
- Solution: mixed-quantization (K-quants for compatible tensors, legacy quants for rest).

### Mixed-Precision GGUF File Structure

sd.cpp supports separate GGUF files per component:
```
model/
  diffusion-model-q8_0.gguf    # DiT/UNet in Q8_0
  clip_l-fp16.gguf              # CLIP-L in FP16
  clip_g-fp16.gguf              # CLIP-G in FP16 (SDXL/SD3)
  t5xxl-fp8.gguf                # T5-XXL in FP8 or Q4_K
  vae-fp16.gguf                 # VAE always in FP16
```

## Algorithm Steps

### Post-Load Dequantization (GGUF inference path)

1. Load GGUF file, parse header and tensor metadata.
2. Memory-map tensor data (stays in quantized format on disk/RAM).
3. For each forward pass:
   a. For each tensor needed by the current layer:
      - Read quantized block data.
      - Dequantize to compute dtype (FP16/BF16) using block-specific formula.
      - Perform matmul/conv operation at compute precision.
      - Discard dequantized values (do not cache — saves memory).
4. Repeat for all denoising steps (typically 20-50 steps).

### In-Memory Quantization (Quanto-style)

1. Load model weights in FP16 from safetensors.
2. For each target module (linear layers, attention projections):
   a. Compute per-channel or per-block scale: `scale = max(abs(weights)) / 127` (for INT8).
   b. Quantize: `q = round(weights / scale)`, clamp to [-128, 127].
   c. Store `q` (int8) and `scale` (fp16) — free original fp16 tensor.
3. During forward pass: dequantize on-the-fly as `weights = q * scale`.
4. Optionally serialize quantized state for future fast loading.

### Mixed-Precision Component Loading

1. Parse model configuration to identify components (backbone, VAE, text encoders).
2. For each component, check user-specified or default precision:
   - Backbone: Load GGUF at specified quant level (default Q8_0).
   - VAE: Always load FP16 (override any user Q4/Q8 request with warning).
   - CLIP: Load FP16 (small enough that quantization is unnecessary).
   - T5-XXL: Load at FP8 or Q4_K if available, FP16 otherwise.
3. Allocate each component to device (GPU preferred, CPU fallback with offloading).
4. Run inference with mixed dtypes — dequantize backbone tensors per-layer, run VAE/encoders natively in FP16.

## Reference Implementations

### stable-diffusion.cpp (sd.cpp)
- **Repository:** [github.com/leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
- Pure C/C++ inference for SD, Flux, Wan, Qwen Image, and more.
- Supports quantization types: f32, f16, q8_0, q5_0, q5_1, q4_0, q4_1, plus K-quants (q2_k through q6_k).
- Conversion command: `./bin/sd-cli -M convert -m model.safetensors -o model.q8_0.gguf -v --type q8_0`
- Separate CLI flags for each component: `--diffusion-model`, `--clip_l`, `--clip_g`, `--t5xxl`, `--vae`.
- Tensors loaded on-demand, optionally quantized for memory efficiency.
- **Docs:** [quantization_and_gguf.md](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/quantization_and_gguf.md)

### ComfyUI-GGUF (city96)
- **Repository:** [github.com/city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)
- GGUF quantization support for ComfyUI diffusion models.
- Uses memory-mapped loading and on-demand dequantization.
- Supports Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K through Q6_K, IQ4_NL, IQ4_XS.
- Initial T5 quantization support added recently.
- VAE quantization NOT supported — kept in FP16/FP32.
- Pre-quantized Flux models: [city96/FLUX.1-dev-gguf](https://huggingface.co/city96/FLUX.1-dev-gguf)

### Hugging Face diffusers quantization
- **Blog:** [Exploring Quantization Backends in Diffusers](https://huggingface.co/blog/diffusers-quantization)
- Supports 5 backends: bitsandbytes (NF4/INT8), torchao (INT4/INT8/FP8), Quanto (INT4/INT8/FP8), GGUF, FP8 layerwise casting.
- `PipelineQuantizationConfig` maps components to quantization configs.
- Flux.1-dev memory with different backends (transformer + T5 quantized):
  - BF16 baseline: 31.4 GB
  - bitsandbytes 4-bit: 12.6 GB (60% reduction)
  - torchao int4: 10.6 GB (66% reduction)
  - Quanto INT4: 12.3 GB (61% reduction)
  - GGUF Q8_0: 21.5 GB (32% reduction)
  - GGUF Q4_1: 16.8 GB (46% reduction)

### Quanto (optimum-quanto)
- **Blog:** [Memory-efficient Diffusion Transformers with Quanto](https://huggingface.co/blog/quanto-diffusers)
- Load FP16 model, quantize in-memory, freeze, then run inference.
- SD3 special case: quantize TE1 + TE3 but NOT TE2.
- INT4 tip: exclude `proj_out` layer to maintain quality.
- FP8 shows "almost no quality degradation" visually.
- INT4 causes notable quality loss without layer exclusion.

### Academic References
- **ViDiT-Q** (ICLR 2025): W8A8 and W4A8 for DiT models with negligible degradation. Metric-decoupled mixed precision. [Paper](https://arxiv.org/abs/2406.02540), [Code](https://github.com/thu-nics/ViDiT-Q)
- **SVDQuant** (2024): 4-bit diffusion via SVD low-rank decomposition. 3.5x memory reduction on Flux. [Paper](https://arxiv.org/html/2411.05007v1)
- **Qua2SeDiMo** (2024): Quantifiable quantization sensitivity analysis for diffusion models. Per-layer sensitivity profiling. [Paper](https://arxiv.org/html/2412.14628v1)
- **PTQ4DiT** (NeurIPS 2024): Post-training quantization for diffusion transformers. [Paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/72d32f4fe0b7af03732bd227bf1c4a5f-Paper-Conference.pdf)
- **Q-Sched** (2025): Quantization-aware scheduling, achieves 15.5% FID improvement over FP16 baselines. [OpenReview](https://openreview.net/forum?id=lqCvR0BDss)
- **TFMQ-DM** (CVPR 2024 / TPAMI 2025): Temporal feature maintenance quantization. [Code](https://github.com/ModelTC/TFMQ-DM)

## Differences Between Implementations

| Aspect | sd.cpp | ComfyUI-GGUF | diffusers (Quanto/torchao) |
|--------|--------|--------------|---------------------------|
| Language | C/C++ | Python (PyTorch) | Python (PyTorch) |
| Format | GGUF | GGUF | safetensors + runtime quant |
| Quant strategy | Pre-quantized GGUF files | Pre-quantized GGUF files | Load FP16 then quantize |
| Component separation | Full (separate files per component) | Partial (UNet/DiT + T5) | Full (per-component config) |
| VAE quantization | Not recommended, kept FP16 | Not supported | Not applied by default |
| K-quant support | Yes (with mixed-quant for SD3.5) | Yes | N/A (uses own quant kernels) |
| Dequant overhead | Per tensor per forward pass | Per tensor per forward pass | Per tensor per forward pass |
| GPU acceleration | CUDA, Vulkan, Metal | CUDA (PyTorch) | CUDA (PyTorch) |
| Mixed quant types | Yes (via PR #447 for SD3.5) | Limited | Yes (per-module config) |
| torch.compile | N/A | N/A | Compatible (torchao, GGUF) |

**Key architectural difference for HartsyInference:** sd.cpp is the closest reference since we are also building a native inference engine (C# instead of C++). We should follow sd.cpp's pattern of separate GGUF files per component with independent precision, and its on-demand dequantization approach. However, we can improve on sd.cpp by caching dequantized tensors when VRAM permits (sd.cpp currently dequantizes every forward pass).

## Open Questions

- [ ] Optimal dequantization caching strategy: should HartsyInference cache dequantized tensors between denoising steps when VRAM permits, or always dequantize on-the-fly?
- [ ] Whether INT8 GEMM kernels on modern GPUs (Ada, Hopper) can skip dequantization entirely and compute directly in INT8 for diffusion workloads.
- [ ] Performance impact of GGUF memory-mapped loading on Windows with .NET 10 — need to verify MemoryMappedFile performance for large GGUF files.

## Implementation Notes

### For HartsyInference.ModelAssets

1. **GGUF loader must support per-component precision.** The model handler should accept separate paths and quant configs for backbone, VAE, and each text encoder. Default: Q8_0 backbone, FP16 everything else.

2. **Dequantization kernels needed:**
   - Q8_0: `weight = qs[i] * d` — trivial, SIMD-friendly (32 int8 multiplied by one fp16 scale).
   - Q4_0: Extract 4-bit nibbles, subtract 8, multiply by scale. Pack/unpack with bit manipulation.
   - Q4_K/Q5_K/Q6_K: More complex superblock layout (256 elements) with sub-blocks and multiple scale/min values. Reference ggml source for exact layout.
   - CPU path: Use .NET SIMD (Vector128/Vector256) for dequantization. Q8_0 is ideal for `Avx2.MultiplyAddAdjacent`.
   - CUDA path: Write PTX kernels for fused dequant + matmul where possible.

3. **VAE protection:** The model handler should log a warning and override if a user attempts to load a quantized VAE. The quality risk is too high and memory savings too small to justify.

4. **Memory-mapped GGUF loading:** Use `System.IO.MemoryMappedFiles.MemoryMappedFile` for GGUF tensor data. Dequantize on-demand into a reusable buffer. For the denoising loop (20-50 steps), the same tensors are accessed repeatedly — consider an LRU cache of dequantized tensors bounded by available VRAM.

5. **K-quant tensor shape validation:** Before applying K-quant dequantization, verify the tensor's element count is divisible by 256 (superblock size). For SD3.5 Large, ~90% of tensors will fail this check and need fallback to legacy quant types.

6. **Conversion tool:** HartsyInference should include a safetensors-to-GGUF converter that supports mixed quantization (different quant levels for different components), following sd.cpp's approach.

### For HartsyInference.Diffusion

1. **Pipeline should enforce mixed precision by default.** When loading a Flux pipeline, automatically route:
   - DiT transformer: user-specified quant (default Q8_0)
   - VAE: FP16 (hardcoded minimum)
   - CLIP-L: FP16
   - T5-XXL: FP8 or user-specified

2. **Timestep-aware precision (advanced, future work):** Research shows initial and final denoising steps are most sensitive to quantization error. A future optimization could run the first k and last k steps at higher precision (FP16/BF16) while running intermediate steps at Q8_0 — this is documented in TFMQ-DM and could meaningfully improve Q4 quality.

3. **Quality presets:**
   - `Quality.Maximum`: FP16 everything (24+ GB VRAM)
   - `Quality.High`: Q8_0 backbone + FP16 VAE/encoders (12-16 GB VRAM) — **default**
   - `Quality.Medium`: Q5_K backbone + FP8 T5 + FP16 VAE/CLIP (8-10 GB VRAM)
   - `Quality.Low`: Q4_K backbone + Q4_K T5 + FP16 VAE/CLIP (6-8 GB VRAM)

## Sources

- [sd.cpp quantization docs](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/quantization_and_gguf.md)
- [sd.cpp GitHub](https://github.com/leejet/stable-diffusion.cpp)
- [sd.cpp DeepWiki](https://deepwiki.com/leejet/stable-diffusion.cpp)
- [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)
- [city96 FLUX.1-dev-gguf K-quant comparison](https://huggingface.co/city96/FLUX.1-dev-gguf/discussions/15)
- [city96 FLUX.1-dev-gguf Q8 vs Q4_1 vs FP8](https://huggingface.co/city96/FLUX.1-dev-gguf/discussions/23)
- [stduhpf SD3.5-Large-GGUF-mixed](https://huggingface.co/stduhpf/SD3.5-Large-GGUF-mixed-sdcpp)
- [Exploring Quantization Backends in Diffusers](https://huggingface.co/blog/diffusers-quantization)
- [Memory-efficient Diffusion Transformers with Quanto](https://huggingface.co/blog/quanto-diffusers)
- [ViDiT-Q: ICLR 2025](https://arxiv.org/abs/2406.02540)
- [SVDQuant: 4-Bit Diffusion Models](https://arxiv.org/html/2411.05007v1)
- [Qua2SeDiMo: Quantization Sensitivity of Diffusion Models](https://arxiv.org/html/2412.14628v1)
- [PTQ4DiT: NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/72d32f4fe0b7af03732bd227bf1c4a5f-Paper-Conference.pdf)
- [Q-Sched: Quantization-Aware Scheduling](https://openreview.net/forum?id=lqCvR0BDss)
- [TFMQ-DM: CVPR 2024](https://github.com/ModelTC/TFMQ-DM)
- [NVIDIA SDXL Int8 Quantization Guide](https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/multimodal/text2img/sdxl_quantization.html)
- [SDXL GGUF (HyperX-Sentience)](https://huggingface.co/HyperX-Sentience/SDXL-GGUF)
- [FLUX GGUF Quantization Guide (Apatero)](https://apatero.com/blog/flux-gguf-quantization-8gb-vram-guide-2026)
- [Comprehensive GGUF Analysis (Furkan Gozukara)](https://medium.com/@furkangozukara/comprehensive-analysis-of-gguf-variants-fp8-and-fp16-gguf-q8-vs-fp8-vs-fp16-c212fc077fb1)
- [GGUF Format Deep Dive](https://apxml.com/courses/practical-llm-quantization/chapter-5-quantization-formats-tooling/gguf-format)

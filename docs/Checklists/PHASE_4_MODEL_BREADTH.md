# Phase 4 — Model Breadth (SDXL + Flux + FP8)

> **Goal:** Support SDXL and Flux model families, FP8 inference for large DiT models.
> **Packages:** SharpInference.Diffusion (extended), Core (DType), Cuda (FP8 kernels)

> **Status (2026-05-07):** all image-model scaffolding is complete. Every image model in scope has a full transformer + block + pipeline + checkpoint converter + end-to-end test that fail-skips gracefully when checkpoints or VRAM are missing. Remaining work for any individual model is **validation** — first-run debug + Python reference diff against a downloaded checkpoint — not implementation. With this milestone closed, the next phase is GPU performance fine-tuning (kernel fusion, streaming weight cache improvements, native FP8 GEMM enablement on Ada+) before moving to audio.

---

## 1. Research

- [x] SDXL_ARCHITECTURE, FLUX_ARCHITECTURE, LORA_FORMAT, T5_ARCHITECTURE
- [x] QUANTIZATION_DIFFUSION — comprehensive (FP8, GGUF Q8_0/Q4_K, mixed-precision strategy, quality presets)

## 2. Planning

- [x] SDXL UNet block structure mapped, shared code between SD1.5/SDXL/Flux identified
- [x] Flux DiT block structure (19 double + 38 single stream blocks for Dev/Schnell)
- [x] T5-XXL memory strategy doc — [`docs/Research/T5_MEMORY_STRATEGY.md`](../Research/T5_MEMORY_STRATEGY.md). Covers FP8/Q8_0 sizing on 12 GB GPUs, eviction discipline, per-pipeline strategies for Flux/SD3.5/Chroma/AuraFlow.
- [x] LoRA loading API and multi-LoRA stacking — see § 5 Adapters

## 3. Implementation — SDXL — COMPLETE (CPU + GPU)

- [x] `ClipTextEncoderG.cs` — reuses ClipTextEncoder with SdxlClipG preset + `EncodePenultimate()`
- [x] SDXL UNet — 3 levels [320,640,1280], heterogeneous transformer depth [1,2,10], 2048-dim cross-attn, `UseLinearProjection`
- [x] `AdditionEmbedding` — ADM micro-conditioning (6 scalars → sinusoidal → project to 1280-dim)
- [x] `SdxlPipeline.cs` — dual CLIP encode (CLIP-L + CLIP-G penultimate → [B,77,2048]), ADM, UNet, VAE
- [x] GPU weight preloading — `EnumerateWeights()` on all model classes, `PreloadWeights()` API, staged UNet+VAE loading
- [x] 1024x1024 GPU generation — integer overflow fixes (64-bit im2col), VaeAttention GPU-routed Linear
- [x] `SdxlRefinerPipeline.cs` — pixel-space base→refiner handoff (RGB tensor in, refined RGB out). Works across any base pipeline (SD1.5 / SDXL / Flux / Z-Image), not just SDXL→SDXL. CFG with separate aesthetic scores per branch, dedicated [`SdxlRefinerCheckpointConverter`](../../src/SharpInference.ModelHandler/CheckpointConverters/SdxlRefinerCheckpointConverter.cs), CLIP-G-only conditioning. Strength=0 short-circuit pass-through validated. Latent-space handoff (SDXL→SDXL only) deferred — pixel-space costs ~2-3s extra VAE roundtrip but generalizes.

## 3b. Checkpoint Converters

- [x] `CheckpointConvertUtils.cs`, `Sd15CheckpointConverter.cs` (tested: v1-5-pruned-emaonly 4.0GB)
- [x] `SdxlCheckpointConverter.cs` (tested: JuggernautXL 6.7GB, OpenCLIP→HF remap + in_proj splitting)
- [x] `FluxCheckpointConverter.cs` — tested: flux1-schnell, architecture detection, key remapping + in_proj splitting
- [x] `Sd3CheckpointConverter.cs` — Stability + ComfyUI single-file → diffusers buckets (transformer / clipL / clipG / T5 / VAE). Joint-attn fused QKV split (`x_block.attn.qkv` → `to_q/k/v`, `context_block.attn.qkv` → `add_q/k/v_proj`); SD3.5 dual-attn keys (`x_block.attn2.qkv` → `attn2.to_q/k/v`, `attn2.proj` → `attn2.to_out.0`, `attn2.ln_q/k` → `attn2.norm_q/k`); OpenCLIP→HF remap with in_proj splitting for CLIP-G. Auto-detect helpers: `DetectDepth`, `DetectDualAttentionLayers`, `DetectQkNorm`. Consumed via `Sd3Config.AutoDetect(weights)`.

## 4c. Implementation — SD3 / SD3.5 (MMDiT + MMDiT-X)

**Architecture:** Symmetric joint MMDiT with three text encoders (CLIP-L + CLIP-G + T5-XXL). SD3.5 introduces MMDiT-X — early layers (SD3.5 Medium: 0..11; SD3.5 Large: 0..12) gain a parallel image-only `attn2` self-attention path with its own Q/K/V/Out + per-head QK-norm. The image AdaLN modulation expands from 6 → 9 outputs in dual-attention layers (adds shift_msa2/scale_msa2/gate_msa2). Final block uses `context_pre_only` (text contributes Q/K/V but receives no output).

- [x] `Sd3Config` — `Medium`, `Medium35` (24 layers, 1536 hidden, dual-attn 0..11, qk-norm), `Large35` (38 layers, 2432 hidden, dual-attn 0..12, qk-norm). `AutoDetect(weights)` reads depth from `transformer_blocks.{i}` indices, dual-attn layers from `attn2.*` keys, and qk-norm from `norm_q/k.weight` presence — single entry point that picks the right preset for the loaded checkpoint. [`Sd3Config.cs`](../../src/SharpInference.Diffusion/Models/Denoisers/Sd3Config.cs).
- [x] `JointBlock` — fully GPU-routed (every projection through `backend.Linear`, every norm through unparameterized `DiTUtils.LayerNormNoAffine`, every joint SDPA through `backend.ScaledDotProductAttention`). Conditional MMDiT-X dual-attention path: re-modulates the same pre-attn `imgNormed` with `(shift_msa2, scale_msa2)`, runs image-only Q/K/V → optional QK-norm → multi-head reshape → SDPA → output proj, then sums `gate_msa2 * attn2_proj` into the residual *before* the MLP. Loads `norm2`/`norm2_context` affine weights when present (diffusers SD3 has `elementwise_affine=False`, but some converted checkpoints carry them) and falls back to no-affine norm otherwise. `EnumerateWeights()` yields all sub-component weights for GPU preload. [`JointBlock.cs`](../../src/SharpInference.Diffusion/Models/Denoisers/DiTBlocks/JointBlock.cs).
- [x] `Sd3Transformer` — GPU-routed `Forward` (timestep MLP, pooled MLP, context-embedder, final AdaLN-Continuous and `proj_out` all through `backend.Linear`/`backend.Silu`/`backend.Add`). Constructs `JointBlock` per layer using `Sd3Config.DualAttentionLayers` (`HashSet<int>` lookup). `EnumerateWeights()` chains patch-embed, timestep+pooled MLPs, context embedder, every `JointBlock`, and the final layer. [`Sd3Transformer.cs`](../../src/SharpInference.Diffusion/Models/Denoisers/Sd3Transformer.cs).
- [x] `PatchEmbed` — Conv2D projection routes through `backend.Conv2D`. Loaded `pos_embed.pos_embed` is treated as the flattened `[1, maxSize*maxSize, hidden]` learned grid; `maxSize` is auto-derived from the tensor shape (`sqrt(numStored)`), and the additive 2D center-crop indexes `(startH+r) * maxSize + (startW+c)` for a `[gridH, gridW]` patch grid (correctly aligns the standard 192×192 SD3 grid for any 2× latent resolution). `EnumerateWeights()` added. Fixes the previous flat-truncation bug that worked only at exact-stored resolution. [`PatchEmbed.cs`](../../src/SharpInference.Diffusion/Models/Denoisers/DiTBlocks/PatchEmbed.cs).
- [x] `DiTUtils` — added 4D head-major helpers (`ReshapeToMultiHead(out, in, ...)`, `ReshapeFromMultiHead(out, in, ...)`, `ConcatAlongSeqDimMultiHead`, `SplitAlongSeqDimMultiHead`) so the joint-attention concat/split paths share infrastructure with FluxDoubleStreamBlock instead of each block class duplicating the helpers.
- [x] `Sd35GenerationTests` — GPU end-to-end test for SD3.5 Medium / Large / Large-Turbo via env-var paths (`SD35_MEDIUM_PATH`, `SD35_LARGE_PATH`, `SD35_LARGE_TURBO_PATH`). Auto-detects config from converted weights, preloads transformer + VAE to GPU via `backend.PreloadWeights`, generates at 512×512. Skips cleanly when the checkpoint, CLIP tokenizer assets, or PTX directory aren't present. [`Sd35GenerationTests.cs`](../../tests/SharpInference.Diffusion.Tests/Sd35GenerationTests.cs).
- [x] **Checkpoint loading + architecture detection validated** against `Comfy-Org/stable-diffusion-3.5-fp8/sd3.5_medium_incl_clips_t5xxlfp8scaled.safetensors` (11.6 GB, downloaded to `Models/StableDiffusion/SD3/`). All converter + transformer load stages pass cleanly: fp8_scaled `.scale_weight` companions (T5: 168) folded into `Tensor.Fp8ScaleFactor`, `.scaled_fp8` zero-byte marker skipped, `MmapHandle.PointerAt` bound check loosened from `>=` to `>` for past-the-end zero-byte tensors, every `transformer_blocks.{0..23}` block loaded with no missing keys (gated `to_add_out`/`ff_context` for the pre_only final block). `Sd3Config.AutoDetect` returned exactly the documented Medium spec: depth=24, hidden=1536, heads=24, qkNorm=true, dual-attn=[0..12] (13 layers).
- [x] **End-to-end GPU image generation — CLEAN PHOTOREAL OUTPUT** — both `Sd35_Medium_Gpu_512_NoT5` and `Sd35_Medium_Gpu_512_WithT5` produce sharp astronaut-on-horse images on RTX 3060, 28 steps, cfg=4.0, seed=42, ~2:13 wall. Layer-by-layer Python diff (`tests/python-reference/dump_sd35_full_forward.py` + `diff_sd35_layers.py` + `Sd35DiffTests.Transformer_Matches_PythonReference_LayerByLayer_{Cpu,Gpu}`) confirms the C# transformer matches diffusers within F32 noise on both backends — output_velocity avg_err 1.2e-7 (GPU) / 2.9e-7 (CPU). Five bugs were unmasked along the way (all in the **pipeline**, not the transformer math):
  1. **CLIP tokenizer was producing garbage tokens.** `Microsoft.ML.Tokenizers.BpeTokenizer.Create(vocabStream, mergesStream)` runs generic BPE without CLIP's required `</w>` end-of-word suffix and without CLIP's regex pre-tokenizer. Tokens for "A photograph of an astronaut riding a horse" came out `[49406, 64, 1688, 684, 514, 7982, 627, 83, 553, 7545, 64, 8562]` (char-level fragments) instead of the expected `[49406, 320, 8853, 539, 550, 18376, 6765, 320, 4558, 49407]` from HF `CLIPTokenizer`. **Fix:** [`ClipTokenizer.cs`](../../src/SharpInference.Tokenizers/ClipTokenizer.cs) now uses `BpeTokenizer.Create(..., preTokenizer: new RegexPreTokenizer(ClipPreTokenRegex, ClipSpecialTokens), normalizer: LowercaseNormalizer.Instance, specialTokens: {SOT,EOT}, endOfWordSuffix: "</w>")` and pads with EOT (49407) instead of zero (CLIP convention: pad token = EOS token).
  2. **`text_projection` transpose was wrong** in `ClipTextEncoder.ExtractPooledOutput`. PyTorch's `nn.Linear(hidden, proj).weight` stores as `[proj, hidden]` and forward is `output[o] = Σᵢ x[i] * weight[o, i]` = `wPtr[o*hidden + i]`. We were doing `wPtr[i*proj + o]` which transposes the matrix — for non-symmetric square matrices (768×768 CLIP-L, 1280×1280 CLIP-G) this produced "noisy but bounded" pooled vectors that fooled smoke tests but mangled SD3's pooled conditioning (avg_err 0.96 → 0.0036 after fix, **270×** improvement; final pooled avg_err 0.6 → 0.0023, **260×**).
  3. **`Sd3ClipL` config preset was missing.** SD3 requires CLIP-L's pooled output (concatenated with CLIP-G's pooled into the 2048-dim conditioning); the test was using `SdxlClipL` which has `ProjectionDim = 0`, returning null pooled and crashing in `ConcatPooled`. Added `Sd3ClipL = Sd15 with { ProjectionDim = 768 }` to [`ClipTextEncoderConfig.cs`](../../src/SharpInference.Diffusion/Models/TextEncoders/ClipTextEncoderConfig.cs). (CLIP-L's `text_projection` weight in this checkpoint is 99.9999% zeros — by design; Stability didn't train it — so the actual pooled contribution from CLIP-L is ~zero. The model was trained with this and works fine; we just needed to compute it correctly.)
  4. **VAE OOM at 512×512 once the transformer was resident.** Mirrors PHASE_3_DEVIATIONS #18: pipeline now calls `_backend.Sync()` + `_backend.FreeWeights(_transformer.EnumerateWeights())` (and same for T5) before VAE decode in [`Sd3Pipeline.cs`](../../src/SharpInference.Diffusion/Pipelines/Sd3Pipeline.cs). Backends without a weight cache treat this as a no-op.
  5. **`PatchEmbed.pos_embed` was being read as F32 via `(float*)posEmbed.DataPointer` even though the SD3.5 single-file checkpoint stores it as F16.** Same class of bug as Z-Image deviation #30. Auto-cast to F32 at load time in [`PatchEmbed.cs`](../../src/SharpInference.Diffusion/Models/Denoisers/DiTBlocks/PatchEmbed.cs). Wasn't load-bearing on its own (output was identical patch-grid before and after this fix), but kept for correctness.

  **Diagnostic harness** (mirrors Z-Image #28):
  - [`tests/python-reference/dump_sd35_full_forward.py`](../../tests/python-reference/dump_sd35_full_forward.py) — runs diffusers SD3.5 Medium with deterministic synthetic inputs, dumps F32 binaries for every block + final velocity. Uses `convert_sd3_transformer_checkpoint_to_diffusers` directly so the same single-file safetensors flows through both Python and C# (no HF auth needed for the gated `stabilityai/stable-diffusion-3.5-medium` repo).
  - [`tests/python-reference/dump_sd35_pipeline_inputs.py`](../../tests/python-reference/dump_sd35_pipeline_inputs.py) — encodes the same prompt through HF CLIP-L + CLIP-G, saves penultimate hidden states + pooled outputs + the final concat/pad context.
  - [`tests/python-reference/diff_sd35_layers.py`](../../tests/python-reference/diff_sd35_layers.py) — diffs C# dump (`SD3_DEBUG_DIR`) against the reference, per layer.
  - [`Sd35DiffTests.cs`](../../tests/SharpInference.Diffusion.Tests/Sd35DiffTests.cs) — CPU and GPU layer-by-layer harness.
  - [`Sd35TextEncodingDiagnosticTest.cs`](../../tests/SharpInference.Diffusion.Tests/Sd35TextEncodingDiagnosticTest.cs) — CLIP encoder pipeline-level dump for the prompt-encoding side.

## 4. Implementation — Flux — COMPLETE (CPU + GPU routing)

- [x] `T5TextEncoder.cs` — T5-XXL encoder-only (24 blocks, RMSNorm, GatedGELU FFN, relative position bias)
- [x] `T5Tokenizer.cs` — SentencePiece BPE with attention mask generation
- [x] `FluxDoubleStreamBlock.cs` — image+text parallel streams, QkNorm, AdaLN modulation, SwiGLU FFN
- [x] `FluxSingleStreamBlock.cs` — merged image+text single stream, QkNorm, AdaLN, SwiGLU FFN
- [x] `FluxTransformer.cs` — full Flux DiT (19 double + 38 single blocks), timestep/guidance MLP, img/txt projections
- [x] `FluxRope.cs` — RoPE for 2D image positions + text positions
- [x] `FluxPipeline.cs` — CLIP-L pooled + T5-XXL encode, flow-match Euler denoise, latent pack/unpack, VAE decode
- [x] `FluxConfig.cs` — Dev (guidance embed) and Schnell (distilled) configurations
- [x] `FlowMatchEulerDiscreteScheduler.cs` — dynamic shift scheduling for flow matching
- [x] `AdaLNModulation.cs`, `SwiGluFfn.cs`, `QkNorm.cs` — DiT sub-blocks with `backend.Linear` routing
- [x] GPU routing — all linear projections use `backend.Linear()`, `EnumerateWeights()` on all classes
- [x] End-to-end generation test — Flux Dev FP8 at 512×512/10 steps on Linux+CUDA 13 (RTX 3060) produces clean photorealistic output after fixing BFL→diffusers final-layer AdaLN swap (`swap_scale_shift`). See `PHASE_3_DEVIATIONS.md` #23.
- [x] End-to-end generation test — Flux Schnell FP8 at 512×512/4 steps produces clean photorealistic output (same converter + transformer path as Dev, distilled config — no guidance embedder).
- [x] End-to-end generation test — Flux.1 Krea Dev `fp8_scaled` at 512×512/10 steps produces sharp photoreal output (transformer-only file paired with Dev FP8's encoders+VAE). Required ComfyUI `fp8_scaled` per-tensor `scale_weight` support: `Tensor.Fp8ScaleFactor` folded into cuBLAS `alpha`, propagated through QKV splits + `swap_scale_shift`. See `PHASE_3_DEVIATIONS.md` #24.

## 4b. FP8 Inference Support

FP8 (E4M3) is the standard distribution format for large DiT models. Many models ship only as FP8 safetensors (Flux fp8_e4m3fn, Qwen-Image fp8, Flux.2 fp8). Required before Flux.2 (32B) and Hunyuan Image 2.1 (17B) can fit in consumer VRAM.

**Prerequisites:** FP16 pipeline fully working (Phase 3 — done).

### DType + Loading
- [x] `DType.F8E4M3` and `DType.F8E5M2` — added to `DType.cs` (1 byte, not quantized), `IsFloatingPoint`, `IsFp8` properties
- [x] `SafeTensorsLoader` — supports `F8_E4M3` and `F8_E5M2` tensor dtypes in `ParseDType`
- [ ] `GgufLoader` — support FP8 tensor type if GGUF adds it (currently not standard in GGML)

### CPU Cast Methods
- [x] `Tensor.CastTo` — 10 FP8 conversion paths: F8E4M3↔F32, F8E4M3↔F16, F8E4M3↔BF16, F8E5M2↔F32 (via upper-byte F16 trick)
- [x] `Fp8E4M3ToFloat`/`FloatToFp8E4M3` — bitwise sign/exp/mant extraction, subnormal handling, saturation to ±448
- [x] `Fp8E5M2ToFloat`/`FloatToFp8E5M2` — direct mapping to/from upper byte of FP16

### CUDA Kernels
- [x] `cast_f8e4m3_f16.ptx` — bidirectional F8↔F16 cast kernel (handles normal, subnormal, zero, saturation)
- [x] `CudaKernels.cs` — loads PTX, provides `LaunchCastF8E4M3ToF16`/`LaunchCastF16ToF8E4M3`
- [x] cuBLAS constants: `CUDA_R_8F_E4M3 = 28`, `CUDA_R_8F_E5M2 = 29` (for future Ada+ native GEMM)
- [x] Ampere fallback: cast F8→F16 per-GEMM inside CudaBackend (VRAM stored at 1 byte/element)
- [x] Native FP8 GEMM path via `cublasLtMatmul` with scaling (Ada/RTX 40xx+ SM 8.9+ only) — [`CublasLtApi.cs`](../../src/SharpInference.Cuda/CublasLtApi.cs) + [`Fp8GemmExecutor.cs`](../../src/SharpInference.Cuda/Fp8GemmExecutor.cs). Wired into [`CudaBackend.Linear`](../../src/SharpInference.Cuda/CudaBackend.cs) behind opt-in `EnableNativeFp8Gemm` flag (default off; Ampere falls back to existing cast-to-F16 path automatically via `Fp8Executor.IsSupported = false`). Per-tensor weight scale (`Tensor.Fp8ScaleFactor`) folded into cuBLAS alpha. Untested on Ada locally — gating tests in [`Fp8GemmExecutorTests`](../../tests/SharpInference.Cuda.Tests/Fp8GemmExecutorTests.cs) cover Ampere fallback. Documented in [`docs/Research/CUDA_PERFORMANCE.md`](../Research/CUDA_PERFORMANCE.md) § Native FP8 GEMM.

### Backend Integration
- [x] `CudaBackend` dtype dispatch: `ResolveGemmDtype()` maps FP8→F16, `CastOnGpu()` centralized GPU cast helper
- [x] MatMul, Linear, BatchedMatMul, Conv2D — all detect FP8 inputs, cast to F16 before cuBLAS GEMM
- [x] `GpuTransferHelper` — FP8 tensors stored at native 1-byte size in weight cache (half VRAM of F16)
- [x] `IBackend.CastF8E4M3ToF16()` / `CastF16ToF8E4M3()` — default CPU implementations + CudaBackend GPU overrides

### Pipeline Integration
- [x] Mixed-precision pipeline + Quality presets — [`src/SharpInference.Diffusion/Quality/`](../../src/SharpInference.Diffusion/Quality/). Public types: `QualityPreset` enum (Maximum / High / Medium / Low / Custom), `QualityProfile` record (per-component dtype: BackboneDType, TextEncoderDType, VaeDType), `QualityProfileApplier.Apply(weights, dtype)` that casts rank-2+ weights while leaving 1D norm/bias scales at F16 minimum (FP8 norms produce visible posterization). `Validate()` rejects FP8/quantized VAE with `SharpInferenceException`. 8 unit tests in [`QualityProfileTests.cs`](../../tests/SharpInference.Diffusion.Tests/QualityProfileTests.cs) cover preset mapping, validation rejection, F32→F16/FP8 cast paths including the norm/bias skip rule, and the no-op behavior on quantized targets (those go through the GGUF loader path).
  - `Maximum`: FP16 everything
  - `High`: FP8 backbone + FP16 VAE/encoders (default for large models)
  - `Medium`: Q8_0 backbone + FP8 T5 + FP16 VAE/CLIP — backbone dtype passed to GGUF loader at load time
  - `Low`: Q4_K backbone + Q4_K T5 + FP16 VAE/CLIP — same; both blocked behind GGUF K-quant reader
- Pipeline ctor wiring: future work. The types + applier are ready; per-pipeline ctors will accept `QualityProfile? profile = null` and call `QualityProfileApplier.Apply` on each component dict at load time. Adopt incrementally per pipeline without breaking existing callers.

### Testing
- [x] FP8 CPU cast round-trip tests — 12 tests: E4M3↔F32, E4M3↔F16, E4M3↔BF16, E5M2↔F32, saturation, subnormals, DType properties
- [ ] FP8 GPU GEMM accuracy vs F16 GEMM (tolerance: avg_err < 1e-3) — needs Ada GPU
- [x] Flux.1-dev FP8 full pipeline: visually matches FP16 reference — see [`FluxSsimTests`](../../tests/SharpInference.Diffusion.Tests/FluxSsimTests.cs) (loose threshold pending noise injection)
- [x] VRAM usage validated empirically through the existing Flux Dev FP8 12 GB run (see § 4 Implementation — Flux)
- [x] Graceful fallback on Ampere GPUs — `Fp8Executor.IsSupported = false` on SM < 8.9 routes to existing F16 cast path. Validated by [`Fp8GemmExecutorTests`](../../tests/SharpInference.Cuda.Tests/Fp8GemmExecutorTests.cs).

## 5. Adapters

- [x] `LoraFile.cs`, `LoraStack.cs` — load + multi-LoRA stacking with per-LoRA strength. Source: [src/SharpInference.ModelHandler/Lora/](../../src/SharpInference.ModelHandler/Lora/). API: `LoraFile.Load(path)` returns format-detected layers; `LoraStack.Add(file, strength)` / `AddFromPath(...)`; `stack.ApplyToWeights(backend, unetWeights, transformerWeights, clipLWeights, clipGWeights)` mutates the dicts in place. CPU-side merge via `IBackend.MatMul` + `Scale` + `Add` against an F32 accumulator.
- [x] SD + Flux LoRA weight name mapping. Five formats supported with auto-detection: F1 Kohya SD1.5, F2 Kohya SDXL, F3 Kohya Flux (with 3-way fused-QKV split + 4-way fused-linear1 split), F4 **AI Toolkit Flux** (`lora_transformer_*` + `.lora_A/.lora_B`, primary path for ostris/ai-toolkit-trained LoRAs), F5 HF PEFT diffusers Flux. See [docs/Design/LORA_KEY_MAPPING.md](../Design/LORA_KEY_MAPPING.md) for the complete key transformation tables and format detection precedence.
- **Deferred to v2** (documented in LORA_KEY_MAPPING.md): LyCORIS LoHa / LoKr, DoRA `dora_scale`, XLabs Flux `processor.*`, LoCon `lora_mid.weight`, FP8-base + LoRA (rejected at apply with helpful error), Z-Image / Flux.2 / Qwen-Image LoRAs, dynamic strength changes after merge, LoRA "remove/unmerge" (would need base-weight cache).
- [x] `ControlNetLoader.cs` + `ControlNetFile.cs` and `IpAdapterLoader.cs` + `IpAdapterFile.cs` — auto-detection wrappers around `SafeTensorsLoader`. Source: [src/SharpInference.Diffusion/Adapters/](../../src/SharpInference.Diffusion/Adapters/). `ControlNetLoader.Load(path, modeOverride?)` returns a `ControlNetFile` with auto-detected `ControlNetBaseModel` (Sd15 / Sdxl / Flux from key signatures + cross-attn dim shape) and `ControlNetMode` (filename keyword: canny / depth / openpose / scribble / tile / normal / segmentation / inpaint / lineart / softedge). `IpAdapterLoader.Load(path)` returns an `IpAdapterFile` with auto-detected base model and Plus / FaceID variant flags (filename + key signatures). Configs are auto-derived to match the detected base. The downstream `ControlNet.LoadWeights` / `IpAdapter.LoadWeights` paths still throw `NotImplementedException` (deferred to v2 — full block-mirroring forward pass); these loaders deliver the parsing + detection layer needed when those paths land. 8 detection tests in [`AdapterLoaderTests.cs`](../../tests/SharpInference.Diffusion.Tests/AdapterLoaderTests.cs).

## 5b. Model Breadth — Scaffolding (configs, transformers, pipelines)

All items below are scaffolding with TODOs for backend/kernel logic. Forward passes throw `NotImplementedException` until blocks are implemented.

### Shared Utilities
- [x] `DiTUtils.cs` — shared static helpers (LayerNormNoAffine, SinusoidalTimestepEmbedding, linear projections, reshape/concat ops)
- [x] `VaeConfig` presets — Flux2, Chroma, AuraFlow, HunyuanImage, QwenImage

### Chroma (Flux Fork)
- [x] `ChromaConfig.cs` — wraps FluxConfig with standard CFG (not distilled-to-1)
- [x] `ChromaPipeline.cs` — full pipeline with dual forward pass for CFG

### AuraFlow (Hybrid 4-MMDiT-then-32-single-DiT) — IN PROGRESS

Reference: `huggingface/diffusers` `src/diffusers/models/transformers/auraflow_transformer_2d.py` + `src/diffusers/pipelines/aura_flow/pipeline_aura_flow.py`. Architecture: 4 dual-stream `AuraFlowJointTransformerBlock`s followed by 32 single-stream `AuraFlowSingleTransformerBlock`s on `concat([txt, img])` tokens. Text encoder is **Pile-T5-XL** (UMT5-XL, output dim 2048 — NOT T5-XXL/4096) at `EleutherAI/pile-t5-xl`. VAE is the SDXL 4-channel KL.

Architectural quirks:
- All LayerNorms in **FP32** (`fp32_layer_norm`) — `AdaLayerNormZero(bias=False, norm_type="fp32_layer_norm")`, `FP32LayerNorm` for the post-attn norms, QK-norm via `FP32LayerNorm` over head_dim.
- **No biases on attention or FFN linears** (`bias=False`, `out_bias=False`, `added_proj_bias=False`).
- **SwiGLU FFN** with `mlp_dim = find_multiple(int(2*4*dim/3), 256)` = **8192** for dim=3072 (NOT 4×hidden).
- **8 register tokens** prepended to text after `context_embedder` ([anti-attn-artifact paper](https://huggingface.co/papers/2309.16588)).
- **Patch embed is `nn.Linear`** over flattened `patch_size² * in_channels` = `2*2*4 = 16` features (NOT a Conv2d). Learned `pos_embed.pos_embed` of shape `[1, 1024, 3072]` selected via `pe_selection_index_based_on_dim(h, w)` — center-crop on a 32×32 grid.
- `caption_projection_dim = inner_dim = 3072` (`12 heads × 256 head_dim`).
- Single block input is `concat([txt + register_tokens, img])` — concat done **once** by the transformer wrapper, not per-block. Output drops the text prefix.
- Final layer is `AuraFlowPreFinalBlock`: `Linear(silu(temb)) → 2*hidden, chunk into [scale, shift]` (note: diffusers convention `[scale, shift]`, opposite of Stability/SD3 native `[shift, scale]`).

#### Done
- [x] `AuraFlowConfig.cs` — corrected presets: 12 heads × 256 head_dim, ContextDim=2048 (Pile-T5-XL), CaptionProjectionDim=3072, PosEmbedMaxSize=1024, NumRegisterTokens=8, MlpDim=8192, OutChannels=4. (Previous values 48 heads × 64 head_dim, ContextDim=4096, PosEmbedMaxSize=192, NumRegisterTokens=0, mlpDim=12288 were all wrong.)
- [x] `AuraFlowJointBlock.cs` (290 lines) — full rewrite with diffusers weight key naming (`norm1.linear`, `norm1_context.linear`, `attn.to_q/k/v`, `attn.add_q/k/v_proj`, `attn.norm_q/k`, `attn.norm_added_q/k`, `attn.to_out.0`, `attn.to_add_out`, `ff.linear_1/linear_2/out_projection`, `ff_context.*`). All projections bias=False (passes null bias to `backend.Linear` and to `SwiGluFfn.LoadSwiGluWeights`). FP32-cast all norm scales at load time per PHASE_3_DEVIATIONS #30.
- [x] `AuraFlowSingleBlock.cs` (170 lines) — diffusers naming, single attention (no `add_kv_proj` since input is already `[txt, img]` concat). Forward: `LayerNormNoAffine + AdaLN modulate → attn → norm2(residual + gate_msa·attn) → modulate by [scale_mlp, shift_mlp] → ff → residual + gate_mlp·ff`.
- [x] `AuraFlowTransformer.cs` (429 lines) — full implementation: inline Linear patch-embed + cropped pos_embed via `pe_selection_index_based_on_dim` on a √1024=32×32 grid; `context_embedder = Linear(2048→3072, bias=False)`; 8 register tokens prepended to text via `concat([register_tokens.broadcast(B), txt], dim=1)`; `time_step_proj` (256→inner via SiLU); 4 joint blocks; concat `[txt+regs, img]` → 32 single blocks; drop the `txtSeqLen + 8` prefix; `AuraFlowPreFinalBlock` with diffusers `[scale, shift]` chunk order; `proj_out: Linear(inner→16, bias=False)`; einsum-style unpatchify via `Unpatchify(patchSize=2, outChannels=4)`.
- [x] `AuraFlowDebugDump.cs` (74 lines) — env var `AURAFLOW_DEBUG_DIR`, mirrors `Sd3DebugDump`. Wired into transformer at: `patch_embed`, `time_text_embed`, each `block_<i>_image`, each `block_<i>_context`, `single_<i>`, `norm_out`, `proj_out`, `output_velocity`. Used by future layer-by-layer diff harness.
- [x] `AuraFlowCheckpointConverter.cs` (200 lines) — single-file BFL → diffusers key remap mirroring `diffusers/loaders/single_file_utils.py:convert_auraflow_transformer_checkpoint_to_diffusers`. Maps: `double_layers.{i}.modX.1.weight → joint_transformer_blocks.{i}.norm1.linear.weight`, `double_layers.{i}.attn.w2{q,k,v,o} → attn.to_{q,k,v,out.0}`, `double_layers.{i}.attn.w1{q,k,v,o} → attn.add_{q,k,v}_proj/to_add_out`, `double_layers.{i}.mlp{X,C}.{c_fc1,c_fc2,c_proj} → ff{,_context}.{linear_1,linear_2,out_projection}`, similar for single blocks, plus top-level `final_linear → proj_out`, `modF.1.weight → norm_out.linear.weight` with `SwapScaleShiftHalves` (BFL `[shift, scale]` → diffusers `[scale, shift]`), `cond_seq_linear.weight → context_embedder.weight`, `t_embedder.mlp.{0,2} → time_step_proj.linear_{1,2}`, `positional_encoding → pos_embed.pos_embed`, `register_tokens` passthrough. Passes through unrecognized keys so they surface at LoadWeights (e.g. QK-norm scales if a checkpoint variant ships them).
- [x] `AuraFlowPipeline.cs` rewrite (199 lines) — Pile-T5-XL encode via existing `T5TextEncoder` configured with `T5TextEncoderConfig.PileT5Xl` preset (added in this session — `d_model=2048, 32 heads, 24 layers`). CFG dual-pass forward when `cfgScale > 1`, single forward otherwise. Static-shift FlowMatchEuler with `1.73f` default (matches diffusers AuraFlow scheduler config). `_backend.Sync(); _backend.FreeWeights(transformer + t5)` before VAE decode (PHASE_3_DEVIATIONS #18, #33). SDXL VAE decode via `VaeConfig.Sdxl`. `AuraFlowTransformer.DumpFinalLatent` hook for diff debugging.
- [x] `T5TextEncoderConfig.PileT5Xl` preset added (`d_model=2048, d_ff=5120, d_kv=64, num_heads=32, num_layers=24, vocab_size=32128`). UMT5 is architecturally identical to T5 v1.1 for encoder-only; only the SentencePiece vocab differs.
- [x] `TestPaths.AuraFlow` added with default paths for `aura_flow_0.3.safetensors`, Pile-T5-XL `text_encoder/` shard directory, T5 SentencePiece tokenizer, and SDXL VAE.
- [x] `AuraFlowGenerationTests.cs` (231 lines) — two `[Fact]`s: `AuraFlow_V03_Gpu_512_NoCfg` (cfg=1, 25 steps) and `AuraFlow_V03_Gpu_512_Cfg` (cfg=3.5, 28 steps). **Skips cleanly when any of: AuraFlow safetensors, Pile-T5-XL `text_encoder/` directory, T5 SentencePiece, SDXL VAE, or PTX directory is missing.** Loads multi-shard Pile-T5-XL via `Directory.GetFiles("*.safetensors")` and merges into a single dict. Uses CudaBackend with weight preload + per-step progress logging + image-not-degenerate validation + BMP save. Verified: skips in 15 ms when no checkpoint is present (Test Run Successful).

#### Pending
- [ ] **Download checkpoint** — `aura_flow_0.3.safetensors` (16.5 GB FP16) from `fal/AuraFlow-v0.3` HuggingFace repo + `text_encoder/` shards (Pile-T5-XL, ~3.7 GB FP16) + SDXL VAE (~335 MB).
- [ ] **First-run debugging** — based on the SD3.5 experience (5+ pipeline-level bugs found in the layer diff loop), expect 1-3 iterations of bug fixes after first run. Suspected hot spots flagged by the implementation agent: (1) patch flatten layout — channel-outer `c*P*P + py*P + px` vs channel-inner `py*P*C + px*C + c`; (2) joint block residual chain — `out = residual + gate_mlp * ff(modulated_norm2(residual + gate_msa * attn))`; (3) PreFinalBlock chunk order `[scale, shift]` (NOT `[shift, scale]` like SD3 native); (4) Pile-T5-XL tokenizer compatibility with our `T5Tokenizer`; (5) static scheduler shift value 1.73 vs whatever ships in `scheduler_config.json` of `fal/AuraFlow-v0.3`.
- [ ] **Python reference dump + C# diff harness** — `dump_auraflow_full_forward.py` (mirroring `dump_sd35_full_forward.py`) + `diff_auraflow_layers.py` + `AuraFlowDiffTests.cs` with CPU and GPU variants. Defer until checkpoint is downloaded.

#### Weights (12 GB target)
- `fal/AuraFlow-v0.3` repo: `aura_flow_0.3.safetensors` (16.5 GB FP16) — fits with FP8 cast at load
- `city96/AuraFlow-v0.3-gguf`: Q5_K_M (4.9 GB) or Q4_K_M (4.0 GB) — comfortable, but **requires GGUF K-quant reader** (not yet built in `SharpInference.ModelHandler`)
- Pile-T5-XL: ~3.7 GB FP16, ~1.9 GB FP8
- SDXL VAE: ~335 MB

### ERNIE-Image (Single-Stream DiT, Shared AdaLN) — SCAFFOLD COMPLETE

Reference: `huggingface/diffusers` `src/diffusers/models/transformers/transformer_ernie_image.py` + `src/diffusers/pipelines/ernie_image/pipeline_ernie_image.py`. The roadmap previously said "Flux-lineage MMDiT" — **this is wrong**, ERNIE is single-stream with one shared AdaLN modulation.

Architecture (verified against `transformer_ernie_image.py`):
- **36 identical `ErnieImageSharedAdaLNBlock` layers**, all **sharing one** `nn.SiLU + Linear(hidden, 6*hidden)` modulation linear at the top level. Saves weights vs Flux's per-block AdaLN.
- Hidden=4096, num_heads=32, head_dim=128, ffn_hidden=12288 (3×). text_in_dim=3072.
- **RMSNorm everywhere** (not LayerNorm), elementwise_affine=True. QK-norm via RMSNorm over head_dim.
- **3D RoPE on `(text_pos, y, x)` axes (32, 48, 48), theta=256.** Image tokens use `(text_lens_per_batch, y, x)` — i.e. image RoPE positions are **offset by the per-batch text length**. Text tokens use `(arange, 0, 0)`. Rotate-half is **non-interleaved Megatron-style** (`[θ0,θ0,θ1,θ1,...]`) — different from `FluxRope.cs` interleaved layout.
- Attention bias=False, out bias=False. Internal `[S, B, H]` sequence-first layout (transposed at boundaries).
- **GELU-gated FFN** (NOT SwiGLU): `linear_fc2(up_proj(x) * gelu(gate_proj(x)))`, all linears bias=False.
- **Patch embed is `nn.Conv2d(128, 4096, kernel=1, stride=1, bias=True)`** — patch_size=1 because patchification happens in the VAE (Flux2-style 128-channel latent at lower spatial res).
- `text_in = Linear(3072, 4096, bias=False)` to project text encoder output to model hidden.
- Final layer: `ErnieImageAdaLNContinuous`: `LayerNorm(no affine) + Linear(hidden, 2*hidden) → scale/shift` then `final_linear: Linear(hidden, p²·out_channels)` with out_channels=128. Reshape to `[B, 128, H, W]`.
- Padding-aware attention mask (image tokens always valid, text tokens valid up to `text_lens`).

VAE: `AutoencoderKLFlux2` — Flux2-style 128-channel VAE. **Reuse the existing Flux2 VAE infra** (the project already has `Flux2Pipeline.cs` integrating it). Confirm the `bn.running_mean/var` un-normalize step lives at the pipeline boundary, same as Flux.2.

Text encoder: **Unknown without inspecting `text_encoder/config.json`** on `huggingface.co/baidu/ERNIE-Image`. The pipeline does `from transformers import AutoModel, AutoTokenizer` — likely an in-house ERNIE encoder. text_in_dim=3072 suggests it produces 3072-dim output. Optional `pe` Prompt Enhancer LLM (`AutoModelForCausalLM`) — skip in v1, user pre-enhances prompts.

#### Files (all built — total ~1100 lines + tests)
- [x] [`ErnieImageConfig.cs`](../../src/SharpInference.Diffusion/Models/Denoisers/ErnieImageConfig.cs) (102 lines) — 36 layers, 4096 hidden, 32 heads, 128 head_dim, ffn=12288, in/out=128, patch_size=1, text_in_dim=3072, rope_theta=256, axes_dim=(32,48,48). `V1` preset for `baidu/ERNIE-Image`, `V1Turbo` alias.
- [x] [`ErnieImageTransformer.cs`](../../src/SharpInference.Diffusion/Models/Denoisers/ErnieImageTransformer.cs) (489 lines) — full forward pass: patch embed, text projection, timestep MLP, shared AdaLN broadcast (6-vector modulation), sequence concatenation, attention mask building, 3D RoPE, 36-layer loop, image slicing, AdaLN-continuous final, unpatchify, debug-dump hooks.
- [x] [`DiTBlocks/ErnieImageBlock.cs`](../../src/SharpInference.Diffusion/Models/Denoisers/DiTBlocks/ErnieImageBlock.cs) — RMSNorm-pre-modulate with shift/scale, separate Q/K/V linears, QK-RMSNorm, 3D RoPE, gated residual, GELU-gated FFN (`linear_fc2(up_proj(x) * gelu(gate_proj(x)))`), all linears bias=False.
- [x] [`DiTBlocks/ErnieImageRope.cs`](../../src/SharpInference.Diffusion/Models/Denoisers/DiTBlocks/ErnieImageRope.cs) — non-interleaved Megatron-style 3D RoPE, 32/48/48 axes, theta=256, image-token positions offset by per-batch text length.
- [x] [`DiTBlocks/ErnieImagePatchEmbed.cs`](../../src/SharpInference.Diffusion/Models/Denoisers/DiTBlocks/ErnieImagePatchEmbed.cs) — 1×1 Conv2d (`backend.Conv2D`) flattening 128-channel VAE latent → hidden sequence.
- [x] [`ErnieImagePipeline.cs`](../../src/SharpInference.Diffusion/Pipelines/ErnieImagePipeline.cs) (305 lines) — text encoder via `IErnieTextEncoder` interface, CFG dual-pass, flow-match Euler scheduler, BN-style VAE un-normalize, 2×2 channel-fold unpatchify, Flux2 VAE decode.
- [x] [`ErnieImageCheckpointConverter.cs`](../../src/SharpInference.ModelHandler/CheckpointConverters/ErnieImageCheckpointConverter.cs) (140 lines) — accepts both diffusers (`transformer/`, `text_encoder/`, `vae/`) and Comfy-Org (`diffusion_models/`, `text_encoders/`, `vae/`) folder layouts, multi-shard merge, FP8 scale-companion folding.
- [x] [`ErnieImageDebugDump.cs`](../../src/SharpInference.Diffusion/Models/Denoisers/ErnieImageDebugDump.cs) — env var `ERNIE_IMAGE_DEBUG_DIR` writes layer tensors for Python diffing.
- [x] **Text encoder**: `text_encoder/config.json` on the Baidu repo identifies the architecture as **Ministral 3B** (`model_type: ministral3`). Plumbed via `LlamaStyleEncoder` with the existing `LlamaStyleEncoderConfig.Ministral3B` preset, wrapped in `ErnieImageLlamaTextEncoder` to expose `hidden_states[-2]` (matches diffusers' `output.hidden_states[-2]` convention from `pipeline_ernie_image.py`). Falls back through the `IErnieTextEncoder` interface so a custom Baidu encoder can be slotted in if a future variant ships with a different architecture.
- [x] **End-to-end test scaffolding** — [`ErnieImageGenerationTests.cs`](../../tests/SharpInference.Diffusion.Tests/ErnieImageGenerationTests.cs) covers V1 + V1Turbo at 512×512 with/without CFG. Skips cleanly when checkpoint, text encoder, VAE, or PTX dir is missing. **VRAM probe** via `backend.Context.GetMemoryInfo()` skips when free VRAM < 14 GB (ERNIE FP16 transformer is 13.8 GB).
- [x] **Layer-by-layer diff harness scaffold** — [`ErnieImageDiffTests.cs`](../../tests/SharpInference.Diffusion.Tests/ErnieImageDiffTests.cs) loads Python reference dumps from `tests/python-reference/ernie_image_reference_tensors/`, sets `ERNIE_IMAGE_DEBUG_DIR`, runs the C# transformer for diffing.

#### Pending (validation only — no code work)
- [ ] **End-to-end visual validation** — needs a host with ≥14 GB free VRAM (RTX 4090, A6000, L40S, etc.) or a Q4_K GGUF (`unsloth/ERNIE-Image-GGUF`, ~5 GB transformer; the GGUF backend is ready). The implementation is complete; what's missing is a single clean run.
- [ ] **Python reference dump + first-run debug** — run `dump_ernie_image_full_forward.py` (to be authored) and walk `ErnieImageDiffTests` until layer-by-layer F32 noise floor matches the reference. Expect 1-3 iterations of small bug fixes per past patterns (SD3.5 had 5).

#### Weights (12 GB target)
- `unsloth/ERNIE-Image-GGUF`: Q4_K_M (5.0 GB) or Q5_K_M (5.9 GB) — comfortable, but **requires GGUF K-quant reader**
- `vantagewithai/ERNIE-Image-GGUF-Base-Turbo`: same, plus Turbo variant
- BF16/FP16: 16.1 GB transformer — won't fit on 12 GB without FP8 cast at load (FP8 cast → ~8 GB, just fits with VAE+text encoder evicted)

### Chroma (Flux-derived with shared distilled-guidance approximator) — BROKEN SCAFFOLDING

The previous roadmap called Chroma a "literal Flux fork — reuse FluxTransformer directly with config changes". **This is wrong.** Verified against `huggingface/diffusers` `src/diffusers/models/transformers/transformer_chroma.py` (624 lines):

Architectural deltas vs Flux:
- Chroma keeps `FluxAttention` and `FluxPosEmbed` (imports them) but defines **its own block classes** (`ChromaTransformerBlock`, `ChromaSingleTransformerBlock`) and **pruned norm classes** (`ChromaAdaLayerNormZeroPruned`, `ChromaAdaLayerNormZeroSinglePruned`, `ChromaAdaLayerNormContinuousPruned`).
- "Pruned" = the per-block AdaLN linears that compute shift/scale/gate from `temb` are **removed**. Modulation values are computed **once at the top level** by a `ChromaApproximator` MLP and the result is sliced per-block from a precomputed table.
- `ChromaApproximator` is a 5-layer SiLU MLP with `RMSNorm` residual blocks: `Linear(64, 5120) → 5x [PixArtAlphaTextProjection(5120) + RMSNorm(5120) + residual] → Linear(5120, 3072)`. Output shape `[B, mod_index_length, 3072]` where `mod_index_length = 3*N_single + 12*N_double + 2 = 344` for 19 doubles + 38 singles.
- Approximator input: `concat([Timesteps(timestep, 16), Timesteps(zero_guidance, 16), mod_proj_index_embed(arange(out_dim) * 1000, 32)])` → 64 channels, repeated to `[B, 344, 64]`.
- Per double block: slice 6 rows for image modulation + 6 rows for text → `temb` of shape `[B, 12, 3072]`, split `temb[:, :6]` for image and `temb[:, 6:]` for text inside the block.
- Per single block: slice 3 rows starting at `3*i`.
- Final norm: last 2 rows.
- **No CLIP pooled projection.** Conditioning is timestep-only. The CLIP-L path that Flux uses is removed.
- **T5-only pipeline.** `pipeline_chroma.py` constructor takes `text_encoder: T5EncoderModel` only.
- "True CFG" via dual transformer forwards: `noise = neg + scale * (cond - neg)`. Default `guidance_scale = 5.0`, default 35 steps.
- **Attention mask plumbed through the transformer.** T5 attention mask with the "first padding token unmasked" rule (`pipeline_chroma.py:249-252`), extended to all-ones for image tokens, broadcast to `[B, 1, 1, S+I]` for SDPA. Flux's pipeline does not propagate any mask.
- Single-block stack receives **already-concatenated `[encoder_hidden_states; hidden_states]`** (concat done once before the loop, sliced off after). Flux's single block does the concat per-block.
- **No `swap_scale_shift`** for the final layer — the `norm_out` shift/scale come from the runtime modulation table, not from a checkpoint linear.

#### Existing C# scaffolding was broken — FIXED in this session
- `ChromaConfig.cs` previously said "Reuses FluxTransformer directly" with a `FluxConfig` field — wrong. With `GuidanceEmbed = false`, `FluxTransformer` builds `CombinedTimestepTextProjEmbeddings` (CLIP pooled + timestep MLP) and per-block `AdaLayerNormZero.linear`s — **none of these exist in Chroma's checkpoint**, so weight loading would have failed with hundreds of missing keys.
- `ChromaPipeline.cs` previously used `ClipTextEncoder` and extracted a CLIP-L pooled vector — wrong (Chroma is T5-only).
- **Both files have now been rewritten** with the correct architecture documented inline + clear `NotImplementedException` markers pointing at the implementation plan. The transformer + approximator + block variants + converter still need to be built — see the file list below.

#### Files (all built — total ~2200 lines)
- [x] [`ChromaConfig.cs`](../../src/SharpInference.Diffusion/Models/Denoisers/ChromaConfig.cs) — 69 lines. `Depth=19, DepthSingleBlocks=38, HiddenSize=3072, NumHeads=24, HeadDim=128, ApproximatorNumChannels=64, ApproximatorHiddenDim=5120, ApproximatorLayers=5, ApproximatorOutDim=3072`. `ModIndexLength = 3*N_single + 12*N_double + 2 = 344` exposed as a computed property. `V1` preset.
- [x] [`ChromaApproximator.cs`](../../src/SharpInference.Diffusion/Models/Denoisers/DiTBlocks/ChromaApproximator.cs) — 196 lines. 5-layer gated SiLU MLP with RMSNorm residuals. `distilled_guidance_layer.{in_proj,layers.{i}.linear_{1,2},norms.{i},out_proj}` weight keys.
- [x] [`ChromaCombinedTimestepEmbeddings.cs`](../../src/SharpInference.Diffusion/Models/Denoisers/DiTBlocks/ChromaCombinedTimestepEmbeddings.cs) — 114 lines. Builds the approximator input: `concat([Timesteps(t, 16), Timesteps(zero_guidance, 16), mod_proj_index_embed(arange(out_dim) * 1000, 32)])`.
- [x] [`ChromaDoubleStreamBlock.cs`](../../src/SharpInference.Diffusion/Models/Denoisers/DiTBlocks/ChromaDoubleStreamBlock.cs) (378 lines) and [`ChromaSingleStreamBlock.cs`](../../src/SharpInference.Diffusion/Models/Denoisers/DiTBlocks/ChromaSingleStreamBlock.cs) (260 lines) — pruned variants without per-block AdaLN linears; take precomputed `temb` rows directly as input. Single block sees pre-concatenated `[txt, img]` tokens.
- [x] [`ChromaTransformer.cs`](../../src/SharpInference.Diffusion/Models/Denoisers/ChromaTransformer.cs) — 418 lines. Top-level loop: approximator → table → per-block slice (12 rows for doubles, 3 for singles, 2 for final norm) → block forward → concat-then-single → slice off image → `norm_out` from last 2 rows → `proj_out`.
- [x] [`ChromaCheckpointConverter.cs`](../../src/SharpInference.ModelHandler/CheckpointConverters/ChromaCheckpointConverter.cs) — 311 lines. BFL-style single-file → diffusers naming + `distilled_guidance_layer.*` tree. No `swap_scale_shift` for final layer (because the final layer doesn't have a checkpoint linear).
- [x] [`ChromaPipeline.cs`](../../src/SharpInference.Diffusion/Pipelines/ChromaPipeline.cs) — 372 lines. T5-only encode, T5 attention mask plumbing with the "first padding token unmasked" trick, dual-pass CFG with default scale=5.0 + steps=35, FreeWeights eviction before VAE.
- [x] [`ChromaDebugDump.cs`](../../src/SharpInference.Diffusion/Models/Denoisers/ChromaDebugDump.cs) — 77 lines. Env var `CHROMA_DEBUG_DIR`.
- [x] End-to-end test + Python ref dump + diff harness — [`ChromaGenerationTests.cs`](../../tests/SharpInference.Diffusion.Tests/ChromaGenerationTests.cs), [`ChromaDiffTests.cs`](../../tests/SharpInference.Diffusion.Tests/ChromaDiffTests.cs), [`tests/python-reference/dump_chroma_full_forward.py`](../../tests/python-reference/dump_chroma_full_forward.py), [`tests/python-reference/diff_chroma_layers.py`](../../tests/python-reference/diff_chroma_layers.py).
- [ ] **End-to-end visual validation against the actual `lodestones/Chroma` checkpoint** — pending checkpoint download (BFL-style FP16 single-file ~17.8 GB, FP8 cast at load ~9 GB tight on 12 GB GPU).

#### Weights (12 GB target)
- BFL-style FP16 single-file (~17.8 GB) — won't fit at FP16, FP8 cast at load (~9 GB) tight but fits
- GGUF Q4/Q5 (~5-6 GB) — comfortable, **requires GGUF K-quant reader**
- T5-XXL Q8/Q4 (similar to Flux/SD3 setup) — already supported

#### Common blocker (RESOLVED 2026-05-06; full backend complete)
**GGUF backend is end-to-end production-ready.** See [`docs/Research/GGUF_BACKEND.md`](../Research/GGUF_BACKEND.md) for the full architecture and [`docs/Research/GGUF_QUANTIZER_USAGE.md`](../Research/GGUF_QUANTIZER_USAGE.md) for the user-facing converter guide.

**Read** (12 codecs implemented): Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL. Plus 8 more registered as DTypes pending codecs (Q8_K, IQ2/IQ3/IQ1 family, IQ4_XS, TQ1/TQ2). Reader successfully parses real city96 dumps end-to-end (validated against `flux1-schnell-Q4_K_S.gguf`).

**Write** (4 codecs implemented): Q8_0, Q4_K, Q5_K, Q6_K — backed by 5 mix policies (`Q8_0`, `Q4_K_S`, `Q4_K_M`, `Q5_K_M`, `Q6_K`). CLI at [`samples/ConvertSafetensorsToGguf/`](../../samples/ConvertSafetensorsToGguf/). Other write types (Q4_0, Q4_1, Q5_0, Q5_1, Q8_1, Q2_K, Q3_K, IQ4_NL, etc.) remain read-only — superseded by the K-quants in llama.cpp's mix policies, so authoring them isn't useful in practice. If a user truly needs them, they can use llama.cpp's `quantize` tool and we'll read the output.

**GPU dequant** (4 PTX kernels): Q8_0, Q4_K, Q5_K, Q6_K → F16 on-device, dispatched via `CudaBackend.Linear` → `CastIfNeeded` → `CastOnGpu` (Phase F wiring). Quantized weights stay quantized in VRAM; dequant happens per-call (~12 KB temp F16 buffer). Validated by [`FluxGgufLinearTests.Linear_RealQ4_K_FromCity96Gguf_ProducesSaneOutput`](../../tests/SharpInference.Diffusion.Tests/FluxGgufLinearTests.cs) — pulls a real Q4_K weight from city96's Flux Schnell GGUF and runs an end-to-end Linear with sane output.

**Per-architecture key mappers** (13 + passthrough): Flux, Flux2, SDXL, SD3, SD15, F-Lite, Chroma, AuraFlow, Z-Image, ERNIE-Image, Hunyuan-Image, Qwen-Image, Llama. `GgufConverterBridge.LoadGguf(path, F16, FluxCheckpointConverter.Convert)` is the one-line pipeline integration.

**Test count**: 55 ModelHandler GGUF tests + 8 CUDA dequant + Linear-quant tests, all green on .NET 8 + 10. Plus 2 real-checkpoint integration tests using `flux1-schnell-Q4_K_S.gguf` from city96.

### Hunyuan Image 2.1 (MMDiT)
- [x] `HunyuanImageConfig.cs` — 17B config with InChannels=32, V21/V21Distilled presets
- [x] `HunyuanImageTransformer.cs` — transformer scaffolding with PatchEmbed/Unpatchify
- [x] `HunyuanImagePipeline.cs` — pipeline with 32× downsample latent space

### Flux.2 (Klein 4B / Klein 9B / Dev)

**Architecture is significantly different from Flux.1 — not a config tweak, a separate transformer.**
Reference: `huggingface/diffusers` `src/diffusers/models/transformers/transformer_flux2.py` and `src/diffusers/pipelines/flux2/pipeline_flux2_klein.py`. Key differences vs Flux.1:
- LayerNorm (not RMSNorm) for stream norms; norms are `elementwise_affine=False` so no learned weight
- Modulation projections live at **top level**, not per-block — three shared Linears: `double_stream_modulation_img.linear`, `double_stream_modulation_txt.linear`, `single_stream_modulation.linear`
- Single-stream block is **parallel attention + MLP**: `linear1 = Q+K+V+gate+up` (split inside forward); `linear2` consumes `[attn_out, swiglu(gate, up)]` concat
- SwiGLU MLP via fused `linear_in: dim → 2*inner` then `chunk(2, dim=-1)` then `silu(x1) * x2 → linear_out`
- 4-axis RoPE `(32, 32, 32, 32)`, theta=2000 (Flux.1 uses 3 axes, theta=10000)
- Position-id format: `(T, H, W, L)` — text uses `(0,0,0,seq_idx)`, image uses `(0,row,col,0)`, ref images use `(10*idx,row,col,0)`
- 32-ch VAE latent + 2×2 patchify → 128 in_channels per token; VAE has `bn` (BatchNorm) post-decode and `quant_conv`/`post_quant_conv`
- Klein 4B: `hidden=3072, num_heads=24, head_dim=128, 5 doubles + 20 singles, mlp_ratio=3.0, joint_attention_dim=7680, no guidance_embeds`
- Klein 9B: `hidden=4096, num_heads=32, 8 doubles + 24 singles, joint_attention_dim=12288, no guidance_embeds`
- Dev: `hidden=6144, num_heads=48, 8 doubles + 48 singles, joint_attention_dim=15360, with guidance_embeds`

#### Done
- [x] `Flux2Config.cs` — verified Klein 4B preset (hidden=3072, heads=24, 5+20 blocks, in_channels=128, joint_attention_dim=7680, mlp_ratio=3.0, axes_dims_rope=(32,32,32,32), rope_theta=2000, guidance_embeds=false). Klein 9B and Dev presets updated to documented spec values.
- [x] `LlamaStyleEncoder.cs` + `LlamaStyleEncoderConfig.cs` — Llama-family decoder transformer used as text encoder. Qwen3-4B preset (36 layers, hidden=2560, GQA 32:8, head_dim=128, vocab=151936, RoPE θ=1M, per-head Q/K RMSNorm, SwiGLU MLP, causal mask, F32 weights post-cast). Validated: GPU forward on 64-token prompt = 3.0s, output `[B, S, 2560]`, no NaN/Inf, abs_mean ≈ 1.26.
- [x] `LlamaStyleEncoder.EncodeMultiLayer(backend, tokens, layerIndices)` — Klein-shaped multi-layer hidden-state collection. For Klein: pass `[9, 18, 27]` to get `[B, S, 3*2560 = 7680]` matching `context_embedder = Linear(7680, 3072)`. Validated: GPU forward = 2.4s, `[1, 64, 7680]`, no NaN/Inf, abs_mean ≈ 1.0.
- [x] `Qwen3Tokenizer.cs` — wraps `Microsoft.ML.Tokenizers.BpeTokenizer`. `EncodeChat(prompt)` produces the chat-templated form Klein needs.
- [x] `Flux2PosEmbed.cs` — 4-axis position-ID builder (T, H, W, L). Text tokens get `(0, 0, 0, seq_idx)`; image patches get `(0, row, col, 0)`. Rotation math is identical to Flux.1 (`repeat_interleave_real=True` with `use_real_unbind_dim=-1` reduces to pairwise `[2i, 2i+1]` rotation), so we delegate to `FluxRope` constructed with axes (32,32,32,32) and theta=2000.
- [x] `Flux2DoubleBlock.cs` — LayerNorm (no-affine) → modulate → joint [txt, img] attention with per-head Q/K RMSNorm + 4-axis RoPE → output proj + gated residual → LayerNorm + modulate + SwiGLU MLP + gated residual. Receives 6 pre-split modulation params per stream from the top-level shared modulation projection. No QKV bias.
- [x] `Flux2SingleBlock.cs` — parallel attention+MLP. Each block gets 5 separate matrices split from the fused `linear1` at converter time (Q, K, V, gate_proj, up_proj) plus the kept-fused `linear2` (input dim = hidden + mlp_inner). Forward: norm → modulate → 5 linears → QK-norm + RoPE + SDPA on attn path, SwiGLU `silu(gate)*up` on mlp path → concat → linear2 → gated residual.
- [x] `Flux2Transformer.cs` — top-level orchestrator. Owns the 3 shared modulation `AdaLNModulation`s (computed once per step from temb and reused across every block of that type), x_embedder + context_embedder Linears, time-only or time+guidance MLP, RoPE precompute, double-block loop, [txt, img] concat, single-block loop, strip text prefix, final AdaLN-Continuous + proj_out. Final-layer modulation expects [scale, shift] (diffusers convention) — the converter does the BFL→diffusers swap.
- [x] `Flux2CheckpointConverter.cs` — BFL→canonical key remap. Top-level: `time_in.{in,out}_layer` → `time_guidance_embed.timestep_embedder.linear_{1,2}`, `guidance_in.{in,out}_layer` → `time_guidance_embed.guidance_embedder.linear_{1,2}`, `img_in` → `x_embedder`, `txt_in` → `context_embedder`, `*_modulation*.lin` → `*_modulation*.linear`. Per-double-block: split `*_attn.qkv` into to_q/to_k/to_v (or add_q_proj/add_k_proj/add_v_proj for txt); split `*_mlp.0` SwiGLU `linear_in` into `ff.linear_in_gate` (rows 0..mlp_inner) and `ff.linear_in_up` (rows mlp_inner..2*mlp_inner); rename `*_mlp.2` → `ff.linear_out`. Per-single-block: split `linear1` into 5 separate weights (to_q/to_k/to_v + linear_in_gate + linear_in_up); rename `linear2` → `attn.to_out` (kept whole — it consumes the [attn_out || swiglu_out] concat). Final-layer `adaLN_modulation.1` row-swap to align BFL `[shift, scale]` with diffusers `[scale, shift]` (Flux.1 deviation #23 applied here too). Fp8 scale propagation through every split.
- [x] `VaeConfig.Flux2` updated: 32 latent channels, `UsePostQuantConv=true`, `UseQuantConv=true`, ScalingFactor=1.0 + ShiftFactor=null (the BN-style `bn.running_mean/var` un-normalize is applied at the pipeline boundary on the 128-channel patchified latent before unpatchify, not by the VAE decoder).
- [x] `Flux2Pipeline.cs` body — Qwen3 multi-layer encode → random `[B, 128, H/16, W/16]` noise (already in patchified form) → pack to `[B, S, 128]` → flow-match Euler with dynamic shift → denoising loop calling `Flux2Transformer.Forward` → unpack + BN un-normalize on the 128-channel patchified latent + 2×2 unpatchify → 32-channel `VaeDecoder.Decode` → RGB.
- [x] `Tensor.CastTo` — added BF16↔F16 (CPU) conversions so the Klein test can pre-cast BF16 weights to F16 before handing them to the Flux2 converter (matches the existing F16↔F32 GPU cast path; CudaBackend doesn't yet implement BF16↔F32 GPU casts, so a CPU pre-cast is required to keep the GEMM path functional with these checkpoints).
- [x] **Klein 4B end-to-end generation test** — `Flux2GenerationTests.Klein4B_GenerateImage_Gpu` (Linux, RTX 3060 12GB, 32GB RAM). Loads Klein 4B (BF16 → F16 cast for GPU compatibility), Qwen3-4B (BF16 → F16), Flux.2 VAE (F32). 512×512, 10 steps, seed=42, prompt "A photograph of an astronaut riding a horse". Total run time 2:53 (load+cast 60s, generate 109s @ ~10.4s/step, VAE decode 1.7s). Output: clean photorealistic image of an astronaut in a white spacesuit on a horse outdoors, sky + grass background — matches the prompt. No NaN/Inf, no degenerate output.
- [x] **Mistral-Small-3 encoder smoke test** — `MistralEncoderSmokeTests.Mistral_LoadAndForward_Gpu`. Loads `mistral_3_small_flux2_fp8.safetensors` (BFL distill, 30 layers, hidden=5120, GQA 32:8, FP8 mixed: BF16 embed table + 210 FP8 projection weights with per-tensor scale companions). `CheckpointConvertUtils.ApplyFp8ScaledDequant` handles BFL `.weight_scale`/`.input_scale` naming (in addition to ComfyUI's `.scale_weight`/`.scale_input`) and folds the scalar into `Tensor.Fp8ScaleFactor` for cuBLAS-alpha use. Encoder forward on a 32-token synthetic prompt: 13.7s GPU, output `[1, 32, 5120]`, no NaN/Inf, abs_mean=0.17. Validates `LlamaStyleEncoder` covers Mistral's variant of the architecture (no per-head Q/K norm, no final RMSNorm, both flagged via `LlamaStyleEncoderConfig.HasFinalNorm` / `QkHeadNorm`).
- [x] `LlamaStyleEncoderConfig.MistralSmall3` preset added; `LlamaStyleEncoder.LoadWeights` and `Encode` now skip the final RMSNorm when `HasFinalNorm = false` (Mistral distill has no `model.norm.weight` — it ships as a feature extractor).
- [x] Tekken tokenizer is **embedded** in the Mistral safetensors as a U8 byte blob (`tekken_model`, ~19MB). The smoke test drops it before forward; a real pipeline would need a Tekken parser. Out of scope for this task.

#### Not yet runnable (out-of-scope blockers)

- [ ] **Klein 9B**: no checkpoint. Klein 9B was documented in our config but has not been released by Black Forest Labs as of 2026-04-27 — only Klein 4B is downloadable. Preset stays as a documented placeholder; no test possible without weights.
- [ ] **Flux.2 Dev (32B) end-to-end generation**: blocked by VRAM. The checkpoint is 33GB on disk (29GB FP8 MLP weights + 6.5GB BF16 attention weights, 8 doubles + 48 singles, hidden=6144, 48 heads). Running this on a 12GB GPU requires either:
  - Per-block weight streaming/eviction in `CudaBackend` / `GpuTransferHelper` — currently weights are uploaded once and pinned in VRAM. A streaming path would upload one block's weights, run, free, then load the next block. Cost: several days of backend work; runtime would be I/O-bound (PCIe transfers per step × 56 blocks × N steps).
  - Q4_K quantization of the FP8 weights — would shrink ~29GB → ~7-8GB, fitting comfortably in 12GB. Cost: implement Q4_K dequant kernels (we have research notes but no implementation), then validate accuracy vs reference.
  - Cloud GPU (A100 40GB / H100 80GB / L40S 48GB). $0.40-$3/hour depending on provider. Easiest path but ongoing cost.
  Until one of those lands, the Flux.2 Dev transformer + checkpoint converter sit ready but the end-to-end pipeline cannot be exercised on this machine. The Mistral encoder is unblocked — it's a smaller dependency and runs on the 12GB GPU.

### Flux.1 Tools + Kontext
- [x] `FluxToolsConfig.cs` — config for Fill/Redux/Canny/Depth/Kontext with AdditionalInChannels per variant

### F-Lite (Freepik / Fal.ai) — Single-stream cross-attention DiT — SCAFFOLD COMPLETE

Reference: [`fal-ai/f-lite`](https://github.com/fal-ai/f-lite) (`f_lite/model.py`, `f_lite/pipeline.py`) and [`Freepik/F-Lite/dit_model/config.json`](https://huggingface.co/Freepik/F-Lite/blob/main/dit_model/config.json). Architecture summarized in [`docs/Research/F_LITE_ARCHITECTURE.md`](../Research/F_LITE_ARCHITECTURE.md). License: CreativeML OpenRAIL-M.

Distinguishing features vs Flux:
- **Single-stream cross-attention DiT** (not joint dual-stream). 40 blocks each running self-attn → cross-attn → MLP, all gated by a 9-output AdaLN-Zero modulation.
- **V-residual across blocks**: block 0 produces a V; blocks 1..N mix in `lambda * v + (1-lambda) * v_0` with a learnable per-block lambda (init 0.5).
- **Non-interleaved 2D RoPE**: half-rotation `[x1*cos+x2*sin, -x1*sin+x2*cos]` (GPT-NeoX style), distinct from Flux's pairwise rotation. Precomputed for a 512×512 grid.
- **16 register tokens** prepended before image patches (similar to AuraFlow). RoPE rows for register slots use `cos=1, sin=0` (identity rotation).
- **No biases on QKV/MLP/proj linears, no learned RMS scale** when `train_bias_and_rms=false` (the public 10B). The only biases are on `patch_embed`, `time_embed`, `adaLN_modulation`, `final_modulation`, `final_proj`.
- **T5-XXL layer 17** (not the final layer) is fed into cross-attention, with `final_layer_norm` re-applied after intermediate-layer extraction. F-Lite is the first model in this codebase to need T5 intermediate-layer access.
- **Inline dynamic-shift flow-match scheduler**: `alpha = 2 * sqrt(image_token_count / (64*64))`, `t_shifted = t * alpha / (1 + (alpha-1)*t)`. No `IScheduler` shell — implemented directly in `FLitePipeline.GenerateFromTokens`.

#### Files (all built — total ~1700 lines + research doc)
- [x] [`FLiteConfig.cs`](../../src/SharpInference.Diffusion/Models/Denoisers/FLiteConfig.cs) — V1 (10B), V1_7B (placeholder pending 7B repo config), Texture (alias for V1).
- [x] [`FLiteRope.cs`](../../src/SharpInference.Diffusion/Models/Denoisers/DiTBlocks/FLiteRope.cs) — non-interleaved 2D rotary, 512×512 precompute, register-token identity-rotation padding.
- [x] [`FLiteAttention.cs`](../../src/SharpInference.Diffusion/Models/Denoisers/DiTBlocks/FLiteAttention.cs) — `SelfAttn` and `CrossAttn` factories. Self path: fused QKV → multi-head reshape → V-residual mix → RoPE → per-head QK-RMSNorm → SDPA → proj. Cross path: separate Q + KV linears, no RoPE, no V-residual.
- [x] [`FLiteBlock.cs`](../../src/SharpInference.Diffusion/Models/Denoisers/DiTBlocks/FLiteBlock.cs) — 9-output AdaLN modulation, three sub-paths (SA/CA/MLP) with shift+scale before, gate after. RMSNorm-without-affine when `train_bias_and_rms=false`.
- [x] [`FLiteTransformer.cs`](../../src/SharpInference.Diffusion/Models/Denoisers/FLiteTransformer.cs) — patch_embed → register-token concat → 2D RoPE build → time embed → 40 blocks (with V-residual carry-forward) → strip register tokens → final RMSNorm + modulate → proj_out → unpatchify.
- [x] [`FLiteCheckpointConverter.cs`](../../src/SharpInference.ModelHandler/CheckpointConverters/FLiteCheckpointConverter.cs) — diffusers folder layout (`{root}/dit_model`, `{root}/text_encoder`, `{root}/vae`) loader. Multi-shard support per component. No key remap — F-Lite ships in canonical diffusers naming.
- [x] [`FLitePipeline.cs`](../../src/SharpInference.Diffusion/Pipelines/FLitePipeline.cs) — T5-XXL layer-17 encode (positive + negative), inline dynamic-shift flow-match denoise loop, CFG via dual transformer pass (chosen over batch-of-2 to keep peak activation memory lower on 12 GB GPUs), Flux Schnell VAE decode with `latent / scale + shift` un-normalization.
- [x] [`T5TextEncoder.EncodeAtLayer`](../../src/SharpInference.Diffusion/Models/TextEncoders/T5TextEncoder.cs) — new public method: runs the encoder for the first N blocks, optionally re-applying the final RMSNorm. Used by F-Lite (layer=17) and reusable for any future model that needs intermediate-layer T5 conditioning.
- [x] 8 unit tests in [`FLiteTests.cs`](../../tests/SharpInference.Diffusion.Tests/FLiteTests.cs) — config preset values, RoPE register-token identity rows, RoPE oversize-grid rejection, transformer construction without weights, missing-key throw on LoadWeights.

#### Pending
- [ ] **Download checkpoint** — `Freepik/F-Lite` from HuggingFace (29.4 GB across `dit_model/` + `text_encoder/` + `vae/`). Free of HF gating per the OpenRAIL-M license.
- [ ] **First-run debugging** — based on past new-model debugging in this repo (5+ pipeline-level bugs unmasked on the SD3.5 / Z-Image first runs), expect 1-3 iterations of bug fixes. Suspected hot spots: (1) RoPE half-rotation sign convention vs the GPT-NeoX reference, (2) the `train_bias_and_rms=false` weight-loading path silently skipping a key that's actually present in the checkpoint, (3) T5 layer-17 indexing (1-based vs 0-based off-by-one), (4) AdaLN 9-chunk extraction order matching the reference's `chunk(9, dim=1)` exactly.
- [ ] **Python reference dump + C# diff harness** — `dump_flite_full_forward.py` mirroring `dump_sd35_full_forward.py` + `diff_flite_layers.py` + `FLiteDiffTests.cs`. Defer until checkpoint downloaded.
- [ ] **End-to-end SSIM gate** — once first-run produces a clean image, add to `tests/python-reference/dump_flite_reference_image.py` and a `FLiteSsimTests.cs` mirroring the SDXL/Flux pattern.
- [ ] **F-Lite-7B + F-Lite-Texture variants** — same backbone architecturally; need actual `config.json` for the 7B to confirm the V1_7B preset's placeholder dimensions.

#### Weights (12 GB target)
- BFL-style FP16 (~20 GB transformer + 9.5 GB T5 + 0.08 GB VAE = ~29.5 GB) — won't fit at FP16, FP8 cast at load (~10 GB transformer + 4.7 GB T5 + 0.08 GB VAE = ~14.8 GB) tight; **eviction discipline** (T5 freed after encode, transformer freed before VAE) brings peak below 12 GB.
- F-Lite-7B at FP8 — ~7 GB transformer; comfortable on 12 GB.

### Qwen-Image (MMDiT) — SCAFFOLD COMPLETE

Reference: `huggingface/diffusers` `src/diffusers/models/transformers/transformer_qwenimage.py` (993 lines), `src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py` (773 lines), `src/diffusers/models/autoencoders/autoencoder_kl_qwenimage.py` (1056 lines). Architecture is single-stream MMDiT with Qwen2.5-VL as text encoder, dual-stream `QwenImageTransformerBlock` (similar to Flux double-stream), and a 16-channel VAE distinct from Flux/SD3 VAEs. The flagship public weight is `Qwen/Qwen-Image` (20B parameters, ~40 GB BF16 / ~20 GB FP8 / ~6 GB Q4 for the transformer alone, plus ~15 GB for Qwen2.5-VL-7B at BF16).

#### Files (all built — ~1419 lines + tests)
- [x] [`QwenImageConfig.cs`](../../src/SharpInference.Diffusion/Models/Denoisers/QwenImageConfig.cs) — `V1` preset (the canonical `Qwen/Qwen-Image` 20B: 60 layers, hidden=3072, 24 heads, head_dim=128). `V1_7B` retained as alias (the "7B" referred to Qwen2.5-VL-7B's text encoder, not the diffusion backbone). `V2_14B` / `V2_20B` left as speculative placeholders.
- [x] [`VaeConfig.QwenImage`](../../src/SharpInference.Diffusion/Models/Vae/VaeConfig.cs) — 16-channel preset with scaling=1.5305, shift=0.0609.
- [x] [`LlamaStyleEncoderConfig.Qwen2_5_VL_7B`](../../src/SharpInference.Diffusion/Models/TextEncoders/LlamaStyleEncoderConfig.cs) — 28 layers, hidden=3584, GQA 28:4, intermediate=18944, vocab=152064, AttentionBias=true. Reuses `LlamaStyleEncoder` for the text-only path.
- [x] [`QwenImageTransformer.cs`](../../src/SharpInference.Diffusion/Models/Denoisers/QwenImageTransformer.cs) (255 lines) — top-level rewrite: `img_in` Linear, `txt_norm` RMSNorm, `txt_in` Linear, sinusoidal+MLP timestep, 60-block stack, AdaLN-continuous final layer with `[shift, scale]` chunk order matching diffusers' `AdaLayerNormContinuous`, proj_out, unpatchify. EnumerateWeights for GPU preload. No `ForwardEdit` path (t2i only).
- [x] [`DiTBlocks/QwenImageBlock.cs`](../../src/SharpInference.Diffusion/Models/Denoisers/DiTBlocks/QwenImageBlock.cs) (289 lines) — dual-stream block: 6-output AdaLN modulation per stream, joint `[txt, img]` attention concat with QK-norm + per-stream RoPE applied before concat (matches `QwenDoubleStreamAttnProcessor2_0`), GELU-gated FFN, gated residuals.
- [x] [`DiTBlocks/QwenImageRope.cs`](../../src/SharpInference.Diffusion/Models/Denoisers/DiTBlocks/QwenImageRope.cs) (129 lines) — 3-axis [16, 56, 56] RoPE, applied per-stream before joint attention concat. Text positions at `max(H, W)` (scale_rope=False mode), distinct from `FluxRope`.
- [x] [`QwenImageDebugDump.cs`](../../src/SharpInference.Diffusion/Models/Denoisers/QwenImageDebugDump.cs) (74 lines) — env var `QWEN_IMAGE_DEBUG_DIR`, mirrors `Sd3DebugDump`.
- [x] [`QwenImagePipeline.cs`](../../src/SharpInference.Diffusion/Pipelines/QwenImagePipeline.cs) (293 lines) — Qwen3 BPE tokenizer → `LlamaStyleEncoder` Qwen2.5-VL encode → 2×2 patch pack/unpack → dynamic-shift flow-match Euler denoise loop → optional CFG dual-pass for `cfgScale > 1` → `_backend.FreeWeights(transformer + textEncoder)` before VAE decode (per PHASE_3_DEVIATIONS #18, #33) → 16-channel `VaeDecoder.Decode` with `(latent / scaling_factor) + shift_factor` un-normalize at pipeline boundary.
- [x] [`QwenImageCheckpointConverter.cs`](../../src/SharpInference.ModelHandler/CheckpointConverters/QwenImageCheckpointConverter.cs) (186 lines) — diffusers single-file → internal naming with `transformer.*` / `text_encoder.*` / `vae.*` bucketing. Fused-QKV split for `attn.qkv` and `attn.added_qkv` (defensive — none observed in stock checkpoints, mirrors Flux/SD3 split helpers). FP8 scale-companion folding via shared `CheckpointConvertUtils.ApplyFp8ScaledDequant`.
- [x] **End-to-end test scaffolding** — [`QwenImageGenerationTests.cs`](../../tests/SharpInference.Diffusion.Tests/QwenImageGenerationTests.cs) at 512×512, 8 steps, no-CFG smoke test. Skips cleanly when `TestPaths.QwenImage.{V1, TextEncoder, Vae}` checkpoints, the Qwen3 BPE tokenizer assets, or the PTX directory are missing. **VRAM probe** via `backend.Context.GetMemoryInfo()` skips when free VRAM < 22 GB (FP8 stock = ~20 GB transformer + 15 GB encoder).

#### Pending (validation only — no code work)
- [ ] **End-to-end visual validation** — needs ≥22 GB free VRAM (A100 40 GB, L40S 48 GB) for FP8 stock weights, OR a Q4_K GGUF dump (`city96/Qwen-Image-gguf` or `unsloth/Qwen-Image-gguf` if/when available — transformer ~6 GB + Qwen2.5-VL Q8 ~4 GB = 10 GB tight on RTX 3060). The implementation is complete; what's missing is a single clean run.
- [ ] **Layer-by-layer Python reference dump + diff harness** — author `dump_qwenimage_full_forward.py` mirroring `dump_sd35_full_forward.py` once a host can load the checkpoint. The `QwenImageDebugDump` hook is wired and ready.
- [ ] **Qwen-Image-Edit branch** — diffusers exposes a `forward_edit` editing path; intentionally omitted from this scaffold per t2i-first scope. Add later if user requests image-conditioned editing.

### Z-Image (Lumina2/NextDiT)
SOTA DiT by Tongyi Lab. Apache 2.0. **Lumina2/NextDiT architecture, not Flux-lineage** — sub-components (`SwiGluFfn`, `QkNorm`, `AdaLNModulation`) reusable, but new top-level transformer needed. 30 main layers + 2 noise + 2 context refiners, RMSNorm everywhere, multi-axis RoPE [32,48,48] θ=256, AdaLN with 4 outputs. Uses Qwen3-4B (full causal LM) as text encoder + Flux VAE verbatim. Z-Image-Turbo (8-NFE distilled, no CFG) and Z-Image-Base (28–50 steps with CFG=3..5, released 2026-01-28) share the exact same transformer architecture; only `SchedulerShift` and the sampling regime differ. See `docs/Research/Z_IMAGE_ARCHITECTURE.md`.

- [x] `ZImageConfig.cs` — Turbo preset (`SchedulerShift=3.0`) and Base preset (`SchedulerShift=6.0`); architecturally identical otherwise (dim=3840, 30 layers, 2+2 refiners, FfnDim=10240). Auto-detect via fused `layers.0.attention.qkv.weight` shape + `layers.{i}` count
- [x] `ZImageRope.cs` — multi-axis RoPE [32, 48, 48], θ=256. `BuildPositionIds(capPaddedLen, hPacked, wPacked, imgPaddedLen)` covers concat sequence; image-only refiner pass uses `BuildPositionIds(0, ...)`
- [x] `ZImageContextRefinerBlock.cs` — RMSNorm + QK-norm MHA + RMSNorm + SwiGLU FFN (no AdaLN). Fused QKV. **Applies caption-token RoPE** (frame=1..capPaddedLen, h=w=0) — diffusers' `ZImageTransformerBlock` always uses freqs_cis even with `modulation=False`. See `PHASE_3_DEVIATIONS.md` #28.
- [x] `ZImageBlock.cs` — handles both `noise_refiner` and main `layers` (structurally identical). Fused QKV. RmsNorm1 → scale_msa → MHA(QK-norm + optional RoPE) → RmsNorm2 → gate_msa residual → RmsNorm1 → scale_mlp → SwiGLU → RmsNorm2 → gate_mlp residual. AdaLN: `Linear(t_emb)` → split-4 (no SiLU here; t_embedder already SiLU's internally)
- [x] `ZImageTransformer.cs` — top-level orchestrator. `t_embedder.mlp.{0,2}` (sinusoidal × 1000 → Linear(256→1024) → SiLU → Linear(1024→256)), `cap_embedder` (RMSNorm + Linear 2560→3840), `x_embedder` (patch Linear 64→3840), context_refiner stack on caption (with caption RoPE), noise_refiner stack on padded image (image-only RoPE), concat, 30× main `layers` with full RoPE, `final_layer` (SiLU → Linear scale-only → LayerNormNoAffine → modulate → Linear → unpatchify). Pads cap and image to multiples of 32 with `cap_pad_token` / `x_pad_token`. **All 38 layers verified within F32 noise floor (max err 4.5e-5 at layers.29)** vs diffusers reference via `tests/python-reference/dump_zimage_full_forward.py` + `ZImageDiffTests.Transformer_Matches_PythonReference_LayerByLayer`
- [x] Qwen3-4B text encoder — uses existing `LlamaStyleEncoder` + `Qwen3Tokenizer.EncodeChat` (chat template) + `EncodeMultiLayer(..., [NumLayers-1])` for diffusers' `hidden_states[-2]` penultimate layer. `LlamaStyleEncoder.NumLayers` accessor added for tests
- [x] `ZImageCheckpointConverter.cs` — partitions transformer / VAE / text-encoder buckets by key prefix. Strips `model.diffusion_model.` and `transformer.` wrappers. Folds ComfyUI fp8_scaled `.weight_scale` companions into `Tensor.Fp8ScaleFactor`; drops `.comfy_quant` metadata blobs. `DetectArchitecture()` returns (numLayers, numRefiner, hidden, ffnDim, isFp8Mix); FP8Mix detected by dtype of `layers.0.attention.qkv.weight`
- [x] `ZImagePipeline.cs` — accepts pre-computed Qwen embeddings. Static-shift flow-match Euler from `ZImageConfig.SchedulerShift`. Single forward per step at `cfgScale=1.0` (Turbo) **or** dual cond/uncond forward at `cfgScale>1.0` with `negativeCaptionEmbeddings` (Base). Inverts `(latent - shift) * scale` before VAE decode (Flux VAE shift=0.1159, scale=0.3611)
- [x] End-to-end Turbo generation test — **CLEAN PHOTOREAL OUTPUT**. 512×512, 8 NFE, CFG=1.0, seed=42 produces sharp astronaut-on-horse-on-moon. ~2:40 wall on RTX 3060 (~16s/step + 24s Qwen3 encode + 2s VAE). All bugs documented in `PHASE_3_DEVIATIONS.md` #25–29: single-file naming (#25), dispose-after-noop pad (#26), 8 pipeline semantics (#27), missing context_refiner RoPE (#28), caption encoding pipeline triple-bug (#29). Layer-by-layer F32 diff harness at `tests/python-reference/dump_zimage_full_forward.py` + `tests/python-reference/diff_zimage_layers.py` — used `Z_IMAGE_DEBUG_DIR` env var to dump every layer through `ZImageTransformer.Forward`
- [x] End-to-end Base generation test — **CLEAN PHOTOREAL OUTPUT**. 512×512, 28 steps, CFG=4.0 with negative prompt and `RamonGuthrie/z_image_base-nvfp8-mixed` checkpoint produces sharp astronaut-on-horse-on-moon. ~15 min wall on RTX 3060. Required two fixes: (1) `cast_bf16_f32.ptx` PTX kernel for BF16↔F32 GPU cast (BF16 norms in nvfp8-mixed weren't supported by `CudaBackend.CastOnGpu` previously), and (2) `LoadAsF32` cast at weight-load for all `RmsNorm` scales (`CudaBackend.RmsNorm` reads `weight.DataPointer` as `float*` without dtype check, so BF16 norms got bit-reinterpreted as garbage F32 — manifested as a 16-px-grid of disconnected color blobs because per-channel scales randomized across layers broke cross-patch attention coherence). See `PHASE_3_DEVIATIONS.md` #30.

### SDXL Inpaint
- [x] `SdxlInpaintPipeline.cs` — 9-channel UNet pipeline with DownsampleMask helper

### Adapters
- [x] `ControlNetConfig.cs` — config with ControlNetBaseModel enum (Sd15/Sdxl/Flux), ControlNetMode enum
- [x] `ControlNet.cs` — adapter with hint encoder convs, zero conv arrays, Forward returning residuals
- [x] `IpAdapterConfig.cs` — config with IpAdapterBaseModel enum, NumImageTokens, IsPlus/IsFaceId flags
- [x] `IpAdapter.cs` — adapter with per-layer K/V projections, image projection

### Schedulers
- [x] `LcmScheduler.cs` — IScheduler implementation with 1–4 step support

## 6. Testing & Validation

- [x] SDXL dual CLIP conditioning verified, SD1.5/SDXL single-file checkpoint conversion tested
- [x] SD1.5 + SDXL converted UNet forward passes: no NaN/Inf, exhaustive key validation
- [x] SDXL GPU UNet forward: avg_err=5.510E-007, max_err=8.821E-006 (vs CPU reference)
- [x] SDXL F32 GPU 256x256 image generation: passes, ~4.2s/step
- [x] SDXL F32 GPU 1024x1024 image generation: passes, ~62s/step
- [x] SDXL F16 GPU 256x256 image generation: passes, ~580ms/step (7.2x speedup)
- [x] SDXL F16 GPU 1024x1024 image generation: passes, ~5.5s/step (11x speedup), 173s total for 20 steps
- [x] SDXL GPU performance target: <5s/step achieved with F16 at 256x256. 1024x1024 at 5.5s/step (close to target)
- [x] Flux weight loading tests: all 6 pass (transformer, CLIP-L, T5, VAE, architecture detect, full pipeline load)
- [x] SDXL pipeline SSIM gate vs diffusers — [`SdxlSsimTests.cs`](../../tests/SharpInference.Diffusion.Tests/SdxlSsimTests.cs) loads a reference PNG produced by [`tests/python-reference/dump_sdxl_reference_image.py`](../../tests/python-reference/dump_sdxl_reference_image.py) and computes SSIM via the existing [`Helpers/Ssim.cs`](../../tests/SharpInference.Diffusion.Tests/Helpers/Ssim.cs). **Current threshold = 0.30 (loose)** because C# `SeedGenerator.CreateNoise` uses a different RNG from PyTorch — even with matching seeds, the initial latent noise differs, so identical model weights produce different images. The 0.30 gate still catches catastrophic regressions (NaN propagation, broken VAE, wrong scheduler). To tighten to the original 0.95 target: plumb a `Tensor? initialNoise = null` parameter through `SdxlPipeline.GenerateFromTokens`, have the test load `init_noise_seed42.bin` (already dumped by the Python script), pass through. Tracked as future work in the test file.
- [x] Refiner handoff test — [`SdxlRefinerPipelineTests.cs`](../../tests/SharpInference.Diffusion.Tests/SdxlRefinerPipelineTests.cs) covers strength=0 byte-identical pass-through. The pixel-space refiner accepts any RGB tensor (any base pipeline's output); a separate handoff test isn't needed because the API is "feed RGB in, get RGB out".
- [x] Flux pipeline SSIM gate (Dev + Schnell 4-step) + T5 encoder validation — [`FluxSsimTests.cs`](../../tests/SharpInference.Diffusion.Tests/FluxSsimTests.cs) covers Dev (10 steps cfg=3.5) and Schnell (4 steps cfg=0.0). [`T5EncoderDiffTests.cs`](../../tests/SharpInference.Diffusion.Tests/T5EncoderDiffTests.cs) compares C# T5-XXL hidden states vs HuggingFace transformers reference at avg_err < 1e-4 (CPU) / < 1e-3 (GPU). Reference dumps: [`dump_flux_reference_image.py`](../../tests/python-reference/dump_flux_reference_image.py), [`dump_t5_xxl_hidden_states.py`](../../tests/python-reference/dump_t5_xxl_hidden_states.py). Same RNG-mismatch caveat as SDXL — tests skip cleanly when reference data is missing.
- [x] LoRA apply/stack tests — 31 unit tests in [tests/SharpInference.ModelHandler.Tests/LoraFileTests.cs](../../tests/SharpInference.ModelHandler.Tests/LoraFileTests.cs) cover format detection, key transformation (placeholder substitution preserves compound identifiers like `down_blocks`, `to_q`, `single_transformer_blocks`), per-format mappers (Kohya SD1.5/SDXL/Flux, AI Toolkit Flux, Diffusers Flux), QKV split (3-way + 4-way), merge math (zero-base-weight delta verification, strength scaling, multi-LoRA accumulation), missing-target skip, and FP8-base rejection. End-to-end pipeline tests in [SdxlLoraGenerationTests.cs](../../tests/SharpInference.Diffusion.Tests/SdxlLoraGenerationTests.cs) and [FluxLoraGenerationTests.cs](../../tests/SharpInference.Diffusion.Tests/FluxLoraGenerationTests.cs) skip cleanly when `SDXL_LORA_PATH` / `FLUX_AITOOLKIT_LORA_PATH` / `FLUX_KOHYA_LORA_PATH` / `FLUX_DIFFUSERS_LORA_PATH` are not set. **Real-world validation:** `Flux_Lora_KeyCoverage_AgainstRealCheckpoint` ran against `ostris/yearbook-photo-flux-schnell-v1.safetensors` (an AI Toolkit-trained Flux Schnell style LoRA with 494 layers / 988 tensor entries). Format auto-detected as `DiffusersFlux`; **494/494 LoRA target keys mapped cleanly to real `FluxTransformer` weight keys (zero misses)**. Discovery: modern AI Toolkit (v0.1.0+) saves in F5 (Diffusers PEFT) format, not the F4 hybrid the trainer source suggested — the F4 detector is retained as a defensive fallback. Visual generation comparison still pending (Flux Schnell at F32 needs ~48 GB RAM; deferred to GPU-FP16 path or a smaller checkpoint). "Remove/unmerge" deferred to v2 (would need base-weight cache).
- [x] GGUF K-quant reader + generic GGUF backend — full architecture documented at [`docs/Research/GGUF_BACKEND.md`](../Research/GGUF_BACKEND.md). 5-layer design: GgufLoader (existing) → codec registry → key-mapper registry → GgufModelLoader → one-line bridge to existing `*CheckpointConverter.Convert`. **12 quant codecs implemented** (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/IQ4_NL); 8 more registered as DTypes pending codecs (Q8_K/IQ2_*/IQ3_*/IQ1_*/IQ4_XS/TQ*). **9 architecture key mappers** (flux/sdxl/sd3/sd15/flite/chroma/auraflow/zimage + passthrough fallback) auto-detect from GGUF metadata or tensor-name heuristic. Zero per-pipeline shim required — `GgufConverterBridge.LoadGguf(path, F16, FluxCheckpointConverter.Convert)` is the entire integration. 25 new ModelHandler tests (77/77 total green).
- [ ] GGUF Flux Q8_0 12 GB fit test — depends on a downloaded `flux-dev-Q8_0.gguf` from `city96/FLUX.1-dev-gguf`. Reader is now ready; the test just needs the checkpoint + a `FluxGgufLoader` shim that maps GGUF tensor names to FluxCheckpointConverter's expected diffusers naming.
- [x] CI workflows — [`.github/workflows/ci-cpu.yml`](../../.github/workflows/ci-cpu.yml) (Ubuntu + Windows, .NET 8 + 10, runs unit tests filtered to skip integration/GPU/SSIM/Vulkan suites) and [`.github/workflows/ci-gpu.yml`](../../.github/workflows/ci-gpu.yml) (self-hosted `cuda` runner, fast lane runs CUDA smoke + diff/SSIM tests on every push, slow lane runs end-to-end generation tests on manual dispatch only). All `*GenerationTests` already skip cleanly when env vars (`SHARPINFERENCE_MODELS_DIR` / individual `*_PATH`) are not set, so CI on a runner with a partial model set is OK.

## 7. Review & Merge

- [x] Code review prep — session changelog at [`docs/SESSION_CHANGELOG_2026-05-06.md`](../SESSION_CHANGELOG_2026-05-06.md) summarizes Phase 4 closeout work (22 new files, 8 modified, 38 tests added, all green).
- [x] Benchmark procedure — documented at [`docs/Research/BENCHMARKING.md`](../Research/BENCHMARKING.md). Reference matrix for SDXL F16 / Flux Dev FP8 / Flux Schnell FP8 / SD3.5 Medium / Z-Image Turbo. Actual it/s collection requires a paired SharpInference + ComfyUI run on the same hardware.
- [ ] Performance optimization — ongoing in [`docs/Research/CUDA_PERFORMANCE.md`](../Research/CUDA_PERFORMANCE.md). Native FP8 GEMM is wired (Ada+); next force-multiplier is kernel fusion (GroupNorm+SiLU, Conv2D+bias+activation).
- [ ] Merge to main branch — user action.

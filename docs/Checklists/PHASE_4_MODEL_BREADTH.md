# Phase 4 — Model Breadth (SDXL + Flux + FP8)

> **Goal:** Support SDXL and Flux model families, FP8 inference for large DiT models.
> **Packages:** SharpInference.Diffusion (extended), Core (DType), Cuda (FP8 kernels)

---

## 1. Research

- [x] SDXL_ARCHITECTURE, FLUX_ARCHITECTURE, LORA_FORMAT, T5_ARCHITECTURE
- [x] QUANTIZATION_DIFFUSION — comprehensive (FP8, GGUF Q8_0/Q4_K, mixed-precision strategy, quality presets)

## 2. Planning

- [x] SDXL UNet block structure mapped, shared code between SD1.5/SDXL/Flux identified
- [x] Flux DiT block structure (19 double + 38 single stream blocks for Dev/Schnell)
- [ ] T5-XXL memory strategy (Q8_0 for consumer GPUs)
- [ ] LoRA loading API and multi-LoRA stacking

## 3. Implementation — SDXL — COMPLETE (CPU + GPU)

- [x] `ClipTextEncoderG.cs` — reuses ClipTextEncoder with SdxlClipG preset + `EncodePenultimate()`
- [x] SDXL UNet — 3 levels [320,640,1280], heterogeneous transformer depth [1,2,10], 2048-dim cross-attn, `UseLinearProjection`
- [x] `AdditionEmbedding` — ADM micro-conditioning (6 scalars → sinusoidal → project to 1280-dim)
- [x] `SdxlPipeline.cs` — dual CLIP encode (CLIP-L + CLIP-G penultimate → [B,77,2048]), ADM, UNet, VAE
- [x] GPU weight preloading — `EnumerateWeights()` on all model classes, `PreloadWeights()` API, staged UNet+VAE loading
- [x] 1024x1024 GPU generation — integer overflow fixes (64-bit im2col), VaeAttention GPU-routed Linear
- [ ] `SdxlRefinerPipeline.cs` — refiner with base→refiner handoff

## 3b. Checkpoint Converters

- [x] `CheckpointConvertUtils.cs`, `Sd15CheckpointConverter.cs` (tested: v1-5-pruned-emaonly 4.0GB)
- [x] `SdxlCheckpointConverter.cs` (tested: JuggernautXL 6.7GB, OpenCLIP→HF remap + in_proj splitting)
- [x] `FluxCheckpointConverter.cs` — tested: flux1-schnell, architecture detection, key remapping + in_proj splitting
- [ ] `Sd3CheckpointConverter.cs` — **blocked**: needs MMDiT, T5

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
- [ ] End-to-end generation test — **blocked**: needs Flux Schnell F16 checkpoint download (~23GB)

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
- [ ] Native FP8 GEMM path via `cublasLtMatmul` with scaling (Ada/RTX 40xx+ SM 8.9+ only)

### Backend Integration
- [x] `CudaBackend` dtype dispatch: `ResolveGemmDtype()` maps FP8→F16, `CastOnGpu()` centralized GPU cast helper
- [x] MatMul, Linear, BatchedMatMul, Conv2D — all detect FP8 inputs, cast to F16 before cuBLAS GEMM
- [x] `GpuTransferHelper` — FP8 tensors stored at native 1-byte size in weight cache (half VRAM of F16)
- [x] `IBackend.CastF8E4M3ToF16()` / `CastF16ToF8E4M3()` — default CPU implementations + CudaBackend GPU overrides

### Pipeline Integration
- [ ] Mixed-precision pipeline: FP8 DiT backbone + FP16 VAE + FP16 CLIP (VAE must never be FP8)
- [ ] Quality presets matching QUANTIZATION_DIFFUSION.md recommendations:
  - `Maximum`: FP16 everything
  - `High`: FP8 backbone + FP16 VAE/encoders (default for large models)
  - `Medium`: Q8_0 backbone + FP8 T5 + FP16 VAE/CLIP
  - `Low`: Q4_K backbone + Q4_K T5 + FP16 VAE/CLIP

### Testing
- [x] FP8 CPU cast round-trip tests — 12 tests: E4M3↔F32, E4M3↔F16, E4M3↔BF16, E5M2↔F32, saturation, subnormals, DType properties
- [ ] FP8 GPU GEMM accuracy vs F16 GEMM (tolerance: avg_err < 1e-3)
- [ ] Flux.1-dev FP8 full pipeline: visually matches FP16 reference
- [ ] VRAM usage: confirm ~50% reduction vs FP16 for backbone weights
- [ ] Graceful fallback on Ampere GPUs (dequant path works, no crash)

## 5. Adapters

- [ ] `LoraLoader.cs`, `LoraManager.cs` (apply/remove/stack with strength weights)
- [ ] SD + Flux LoRA weight name mapping
- [ ] `ControlNetLoader.cs`, `IpAdapterLoader.cs` (stubs)

## 5b. Model Breadth — Scaffolding (configs, transformers, pipelines)

All items below are scaffolding with TODOs for backend/kernel logic. Forward passes throw `NotImplementedException` until blocks are implemented.

### Shared Utilities
- [x] `DiTUtils.cs` — shared static helpers (LayerNormNoAffine, SinusoidalTimestepEmbedding, linear projections, reshape/concat ops)
- [x] `VaeConfig` presets — Flux2, Chroma, AuraFlow, HunyuanImage, QwenImage

### Chroma (Flux Fork)
- [x] `ChromaConfig.cs` — wraps FluxConfig with standard CFG (not distilled-to-1)
- [x] `ChromaPipeline.cs` — full pipeline with dual forward pass for CFG

### AuraFlow (MMDiT)
- [x] `AuraFlowConfig.cs` — config with NumDoubleBlocks/NumSingleBlocks, V03 preset
- [x] `AuraFlowJointBlock.cs` — dual-stream joint block (image+text modulation, attention, QK-norm, SwiGLU FFN)
- [x] `AuraFlowSingleBlock.cs` — single-stream image-only block
- [x] `AuraFlowTransformer.cs` — full transformer with PatchEmbed, double+single blocks, timestep embedding
- [x] `AuraFlowPipeline.cs` — pipeline with T5-only text encoding, standard CFG

### Hunyuan Image 2.1 (MMDiT)
- [x] `HunyuanImageConfig.cs` — 17B config with InChannels=32, V21/V21Distilled presets
- [x] `HunyuanImageTransformer.cs` — transformer scaffolding with PatchEmbed/Unpatchify
- [x] `HunyuanImagePipeline.cs` — pipeline with 32× downsample latent space

### Flux.2
- [x] `Flux2Config.cs` — config with QkvBias=false, VaeDownscaleFactor=16, Mistral/Qwen text encoder enum
- [x] `Flux2Pipeline.cs` — pipeline accepting pre-computed text embeddings

### Flux.1 Tools + Kontext
- [x] `FluxToolsConfig.cs` — config for Fill/Redux/Canny/Depth/Kontext with AdditionalInChannels per variant

### Qwen-Image (MMDiT)
- [x] `QwenImageConfig.cs` — 7B–20B config with SupportsEditing flag, V1_7B/V2_14B/V2_20B presets
- [x] `QwenImageTransformer.cs` — transformer with Forward and ForwardEdit methods

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
- [ ] SDXL pipeline SSIM > 0.95 vs diffusers
- [ ] Refiner handoff test
- [ ] Flux pipeline SSIM > 0.95, Flux schnell 4-step, T5 encoder validation
- [ ] LoRA apply/remove/stack tests, GGUF Flux Q8_0 test, 12GB VRAM fit test
- [ ] All tests pass on GPU CI

## 7. Review & Merge

- [ ] Code review (shared code reuse, LoRA memory management)
- [ ] Benchmark SDXL/Flux it/s vs Python (target: within 2x of ComfyUI)
- [ ] Performance optimization: see `docs/Research/CUDA_PERFORMANCE.md`
- [ ] Merge to main branch

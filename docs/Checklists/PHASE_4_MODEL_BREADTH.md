# Phase 4 — Model Breadth (SDXL + Flux)

> **Goal:** Support SDXL and Flux model families.
> **Packages:** SharpInference.Diffusion (extended)

---

## 1. Research

- [x] SDXL_ARCHITECTURE, FLUX_ARCHITECTURE, LORA_FORMAT, T5_ARCHITECTURE
- [ ] QUANTIZATION_DIFFUSION — **still Draft**

## 2. Planning

- [x] SDXL UNet block structure mapped, shared code between SD1.5/SDXL/Flux identified
- [ ] Flux DiT block structure (double/single-stream counts)
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
- [ ] `FluxCheckpointConverter.cs` — **blocked**: needs DiT, T5, FlowMatchScheduler
- [ ] `Sd3CheckpointConverter.cs` — **blocked**: needs MMDiT, T5

## 4. Implementation — Flux

- [ ] `T5TextEncoder.cs` — T5-XXL encoder-only transformer
- [ ] `DoubleStreamBlock.cs`, `SingleStreamBlock.cs`, `MmDiTBlock.cs`
- [ ] `DiT.cs` — full Flux DiT (double→single stream blocks)
- [ ] RoPE for 2D image + text positions
- [ ] `FluxPipeline.cs` — T5+CLIP encode, flow-match denoise, VAE decode
- [ ] Flux guidance embedding

## 5. Adapters

- [ ] `LoraLoader.cs`, `LoraManager.cs` (apply/remove/stack with strength weights)
- [ ] SD + Flux LoRA weight name mapping
- [ ] `ControlNetLoader.cs`, `IpAdapterLoader.cs` (stubs)

## 6. Testing & Validation

- [x] SDXL dual CLIP conditioning verified, SD1.5/SDXL single-file checkpoint conversion tested
- [x] SD1.5 + SDXL converted UNet forward passes: no NaN/Inf, exhaustive key validation
- [x] SDXL GPU UNet forward: avg_err=5.510E-007, max_err=8.821E-006 (vs CPU reference)
- [x] SDXL F32 GPU 256x256 image generation: passes, ~4.2s/step
- [x] SDXL F32 GPU 1024x1024 image generation: passes, ~62s/step
- [x] SDXL F16 GPU 256x256 image generation: passes, ~580ms/step (7.2x speedup)
- [x] SDXL F16 GPU 1024x1024 image generation: passes, ~5.5s/step (11x speedup), 173s total for 20 steps
- [x] SDXL GPU performance target: <5s/step achieved with F16 at 256x256. 1024x1024 at 5.5s/step (close to target)
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

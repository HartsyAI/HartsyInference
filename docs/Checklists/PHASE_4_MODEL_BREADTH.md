# Phase 4 — Model Breadth (SDXL + Flux)

> **Goal:** Support the two most popular model families beyond SD1.5.
> **Packages:** SharpInference.Diffusion (extended)

---

## 1. Research

- [x] Complete [SDXL_ARCHITECTURE.md](../Research/SDXL_ARCHITECTURE.md) research — done and verified
- [x] Complete [FLUX_ARCHITECTURE.md](../Research/FLUX_ARCHITECTURE.md) research — done and verified
- [x] Complete [LORA_FORMAT.md](../Research/LORA_FORMAT.md) research — done and verified
- [x] Complete [T5_ARCHITECTURE.md](../Research/T5_ARCHITECTURE.md) research (full encoder) — done and verified
- [ ] Complete [QUANTIZATION_DIFFUSION.md](../Research/QUANTIZATION_DIFFUSION.md) research — **still Draft**

## 2. Planning

- [x] Map SDXL UNet block structure (channels, dual conditioning)
- [ ] Map Flux DiT block structure (double-stream vs single-stream counts)
- [ ] Plan T5-XXL memory strategy (Q8_0 for consumer GPUs)
- [ ] Plan LoRA loading API (load, apply, remove, adjust weight)
- [ ] Plan multi-LoRA stacking strategy
- [x] Identify shared code between SD1.5/SDXL/Flux UNet blocks
- [ ] Write agent instructions for Phase 4

## 3. Implementation — SDXL

- [x] `ClipTextEncoderG.cs` — CLIP-G reuses `ClipTextEncoder` with `ClipTextEncoderConfig.SdxlClipG` preset and `EncodePenultimate()` for penultimate hidden state + pooled output
- [x] SDXL UNet modifications — `UNetConfig.SdxlBase` preset with 3 levels [320,640,1280], heterogeneous transformer depth [1,2,10], 2048-dim cross-attention, `UseLinearProjection`, `TransformerLayersPerBlock`
- [x] SDXL conditioning — `AdditionEmbedding` implements ADM micro-conditioning (6 scalars → sinusoidal embed + pooled text → 2816-dim → project to 1280-dim)
- [x] `SdxlPipeline.cs` — dual CLIP encode (CLIP-L penultimate + CLIP-G penultimate concatenated → [B,77,2048]), ADM conditioning, SDXL UNet, VAE decode
- [ ] `SdxlRefinerPipeline.cs` — refiner UNet with base→refiner handoff
- [x] SDXL-specific VAE scaling factor (0.13025) — already in `VaeConfig.Sdxl`

## 3b. Implementation — Single-File Checkpoint Converters

- [x] `CheckpointConvertUtils.cs` — shared utilities: ResNet sub-key remapping, time_embed, mid_block, VAE key conversion, in_proj tensor splitting
- [x] `Sd15CheckpointConverter.cs` — single-file SD1.5 LDM → diffusers format (UNet 4-level, single CLIP-L via `cond_stage_model`, VAE). Tested against v1-5-pruned-emaonly.safetensors (4.0GB)
- [x] `SdxlCheckpointConverter.cs` — single-file SDXL LDM → diffusers format (UNet 3-level, CLIP-L via `conditioner.embedders.0`, CLIP-G via `conditioner.embedders.1` with OpenCLIP→HF remapping + in_proj splitting, VAE, ADM via `label_emb`). Tested against JuggernautXL (6.7GB)
- [ ] `FluxCheckpointConverter.cs` — **blocked**: requires DiT, T5TextEncoder, FlowMatchScheduler implementations
- [ ] `Sd3CheckpointConverter.cs` — **blocked**: requires MMDiT, T5TextEncoder implementations

## 4. Implementation — Flux

- [ ] `T5TextEncoder.cs` — full T5-XXL encoder-only transformer
- [ ] `DoubleStreamBlock.cs` — Flux double-stream joint attention
- [ ] `SingleStreamBlock.cs` — Flux single-stream concatenated attention
- [ ] `MmDiTBlock.cs` — shared MMDiT block structure (reusable for SD3)
- [ ] `DiT.cs` — full Flux DiT (double-stream blocks → single-stream blocks)
- [ ] RoPE implementation for 2D image positions + text positions
- [ ] `FluxPipeline.cs` — T5 + CLIP encode, flow-match denoise, VAE decode
- [ ] Flux guidance embedding (different from CFG dual pass)

## 5. Implementation — Adapters

- [ ] `LoraLoader.cs` — parse LoRA safetensors, extract A/B weight pairs
- [ ] `LoraManager.cs` — apply/remove/stack multiple LoRAs with strength weights
- [ ] SD LoRA weight name mapping (lora_unet_... → UNet parameter paths)
- [ ] Flux LoRA weight name mapping (different naming convention)
- [ ] `ControlNetLoader.cs` — load ControlNet weights (stub for Phase 2 models)
- [ ] `IpAdapterLoader.cs` — load IP-Adapter weights (stub)

## 6. Testing & Validation

- [ ] SDXL pipeline — fixed seed + prompt → visually identical to diffusers (SSIM > 0.95)
- [ ] SDXL refiner — base + refiner handoff produces expected quality improvement
- [x] SDXL dual CLIP — verify both encoders produce correct conditioning (SdxlWeightLoadingTests)
- [x] SD1.5 single-file checkpoint → convert → load all components (UNet, CLIP-L, VAE) — tested against v1-5-pruned-emaonly.safetensors
- [x] SDXL single-file checkpoint → convert → load all components (UNet, CLIP-L, CLIP-G, VAE) — tested against JuggernautXL
- [x] SD1.5 converted UNet forward pass — no NaN/Inf, reasonable statistics
- [x] SDXL converted UNet forward pass — no NaN/Inf, reasonable statistics
- [x] SD1.5 converted UNet has all expected diffusers keys (exhaustive key validation)
- [x] SDXL converted UNet has all expected diffusers keys (exhaustive key validation)
- [ ] Flux pipeline — fixed seed + prompt → visually identical to diffusers (SSIM > 0.95)
- [ ] Flux schnell (4-step) — verify fast generation works correctly
- [ ] T5 encoder — same tokens → same embeddings as HuggingFace (within 1e-3)
- [ ] LoRA — apply SD1.5 LoRA, verify weight delta is correct
- [ ] LoRA — apply Flux LoRA, verify weight delta is correct
- [ ] Multi-LoRA — stack two LoRAs, verify combined output
- [ ] GGUF Flux — load Q8_0 GGUF Flux model, compare output to FP16
- [ ] Memory usage — verify Flux Q8_0 fits in 12GB VRAM
- [ ] All tests pass on GPU CI

## 7. Review & Merge

- [ ] Code review — shared code reuse between pipeline variants
- [ ] Code review — LoRA memory management (proper cleanup on remove)
- [ ] Benchmark: SDXL it/s, Flux it/s, compare to Python
- [ ] Merge to main branch

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
- [x] LoRA loading API and multi-LoRA stacking — see § 5 Adapters

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

- [x] `LoraFile.cs`, `LoraStack.cs` — load + multi-LoRA stacking with per-LoRA strength. Source: [src/SharpInference.ModelHandler/Lora/](../../src/SharpInference.ModelHandler/Lora/). API: `LoraFile.Load(path)` returns format-detected layers; `LoraStack.Add(file, strength)` / `AddFromPath(...)`; `stack.ApplyToWeights(backend, unetWeights, transformerWeights, clipLWeights, clipGWeights)` mutates the dicts in place. CPU-side merge via `IBackend.MatMul` + `Scale` + `Add` against an F32 accumulator.
- [x] SD + Flux LoRA weight name mapping. Five formats supported with auto-detection: F1 Kohya SD1.5, F2 Kohya SDXL, F3 Kohya Flux (with 3-way fused-QKV split + 4-way fused-linear1 split), F4 **AI Toolkit Flux** (`lora_transformer_*` + `.lora_A/.lora_B`, primary path for ostris/ai-toolkit-trained LoRAs), F5 HF PEFT diffusers Flux. See [docs/Design/LORA_KEY_MAPPING.md](../Design/LORA_KEY_MAPPING.md) for the complete key transformation tables and format detection precedence.
- **Deferred to v2** (documented in LORA_KEY_MAPPING.md): LyCORIS LoHa / LoKr, DoRA `dora_scale`, XLabs Flux `processor.*`, LoCon `lora_mid.weight`, FP8-base + LoRA (rejected at apply with helpful error), Z-Image / Flux.2 / Qwen-Image LoRAs, dynamic strength changes after merge, LoRA "remove/unmerge" (would need base-weight cache).
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

### Qwen-Image (MMDiT)
- [x] `QwenImageConfig.cs` — 7B–20B config with SupportsEditing flag, V1_7B/V2_14B/V2_20B presets
- [x] `QwenImageTransformer.cs` — transformer with Forward and ForwardEdit methods

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
- [ ] SDXL pipeline SSIM > 0.95 vs diffusers
- [ ] Refiner handoff test
- [ ] Flux pipeline SSIM > 0.95, Flux schnell 4-step, T5 encoder validation
- [x] LoRA apply/stack tests — 31 unit tests in [tests/SharpInference.ModelHandler.Tests/LoraFileTests.cs](../../tests/SharpInference.ModelHandler.Tests/LoraFileTests.cs) cover format detection, key transformation (placeholder substitution preserves compound identifiers like `down_blocks`, `to_q`, `single_transformer_blocks`), per-format mappers (Kohya SD1.5/SDXL/Flux, AI Toolkit Flux, Diffusers Flux), QKV split (3-way + 4-way), merge math (zero-base-weight delta verification, strength scaling, multi-LoRA accumulation), missing-target skip, and FP8-base rejection. End-to-end pipeline tests in [SdxlLoraGenerationTests.cs](../../tests/SharpInference.Diffusion.Tests/SdxlLoraGenerationTests.cs) and [FluxLoraGenerationTests.cs](../../tests/SharpInference.Diffusion.Tests/FluxLoraGenerationTests.cs) skip cleanly when `SDXL_LORA_PATH` / `FLUX_AITOOLKIT_LORA_PATH` / `FLUX_KOHYA_LORA_PATH` / `FLUX_DIFFUSERS_LORA_PATH` are not set. **Real-world validation:** `Flux_Lora_KeyCoverage_AgainstRealCheckpoint` ran against `ostris/yearbook-photo-flux-schnell-v1.safetensors` (an AI Toolkit-trained Flux Schnell style LoRA with 494 layers / 988 tensor entries). Format auto-detected as `DiffusersFlux`; **494/494 LoRA target keys mapped cleanly to real `FluxTransformer` weight keys (zero misses)**. Discovery: modern AI Toolkit (v0.1.0+) saves in F5 (Diffusers PEFT) format, not the F4 hybrid the trainer source suggested — the F4 detector is retained as a defensive fallback. Visual generation comparison still pending (Flux Schnell at F32 needs ~48 GB RAM; deferred to GPU-FP16 path or a smaller checkpoint). "Remove/unmerge" deferred to v2 (would need base-weight cache).
- [ ] GGUF Flux Q8_0 test, 12GB VRAM fit test
- [ ] All tests pass on GPU CI

## 7. Review & Merge

- [ ] Code review (shared code reuse, LoRA memory management)
- [ ] Benchmark SDXL/Flux it/s vs Python (target: within 2x of ComfyUI)
- [ ] Performance optimization: see `docs/Research/CUDA_PERFORMANCE.md`
- [ ] Merge to main branch

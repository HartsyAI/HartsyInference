# Phase 9 — Video (LTX-Video + Wan + Lance)

> **Goal:** Video generation starting with LTX-Video, with Lance (ByteDance) layered on top once Phase 4's MoT + packed-attention infra is in place.
> **Packages:** HartsyInference.Video (+ shared 3D VAE / packed attention in HartsyInference.Diffusion / Core / per-backend kernels)

> **Status (2026-06-08): FIRST VIDEO MODEL LANDED — Lance T2V is built end-to-end** (structurally verified on CPU; numeric validation pending). This brought the reusable 3D-video foundation online: `CausalConv3d` (Conv2D-decomposed, all backends), the Wan2.2 3D causal VAE with streaming `feat_cache` decode, `Multimodal3DRope`, `Sincos3DPositionEmbedding`, frame-streaming output + ffmpeg/BMP encoders. **LTX-Video and Wan now reuse all of this** — they need only their own transformer + (for LTX) a different VAE. See § 6 Lance video. Remaining for Lance video: first-run numeric validation, the ViT (editing/3-way CFG), and the secondary-stream decode-overlap perf optimization.

---

## 1. Research

- [x] LTX-Video — [`LTX_VIDEO_ARCHITECTURE.md`](../Research/LTX_VIDEO_ARCHITECTURE.md). **T2V COMPLETE END-TO-END (2026-06-10)** — structurally CPU-verified, numerics validation-pending. DiT (`LtxVideoTransformer`/`LtxRope`/`LtxVideoBlock`, 28 layers, self-attn+3D RoPE / cross-attn to T5-XXL / AdaLN-Single) + base 3D VAE decoder (`LtxVideoVaeDecoder` + `LtxVaeResnetBlock3d`/`LtxVaeUpsampler3d`; released 0.9 config = non-causal, no timestep cond) + `LtxVideoPipeline` (flow-match Euler + 2-way CFG + frame streaming). ~16 LTX tests. Reuses T5/RMSNorm/SDPA/CausalConv3d (extended with non-causal symmetric-replicate padding)/frame-streaming. **Converter + generation entry DONE (2026-06-10):** `LtxVideoCheckpointConverter` (single-file `model.diffusion_model.*`/`vae.*` prefix routing + the diffusers rename tables verbatim — transformer `patchify_proj/adaln_single/q_norm/k_norm`, VAE flat `up_blocks.0..9`→`mid_block`/`conv_in`/`upsamplers`/`resnets` regroup, `per_channel_statistics`→`latents_mean/std`; diffusers-folder pass-through; FP8 fold) + `LtxVideoGenerationTests` (T5-XXL encode from the SD3.5 bundle like Chroma, VRAM probe ≥10 GB, BMP frame sequence). Remaining: timestep-conditioned 0.9.5 VAE variant + first-run numeric validation.
- [x] Wan-Video (Wan2.2 **TI2V-5B**) — [`WAN_VIDEO_ARCHITECTURE.md`](../Research/WAN_VIDEO_ARCHITECTURE.md). **T2V COMPLETE END-TO-END (2026-06-10)** — structurally CPU-verified, numerics validation-pending. DiT (`WanVideoTransformer`/`WanRope`/`WanVideoBlock`, 30 layers, self-attn+per-head 3D RoPE / cross-attn to umT5 / 6-param AdaLN, FP32 LayerNorms, Conv3d patchify) + `WanVideoPipeline` (latent-space flow-match Euler + 2-way CFG + frame streaming). **VAE reused directly** — TI2V-5B's `AutoencoderKLWan` (z=48, 16×/4×) IS the already-built `Wan22VaeDecoder`. **Converter + umT5 entry + I2V DONE (2026-06-10):** `WanVideoCheckpointConverter` (original/Comfy naming → diffusers via the `convert_wan_to_diffusers.py` rename table verbatim incl. the norm2/norm3 swap; diffusers-folder pass-through; FP8 fold; VAE loads via `LanceCheckpointConverter.LoadVae`) + `T5TextEncoderConfig.Umt5Xxl` preset (per-layer relative bias, 256k vocab) + `WanVideoGenerationTests` (umT5 encode → free → DiT, VRAM probe ≥14 GB) + **I2V variant** (diffusers `expand_timesteps` path: per-latent-frame timesteps through all AdaLN modulation + final layer, first-frame-latent imposition on the model input each step + once post-loop; `GenerateFromEmbeddings(..., firstFrameLatent:)`). ~8 Wan tests (incl. uniform-per-frame == scalar exact-match gate). Remaining: first-run numeric validation + `Wan22VaeEncoder` (RGB-input I2V currently needs an offline-encoded, `Wan22VaeLatentNorm`-normalized first-frame latent).
- [x] Temporal attention, video VAE decoder (3D convolutions), video output encoding — done across Lance/LTX/Wan (CausalConv3d + 3D VAE decoders + `IVideoEncoder` ffmpeg/BMP + frame streaming).
- [x] [LANCE_ARCHITECTURE](../Research/LANCE_ARCHITECTURE.md) — `Lance_3B_Video` unified multimodal model (T2I + T2V + edit + understanding in one 3B-active backbone). Wan2.2 3D causal VAE (z=48, 16× spatial / 4× temporal), MoT dual-stream LLM, MaPE on M-RoPE, 3-way CFG. Shares the entire transformer + tokenizer + VAE + connectors with the Phase 4 image variant — only `model.safetensors` differs.

## 2. Planning

- [ ] LTX-Video model structure/weights, temporal attention integration
- [ ] VRAM management for video, frame streaming output, video encoding (FFmpeg or pure C# mp4)
- [ ] **Lance video reuse** — confirm that the Phase 4 `LanceTransformer` / `LanceMoTBlock` / `LanceMRopeMaPE` / `Qwen25VlVit` / `Wan22VaeDecoder` modules carry over byte-identically and that the video pipeline only adds frame-streaming + multi-frame CausalConv3d state management on top.

## 3. Shared Infrastructure (cross-cutting prereqs for all video + interactive models)

These items are required by Lance (Phase 4 image + Phase 9 video), LTX-Video, Wan, Cosmos-Predict V2W, **and** the Phase 10 interactive world models. The canonical task list is whichever model lands first. Lance image in Phase 4 § Lance is the forcing function that lights up the first three; the rest are introduced as the corresponding video/interactive model needs them.

See [`docs/Research/INTERACTIVE_INFERENCE.md`](../Research/INTERACTIVE_INFERENCE.md) for the full design rationale of every shared abstraction below.

### 3a. Backend kernels (cross-vendor)

- [ ] **`IBackend.PackedAttention(q, k, v, cu_seqlens, max_seqlen, block_mask, ...)`** — variable-length / packed attention. First pass: padded dense + bool mask (correct, slow on long sequences). Follow-up: real varlen FlashAttention-style CUDA PTX kernel. Shared with Lance + LTX + Cosmos-AR + any future packed-sequence model.
- [x] **3D Convolution — delivered WITHOUT a new `IBackend.Conv3D`** (2026-06-08, Lance image build). [`CausalConv3d`](../../src/HartsyInference.Diffusion/Models/Vae/CausalConv3d.cs) decomposes a 3D causal conv into existing `Conv2D` calls over temporal taps, so it runs on CPU/CUDA/Vulkan today with no new kernel. Reusable by LTX/Wan/Hunyuan video VAEs + Cosmos. A native `IBackend.Conv3D` (3D im2col+GEMM) remains an optional perf follow-up. 3 unit tests.
- [x] **`CausalConv3d` streaming wrapper** — DONE (2026-06-08, Lance video V1). [`Wan22StreamCache`](../../src/HartsyInference.Diffusion/Models/Vae/Wan22StreamCache.cs) threads the per-conv `CACHE_T=2` cache (`StepConv` + the Resample `time_conv` "Rep"/first-chunk `StepTimeConv`) across the per-latent-frame decode loop in `Wan22VaeDecoder`. Verified streamed == full-clip (byte-identical).

### 3b. Conditioning (shared across video + interactive)

- [x] **`HartsyInference.Conditioning/IActionEncoder.cs`** — DONE (2026-06-10, Matrix-Game 3.0 build): [`ActionInput`](../../src/HartsyInference.Diffusion/Conditioning/ActionInput.cs) + multi-stream [`IActionEncoder`](../../src/HartsyInference.Diffusion/Conditioning/IActionEncoder.cs) (`ActionStreamSpec`/`ActionStreamRole`) in `HartsyInference.Diffusion/Conditioning/`. First per-model encoder: `MatrixGame3ActionEncoder` (Interactive). Oasis/GameCraft encoders land with their models.
- [x] **3D sin-cos position embedding** — delivered as [`DiTUtils.Sincos3DPositionEmbedding`](../../src/HartsyInference.Diffusion/Models/Denoisers/DiTBlocks/DiTUtils.cs) (2026-06-08, Lance image build). Frozen, CPU precompute, generic/reusable. Also landed: generic [`Multimodal3DRope`](../../src/HartsyInference.Diffusion/Models/Denoisers/DiTBlocks/Multimodal3DRope.cs) (Qwen2.5-VL M-RoPE, reusable across VL-family video models).

### 3c. Caches + schedulers

- [ ] **`HartsyInference.Diffusion/Utilities/DenoiseKvCache.cs`** — first inference-time KV-cache use case. Caches K/V for the (text + action + clean-cond) prefix on step 0; recomputes only the noisy slot on steps 1..N. Reusable across Lance / LTX / Wan video / Matrix-Game / GameCraft (~2-3× wall speedup). AR-token variant (Cosmos AR / future MineWorld) deferred.
- [ ] **`HartsyInference.Diffusion/Schedulers/DistilledFlowMatchEuler.cs`** — extends flow-match Euler with **DMD** (Distribution Matching Distillation, 3-4 step — Matrix-Game 2/3), **CM** (Consistency Models, 1-4 step), **Lightning** (1-4 step). Wired into `SchedulerFactory`. Required for any model running below the offline 30-step regime.
- [x] **`HartsyInference.Diffusion/Schedulers/DdimVPredScheduler.cs`** — DONE (2026-06-10, Oasis build): sigmoid β-schedule (T=1000, −3..3) + per-frame v-param DDIM step with <c>t = −1</c> clean state. Diffusion Forcing (context pinned at stabilization 14, target stepped) is realized at the pipeline level via the per-frame index API — see `OasisPipeline`. Gated by analytic exactness tests.

### 3d. Streaming + tokenizers

- [ ] **`HartsyInference.Video/Tokenizers/IDiscreteVideoTokenizer.cs`** — interface for discrete video codecs (encode RGB → int32 indices; decode indices → RGB; embedding lookup). First implementation `CosmosDvTokenizer` lands with Cosmos-Predict V2W. **VQ-GAN / MagViT-v2 / FSQ variants deferred** until a model that needs them lands (Oasis uses a continuous Gaussian VAE, not discrete).
- [x] **Per-frame streaming VAE decode** — DONE as `Wan22VaeDecoder.DecodeStreaming` (yields one RGB frame-group per latent frame; memory bounded to a single group). A secondary-CUDA-stream overlap of decode with denoise is a perf follow-up (denoise is joint across frames, so only the decode stage streams).
- [x] **Frame-streaming output** — DONE. `LanceVideoPipeline.GenerateFramesAsync` → `IAsyncEnumerable<VideoFrame>` (pull-based backpressure) + `IVideoEncoder` (`FfmpegProcessEncoder` process-spawn, `BmpSequenceEncoder` fallback). Pure-C# MP4 muxer deferred.

### 3e. Licensing

- [x] ~~License-acceptance plumbing (`Licensing/ModelLicense.cs` + `LicenseAcceptance.cs`)~~ — **DROPPED (owner decision, 2026-06-15).** The engine is MIT and ships no weights/model code; the user supplies weights into `/Models` like every other model, and weight-license compliance is the user's responsibility. No `Licensing/` framework is built — GameCraft / Cosmos / any restricted-weight model loads the same way as SD/Flux.

## 4. Implementation — LTX-Video

- [x] DiT + VAE decoder — delivered as `LtxVideoTransformer`/`LtxRope`/`LtxVideoBlock` (spatio-temporal self-attn via 3D RoPE — no separate `TemporalAttention`) + `LtxVideoVaeDecoder` (base 0.9). See § 1.
- [ ] Video-specific PTX kernels (native 3D conv, fused temporal attention) — optional perf follow-up; everything runs on existing backend ops today.
- [x] [`LtxVideoPipeline.cs`](../../src/HartsyInference.Video/Pipelines/LtxVideoPipeline.cs) — flow-match Euler (dynamic μ shift) + 2-way CFG + frame streaming.
- [x] Frame output / encoding / progress — reuses the shared `VideoFrame` / `IVideoEncoder` / `GenerationProgress` infra from Lance video.
- [x] [`LtxVideoCheckpointConverter.cs`](../../src/HartsyInference.ModelHandler/CheckpointConverters/LtxVideoCheckpointConverter.cs) (2026-06-10) — single-file (`model.diffusion_model.*` + `vae.*`, original naming → diffusers rename tables) and diffusers-folder layouts; FP8 scale folding; per-channel-stats → `latents_mean`/`latents_std`. Pure `RouteKey` unit-tested.
- [x] `LtxVideoGenerationTests` (2026-06-10) — env/`TestPaths`-gated real-checkpoint entry: T5-XXL from the SD3.5 bundle (Chroma pattern), VRAM probe ≥10 GB, 25-frame 704×480 → BMP sequence. First-run numeric validation pending (needs checkpoint + GPU).
- [ ] Timestep-conditioned 0.9.5 VAE decoder variant (`LtxVaeResnetBlock3d` already supports the conditioning path; decoder assembly + converter renames pending).

## 5. Implementation — Wan

- [x] [`WanVideoPipeline.cs`](../../src/HartsyInference.Video/Pipelines/WanVideoPipeline.cs) — Wan2.2 TI2V-5B T2V; VAE shared with Lance (`Wan22VaeDecoder`). See § 1.
- [x] **Wan LoRA support (2026-06-10)** — `LoraFormat.KohyaWan` (kohya/musubi-tuner underscored: `lora_unet_blocks_{i}_self_attn_q.*`) + `LoraFormat.DiffusersWan` (ComfyUI-style `diffusion_model.blocks.*`, PEFT or kohya suffixes, `.diff` entries skipped) via one [`WanLoraMapper`](../../src/HartsyInference.ModelHandler/Lora/Mappers/WanLoraMapper.cs) that reuses the checkpoint converter's verbatim rename table; diffusers-PEFT Wan (`transformer.blocks.*`) rides the existing architecture-agnostic passthrough unchanged. `WanVideoGenerationTests` auto-merges `WAN_LORA_PATH` (strength via `WAN_LORA_STRENGTH`) before `LoadWeights` — e.g. a lightx2v distill LoRA for few-step inference. Numeric merge gate in `LoraWanTests`. See `docs/Design/LORA_KEY_MAPPING.md` § F6/F7.
- [x] **I2V variant (2026-06-10)** — diffusers `expand_timesteps` path: `WanVideoTransformer.Forward(…, float[] timesteps)` per-latent-frame AdaLN modulation (blocks + final layer; uniform input == scalar path bit-exact, gated by test), pipeline `firstFrameLatent:` conditioning (imposed on the model input each step at frame-timestep 0, re-imposed post-loop). RGB-input I2V blocked on `Wan22VaeEncoder` (not built) — condition latent must be VAE-encoded + `Wan22VaeLatentNorm.Normalize`d offline for now.
- [x] [`WanVideoCheckpointConverter.cs`](../../src/HartsyInference.ModelHandler/CheckpointConverters/WanVideoCheckpointConverter.cs) (2026-06-10) — original/Comfy single-file naming → diffusers (rename table from `convert_wan_to_diffusers.py` verbatim, incl. norm2/norm3 swap + I2V-14B `img_emb`/`k_img` keys for a future variant), diffusers-folder pass-through, FP8 folding. Pure `MapKey` unit-tested.
- [x] `WanVideoGenerationTests` + `T5TextEncoderConfig.Umt5Xxl` (2026-06-10) — env/`TestPaths`-gated entry: umT5-XXL encode (per-layer relative bias, 256k SentencePiece) → free umT5 → DiT denoise → streamed BMP frames; VRAM probe ≥14 GB. First-run numeric validation pending.
- [x] [`Wan22VaeEncoder`](../../src/HartsyInference.Diffusion/Models/Vae/Wan22VaeEncoder.cs) (2026-06-10) — Wan2.2 3D causal VAE encoder, single-frame/first-chunk path (patchify 2 → conv1 → 4 down-stages with `AvgDown3D` shortcuts + `Wan22Resample` downsample modes → middle → head→96ch → quant conv → μ → normalize). Loads from the same `wan22_vae.safetensors` (keys `encoder.*` + `conv1.*`). **RGB-input I2V is now end-to-end**: `WanVideoPipeline.EncodeFirstFrame(rgb24, w, h)` (+ `MatrixGame3Pipeline.EncodeSeedImage`). Remaining: multi-frame video encode (4-frame streaming chunks with the temporal stride-2 `time_conv`) for video-extend; numerics validation-pending.

## 6. Implementation — Lance video (`Lance_3B_Video`) — IMPLEMENTED (structurally verified end-to-end; numeric validation pending)

> Apache 2.0. Reuses every Phase 4 § Lance class verbatim and adds a video-specific pipeline.

> **Status (2026-06-08):** Lance T2V is **built and runs end-to-end on CPU** (tiny-config test → 5 frames). Delivered across 4 phases: (V1) Wan2.2 VAE **streaming decode** — the `feat_cache` temporal state machine (`Wan22StreamCache` + temporal `time_conv` in `Wan22Resample` + per-frame driver in `Wan22VaeDecoder`); (V2) `LanceVideoPipeline`; (V3) frame streaming (`GenerateFramesAsync` → `IAsyncEnumerable<VideoFrame>`) + encoders (`FfmpegProcessEncoder`, `BmpSequenceEncoder`); (V4) tests + env-gated generation entry. **52 Lance/Wan tests pass.** Strong gates verified: video frame-0 byte-identical to the image decode, and **streamed decode byte-identical to full-clip decode**. Numerics vs the real checkpoint remain validation-pending (no weights/GPU here). Reuses the entire image stack + all 3D primitives; the streaming VAE + frame-streaming infra is reusable by LTX/Wan video.

**Prerequisites — MOSTLY MET (2026-06-08):** Phase 4 § Lance image pipeline is built — `LanceTransformer`, `LanceMoTBlock`, `Multimodal3DRope`, `Wan22VaeDecoder` (image/T=1 path), `LanceLatentPatch`, `LanceCheckpointConverter`, `LanceImagePipeline`, plus the shared infra in § 3 (`CausalConv3d`, `Sincos3DPositionEmbedding`) all exist + are unit-tested. **Remaining prereqs for video:** (a) the `Wan22VaeDecoder` T>1 streaming path (feat_cache chunk loop + Resample temporal `time_conv`), and (b) `Qwen25VlVit` only if video *editing*/understanding is wanted (pure T2V doesn't need it). Numeric validation of the image path against the real checkpoint should land first so the shared transformer/VAE are trusted before adding the time axis.

#### Architecture deltas vs the image pipeline (see [`docs/Research/LANCE_ARCHITECTURE.md`](../Research/LANCE_ARCHITECTURE.md))

- `timestep_shift = 4.0` (not 3.5 — image inference).
- Latent T = `ceil(num_frames / 4) + 1` (4× temporal VAE downsample), up to **121 frames** at 480p (`480 × 848`). Image pipeline pinned T=1.
- `<|video_pad|>` (token 151656) replaces `<|image_pad|>` in the chat template's vision slot.
- VAE decode path uses the full `CausalConv3d` streaming wrapper (image path can collapse to Conv2D at T=1; video cannot).
- `PositionEmbedding3D` actually varies over t; image was the trivial t=0 case.
- 3-way CFG semantics are identical (just bigger noisy slot per step).

#### Files

- [x] **Wan2.2 VAE streaming decode (V1)** — [`Wan22StreamCache.cs`](../../src/HartsyInference.Diffusion/Models/Vae/Wan22StreamCache.cs) (`feat_cache`/`feat_idx` machine: `StepConv` + `StepTimeConv` "Rep" logic), temporal `time_conv` branch in [`Wan22Resample`](../../src/HartsyInference.Diffusion/Models/Vae/Wan22Resample.cs), per-frame driver + `DecodeStreaming` iterator in [`Wan22VaeDecoder`](../../src/HartsyInference.Diffusion/Models/Vae/Wan22VaeDecoder.cs), frame-slice/concat helpers in [`Vae3dLayout`](../../src/HartsyInference.Diffusion/Models/Vae/Vae3dLayout.cs). Decodes `T_lat → (T_lat−1)·4 + 1` frames. Gates: video frame-0 == image decode, **streamed == full-clip** (byte-identical).
- [x] [`src/HartsyInference.Video/Pipelines/LanceVideoPipeline.cs`](../../src/HartsyInference.Video/Pipelines/LanceVideoPipeline.cs) (V2/V3) — packs `[text(und) | noisy-VAE(gen)]` spanning `T_lat = (num_frames−1)/4 + 1` latent frames, MaPE positions over t, 2-way text CFG (3-way vision CFG = editing, needs ViT, deferred), logit-normal Euler (shift 4.0), streamed VAE decode → `GenerateFromTokens` (all frames) + `GenerateFramesAsync` (`IAsyncEnumerable<VideoFrame>`, pull-based backpressure). Shared sampling via [`LancePipelineCommon`](../../src/HartsyInference.Diffusion/Pipelines/LancePipelineCommon.cs).
- [x] [`src/HartsyInference.Video/VideoFrame.cs`](../../src/HartsyInference.Video/VideoFrame.cs) + [`Encoding/IVideoEncoder.cs`](../../src/HartsyInference.Video/Encoding/IVideoEncoder.cs) + [`Encoding/FfmpegProcessEncoder.cs`](../../src/HartsyInference.Video/Encoding/FfmpegProcessEncoder.cs) (raw RGB24 → H.264 MP4 via process pipe) + [`Encoding/BmpSequenceEncoder.cs`](../../src/HartsyInference.Video/Encoding/BmpSequenceEncoder.cs) (no-dependency frame-sequence fallback). Pull-based encoders → decode paced by encoding (backpressure); pure-C# `Mp4Muxer` deferred.
- [x] [`tests/HartsyInference.Video.Tests/`](../../tests/HartsyInference.Video.Tests/) — `LanceVideoPipelineTests` (tiny-config e2e, frame-count reject, streaming, BMP encoder) + `LanceVideoGenerationTests` (env-gated real-checkpoint entry, VRAM probe ≥24 GB, streams → BMP sequence). Streaming-correctness gate lives in `Wan22VaeDecoderTests.Decode_StreamingMatchesFullClip`.
- [ ] `tests/python-reference/dump_lance_video_full_forward.py` + `diff_lance_video_layers.py` — layer-by-layer F32 diff harness for video shapes. Deferred to first-run numeric validation (needs the checkpoint).

#### Weights (24 GB target)

- `Lance_3B_Video/model.safetensors` BF16 (28.4 GB) — won't fit at FP16 on 24 GB without eviction; FP8 cast at load (~14.2 GB) plus ViT (~0.7 GB) plus VAE (~1.4 GB) ≈ 16 GB, leaves ~8 GB for activations — works for ~50-frame 480p clips. 121-frame clips need 40+ GB VRAM or per-block streaming.
- `Qwen2.5-VL-ViT/vit.safetensors` BF16 (1.34 GB) — comfortable FP16.
- `Wan2.2_VAE.pth` FP32 (2.82 GB) — cast to FP16 at load.

#### Notes

- The `Lance_3B` (image-only) and `Lance_3B_Video` (unified) checkpoints can both load into a single `LanceTransformer` — the converter just chooses which `model.safetensors` to consume. The video pipeline is a thin wrapper on top of the same forward pass with T>1.
- Editing variants (image edit, video edit) are deferred until after both base pipelines are clean — they reuse the same transformer with `clean_vae_cond` slots populated from the source media.
- `cfg_vision_scale` default for video-edit is an open question in the research doc (§ Open Questions #4); confirm against `config/examples/video_edit.json` before shipping edit support.

## 7. Server Integration

- [ ] Video generation endpoint, SSE streaming, video file serving
- [ ] Lance-specific endpoint surface (or generic unified endpoint covering LTX / Wan / Lance with a model-name discriminator)

## 8. Testing

- [ ] Temporal attention consistency, video VAE vs reference
- [ ] Pipeline quality (manual check), VRAM usage, server endpoint
- [ ] All tests pass on GPU CI
- [ ] Lance video diff harness — layer-by-layer F32 noise floor vs reference Python forward

## 9. Review & Merge

- [ ] Code review (temporal correctness, frame buffer memory)
- [ ] Benchmark frames/sec
- [ ] Merge to main branch

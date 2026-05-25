# Phase 9 — Video (LTX-Video + Wan + Lance)

> **Goal:** Video generation starting with LTX-Video, with Lance (ByteDance) layered on top once Phase 4's MoT + packed-attention infra is in place.
> **Packages:** SharpInference.Video (+ shared 3D VAE / packed attention in SharpInference.Diffusion / Core / per-backend kernels)

---

## 1. Research

- [ ] LTX-Video architecture (temporal attention, 3D VAE, conditioning)
- [ ] Wan 2.1 architecture
- [ ] Temporal attention, video VAE decoder (3D convolutions), video output encoding
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
- [ ] **`IBackend.Conv3D(input, weight, bias, stride, padding, dilation)`** — 3D im2col + GEMM. Required by Wan2.2 VAE (Lance, Matrix-Game 2/3), Cosmos DV tokenizer, and any future 3D VAE in LTX / Wan-Video / Hunyuan-Video. CPU SIMD path, CUDA PTX kernel, Vulkan SPIR-V shader.
- [ ] **`CausalConv3d` streaming wrapper** — maintains a per-conv 2-frame cache (`CACHE_T=2` for Wan2.2) so video VAE decode can stream chunks of frames without OOM. Required for any video clip longer than a single GPU-resident chunk.

### 3b. Conditioning (shared across video + interactive)

- [ ] **`SharpInference.Conditioning/IActionEncoder.cs`** — generic action-conditioning abstraction. `ActionInput` (raw `ReadOnlyMemory<byte>` payload + frame index + timestamp) → preallocated `Span<float>` tokens. Lives in `Diffusion` for cross-domain reuse. Per-model encoders (keyboard+mouse, camera pose, Minecraft VPT 25-dim, gamepad) land in their respective Phase 10 model packages.
- [ ] **`SharpInference.Diffusion/Models/Denoisers/DiTBlocks/PositionEmbedding3D.cs`** — frozen 3D sin-cos position embedding. CPU precompute, upload once. No new kernel.

### 3c. Caches + schedulers

- [ ] **`SharpInference.Diffusion/Utilities/DenoiseKvCache.cs`** — first inference-time KV-cache use case. Caches K/V for the (text + action + clean-cond) prefix on step 0; recomputes only the noisy slot on steps 1..N. Reusable across Lance / LTX / Wan video / Matrix-Game / GameCraft (~2-3× wall speedup). AR-token variant (Cosmos AR / future MineWorld) deferred.
- [ ] **`SharpInference.Diffusion/Schedulers/DistilledFlowMatchEuler.cs`** — extends flow-match Euler with **DMD** (Distribution Matching Distillation, 3-4 step — Matrix-Game 2/3), **CM** (Consistency Models, 1-4 step), **Lightning** (1-4 step). Wired into `SchedulerFactory`. Required for any model running below the offline 30-step regime.
- [ ] **`SharpInference.Diffusion/Schedulers/DdimVPredScheduler.cs`** — DDIM with v-parameterization + sigmoid β-schedule (T=1000, start=-3, end=3). Specifically required by Oasis-500m; documented as a separate scheduler family from flow-match because the noise schedule + parameterization differ. Plus a **Diffusion Forcing** helper that holds context frames at a fixed noise level while the target frame denoises to zero.

### 3d. Streaming + tokenizers

- [ ] **`SharpInference.Video/Tokenizers/IDiscreteVideoTokenizer.cs`** — interface for discrete video codecs (encode RGB → int32 indices; decode indices → RGB; embedding lookup). First implementation `CosmosDvTokenizer` lands with Cosmos-Predict V2W. **VQ-GAN / MagViT-v2 / FSQ variants deferred** until a model that needs them lands (Oasis uses a continuous Gaussian VAE, not discrete).
- [ ] **`SharpInference.Video/Streaming/VideoVaeStreamDecoder.cs`** — per-frame / per-chunk VAE decode helper running on a secondary CUDA stream, overlapping decode with next-frame denoise. Critical for hitting interactive 25-40 FPS in Phase 10; also useful for offline frame-streaming in Phase 9.
- [ ] **Frame-streaming output** — `IAsyncEnumerable<VideoFrame>` shape on video pipelines, with optional `IVideoEncoder` (FFmpeg via process-spawn or a pure C# MP4 muxer) for users that want a single file at the end. Bound by `MAX_FRAMES_IN_FLIGHT` to keep VRAM flat.

### 3e. Licensing

- [ ] **`SharpInference.ModelHandler/Licensing/ModelLicense.cs` + `LicenseAcceptance.cs`** — restricted-license plumbing. `ApacheLicense2` / `MitLicense` / `NvidiaOpenModelLicense` / `TencentHunyuanCommunityLicense` typed records. Checkpoint converters for restricted models throw `LicenseNotAcceptedException` until `LicenseAcceptance.Accept(...)` has been called with the required token (captured to a user-local file so it's a one-time acceptance). Required for Hunyuan-GameCraft (Phase 10) and Cosmos-Predict V2W (Phase 9). Permissively-licensed models skip this entirely.

## 4. Implementation — LTX-Video

- [ ] `TemporalAttention.cs`, `VideoVaeDecoder.cs`
- [ ] Video-specific PTX kernels (3D conv, temporal attention)
- [ ] `LtxVideoPipeline.cs`
- [ ] Frame output (PNGs or video), video encoding, progress streaming

## 5. Implementation — Wan

- [ ] `WanPipeline.cs` — Wan 2.1 / 2.2 video. Wan2.2's 3D causal VAE is the same family as the one Lance uses (`Wan22VaeDecoder`), so the VAE module is shared.

## 6. Implementation — Lance video (`Lance_3B_Video`) — NOT STARTED

> Apache 2.0. Reuses every Phase 4 § Lance class verbatim and adds a video-specific pipeline.

**Prerequisites:** Phase 4 § Lance image pipeline complete (lands `LanceTransformer`, `LanceMoTBlock`, `LanceMRopeMaPE`, `Qwen25VlVit`, `Wan22VaeDecoder`, `LanceCheckpointConverter`, plus the shared infra in § 3 above).

#### Architecture deltas vs the image pipeline (see [`docs/Research/LANCE_ARCHITECTURE.md`](../Research/LANCE_ARCHITECTURE.md))

- `timestep_shift = 4.0` (not 3.5 — image inference).
- Latent T = `ceil(num_frames / 4) + 1` (4× temporal VAE downsample), up to **121 frames** at 480p (`480 × 848`). Image pipeline pinned T=1.
- `<|video_pad|>` (token 151656) replaces `<|image_pad|>` in the chat template's vision slot.
- VAE decode path uses the full `CausalConv3d` streaming wrapper (image path can collapse to Conv2D at T=1; video cannot).
- `PositionEmbedding3D` actually varies over t; image was the trivial t=0 case.
- 3-way CFG semantics are identical (just bigger noisy slot per step).

#### Files (planned, none built)

- [ ] `src/SharpInference.Video/Pipelines/LanceVideoPipeline.cs` — Qwen2 chat-template encode with `<|video_pad|>` slot → MoT-aware packed sequence with text + (optional ViT + clean-VAE-cond for edits) + noisy-target spanning `T_lat = ceil(num_frames/4) + 1` latent frames → 3-way CFG flow-match Euler (30 steps, shift 4.0) → `FreeWeights(transformer + vit)` before VAE → streamed Wan2.2 VAE decode (`CausalConv3d` 2-frame cache) → `IAsyncEnumerable<VideoFrame>`.
- [ ] `src/SharpInference.Video/Streaming/VideoFrameStreamer.cs` — bounded-buffer producer for `IAsyncEnumerable<VideoFrame>` with backpressure.
- [ ] `src/SharpInference.Video/Encoding/IVideoEncoder.cs` + at least one impl (`FfmpegProcessEncoder.cs` first, optional pure-C# `Mp4Muxer.cs` follow-up).
- [ ] `tests/SharpInference.Video.Tests/LanceVideoGenerationTests.cs` — env-var-gated, VRAM probe ≥ 24 GB free for FP8 short-clip / ≥ 40 GB for FP16. Skips cleanly when `LANCE_3B_VIDEO_PATH`, `LANCE_VIT_PATH`, `LANCE_VAE_PATH`, or PTX dir is missing. Skips additionally when the requested frame count would exceed memory for the detected VRAM budget.
- [ ] `tests/SharpInference.Video.Tests/Wan22VaeStreamingTests.cs` — verifies `CausalConv3d` 2-frame cache produces byte-identical output to a non-streamed full-clip decode for small clips (regression gate for any future temporal-cache refactor).
- [ ] `tests/python-reference/dump_lance_video_full_forward.py` + `diff_lance_video_layers.py` — layer-by-layer F32 diff harness for video shapes (`(B, 48, T_lat, H, W)` latents).

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

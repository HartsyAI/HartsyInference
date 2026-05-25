# Phase 10 — Interactive / World Models

> **Goal:** Action-conditioned, real-time, frame-by-frame video generation. Drive a model from a live input stream (keyboard, mouse, gamepad, camera pose) and emit a streamed video output at 25–40 FPS. New `SharpInference.Interactive` NuGet package.
> **Packages:** SharpInference.Interactive (new) + SharpInference.Video (extended) + SharpInference.Diffusion (extended)
>
> **Prereqs:** Phase 9 shared infrastructure must land first — see [`PHASE_9_VIDEO.md` § 3 Shared Infrastructure](PHASE_9_VIDEO.md#3-shared-infrastructure-cross-cutting-prereqs-for-all-video--interactive-models). Specifically: `IBackend.Conv3D` + `CausalConv3d` streaming wrapper, `IBackend.PackedAttention`, `IActionEncoder` abstraction, `DenoiseKvCache` (DiffusionPrefix + SlidingWindowVideoFrames modes), `DistilledFlowMatchEuler` scheduler, `VideoVaeStreamDecoder`, license-acceptance plumbing.
>
> **Foundational design doc:** [`docs/Research/INTERACTIVE_INFERENCE.md`](../Research/INTERACTIVE_INFERENCE.md) — read this before starting any work in this phase. It defines `IInteractiveSession`, the multi-stream `IActionEncoder`, the three KV-cache modes, the discrete tokenizer abstraction, the memory-augmented sequence pattern, distilled schedulers, and license plumbing.

---

## 1. Research

- [x] [INTERACTIVE_INFERENCE.md](../Research/INTERACTIVE_INFERENCE.md) — cross-cutting infrastructure design (action encoders, streaming sessions, KV-cache modes, discrete tokenizers, distilled schedulers, license plumbing, deferred backlog)
- [x] [MATRIX_GAME_3_ARCHITECTURE.md](../Research/MATRIX_GAME_3_ARCHITECTURE.md) — flagship 5B world model, Wan2.2-TI2V-5B finetune, ActionModule + camera-aware memory, FlowUniPC + DMD distilled
- [x] [MATRIX_GAME_2_ARCHITECTURE.md](../Research/MATRIX_GAME_2_ARCHITECTURE.md) — 1.8B entry-level, SkyReels-V2 (Wan2.1) lineage, per-variant action vocabs, 3-4 step distilled
- [x] [OASIS_ARCHITECTURE.md](../Research/OASIS_ARCHITECTURE.md) — tiny 500M DiT-S/2 with spatio-temporal axial attention, continuous Gaussian VAE (NOT VQ), DDIM v-pred + Diffusion Forcing
- [x] [HUNYUAN_GAMECRAFT_ARCHITECTURE.md](../Research/HUNYUAN_GAMECRAFT_ARCHITECTURE.md) — 12.5B HunyuanVideo MM-DiT + CameraNet + Plücker, 33-channel composite history input, PCM+CFG distilled. **License-restricted.**

## 2. Planning

- [ ] Package boundary review — confirm `SharpInference.Interactive` depends on Video transitively (which brings Diffusion + ModelHandler). Verify no Server / Web dependencies leak into the streaming session contract.
- [ ] `IInteractiveSession` API freeze — review the surface in [INTERACTIVE_INFERENCE.md § 3](../Research/INTERACTIVE_INFERENCE.md) with one application integration target in mind before locking the contract.
- [ ] License-acceptance UX — settle the Server-side endpoint shape (`POST /v1/licenses/accept`) and the local-cache filename convention (e.g. `~/.sharpinference/licenses/<license-id>.accepted`).

## 3. Implementation — `SharpInference.Interactive` core (cross-model)

- [ ] `src/SharpInference.Interactive/Sessions/IInteractiveSession.cs` — interface (`SubmitActionAsync` / `ReadFramesAsync` / `GetStats` / `SetQualityProfile`).
- [ ] `src/SharpInference.Interactive/Sessions/BackgroundComputeSession.cs` — default implementation. Owns one dedicated compute thread + CUDA stream + bounded `Channel<ActionInput>` (action queue) + bounded `Channel<VideoFrame>` (output queue, drop-oldest policy).
- [ ] `src/SharpInference.Interactive/Sessions/InteractiveSessionStats.cs` — readonly record struct: p50/p99 step latency, dropped frames, queue depths.
- [ ] `src/SharpInference.Interactive/ActionEncoders/IActionEncoder.cs` — multi-stream encoder interface per the corrected design in [INTERACTIVE_INFERENCE.md § 2](../Research/INTERACTIVE_INFERENCE.md).
- [ ] `src/SharpInference.Interactive/ActionEncoders/KeyboardOneHotEncoder.cs` — generic discrete-key one-hot helper used by every model.
- [ ] `src/SharpInference.Interactive/ActionEncoders/MouseDeltaEncoder.cs` — generic mouse Δx/Δy normalized encoder.
- [ ] `src/SharpInference.Interactive/Camera/SE3Math.cs` — SE(3) inverse, SLERP, integrate-actions-to-poses. **Shared between Matrix-Game 3.0 and Hunyuan-GameCraft** (both need camera pose integration).
- [ ] `src/SharpInference.Interactive/Camera/PluckerEmbedding.cs` — Plücker ray-coordinate computation (6-channel for GameCraft, 16-channel for Matrix-Game 3.0; exact channel layout per per-model doc).
- [ ] `src/SharpInference.Interactive/Memory/FrameHistoryBuffer.cs` — rolling buffer of `(latent, camera_pose, frame_index)` tuples. Bounded capacity, evicts oldest when full.
- [ ] `src/SharpInference.Interactive/Memory/FrustumOverlapSelector.cs` — port of Matrix-Game 3.0's `select_memory_idx_fov`. GPU kernel for production; CPU fallback for tests.
- [ ] `tests/SharpInference.Interactive.Tests/InteractiveSessionTests.cs` — unit tests for queue back-pressure, drop-oldest policy, dispose-during-step safety, action-replay determinism (with a stub action encoder).

## 4. Implementation — Matrix-Game 2.0 pipeline (entry-level, MIT)

> Target: RTX 3060 12 GB at distilled-3-step. Source-of-truth: [MATRIX_GAME_2_ARCHITECTURE.md](../Research/MATRIX_GAME_2_ARCHITECTURE.md).

- [ ] `src/SharpInference.Interactive/Pipelines/MatrixGame2Pipeline.cs` — three variant entry points (`CreateUniversal`, `CreateGta`, `CreateTempleRun`); each constructs the appropriate `IActionEncoder` + DiT config.
- [ ] `src/SharpInference.Video/Models/Denoisers/Wan21Dit.cs` — 30-layer SkyReels-V2/Wan2.1 DiT (dim=1536, heads=12, ffn=8960, head_dim=128, patch=(1,2,2), in_dim=36, out_dim=16). **Reusable for SkyReels-V2 standalone** if added later.
- [ ] `src/SharpInference.Video/Models/Denoisers/DiTBlocks/Wan21AttentionBlock.cs` — self-attn + cross-attn (CLIP-cond) + ffn, RMSNorm + QK-norm. **Shared with Wan2.1 video pipeline** (if/when added).
- [ ] `src/SharpInference.Interactive/Models/Denoisers/DiTBlocks/MatrixGame2ActionModule.cs` — dual-stream (mouse=self-attn, keyboard=cross-attn) with `windows_size=3` past-latent-frames context, 16 heads × 64 dim, RoPE `[8,28,28]` θ=256. Attached to first 15 of 30 blocks (distilled) or all 30 (foundation).
- [ ] `src/SharpInference.Video/Models/Vae/Wan21VaeDecoder.cs` — Wan2.1 3D causal VAE, 16 latent channels, 8×8 spatial / 4× temporal. Tiling defaults `[44, 80]` latent cells = 352×640 pixel tiles. **First Wan2.1 user** (Wan2.2 lands earlier with Lance).
- [ ] `src/SharpInference.Diffusion/Models/TextEncoders/ClipVitH14ImageEncoder.cs` — CLIP-ViT-H/14 image tower (257 tokens × 1280 dim). Used for I2V cross-attention conditioning at session start.
- [ ] `src/SharpInference.Interactive/ActionEncoders/MatrixGameUniversalActionEncoder.cs` — 4-key keyboard (fwd/back/left/right) + 2-dim mouse (pitch/yaw, `CAM_VALUE=0.1`).
- [ ] `src/SharpInference.Interactive/ActionEncoders/MatrixGameGtaActionEncoder.cs` — 2-key + 2-dim mouse.
- [ ] `src/SharpInference.Interactive/ActionEncoders/MatrixGameTempleRunActionEncoder.cs` — 7-key, no mouse.
- [ ] `src/SharpInference.ModelHandler/CheckpointConverters/MatrixGame2CheckpointConverter.cs` — splits checkpoint into `dit.*` / `action_module.*` / `vae.*` / `clip.*` buckets. Handles per-variant `config.json` discovery.
- [ ] `tests/SharpInference.Interactive.Tests/MatrixGame2GenerationTests.cs` — env-var-gated (`MATRIX_GAME_2_BASE_PATH`, `MATRIX_GAME_2_VAE_PATH`, `CLIP_VITH14_PATH`); VRAM probe ≥ 8 GB FP8 / ≥ 12 GB bf16; skips cleanly when missing.
- [ ] `tests/python-reference/dump_matrix_game_2_full_forward.py` + `diff_matrix_game_2_layers.py` + `MatrixGame2DiffTests.cs` — layer-by-layer F32 diff harness per project convention.

## 5. Implementation — Matrix-Game 3.0 pipeline (flagship, Apache-2.0)

> Target: RTX 4090 24 GB minimum. Source-of-truth: [MATRIX_GAME_3_ARCHITECTURE.md](../Research/MATRIX_GAME_3_ARCHITECTURE.md). User explicitly requested this; primary deliverable for the phase.

- [ ] `src/SharpInference.Interactive/Pipelines/MatrixGame3StandardPipeline.cs` — canned-action one-shot. Matches `inference_pipeline.py`.
- [ ] `src/SharpInference.Interactive/Pipelines/MatrixGame3InteractivePipeline.cs` — live-action streaming. Matches `inference_interactive_pipeline.py`. Maintains rolling buffer + 5-slot memory cache + per-segment denoise loop + async VAE worker (multi-GPU only).
- [ ] `src/SharpInference.Video/Models/Denoisers/Wan22Dit.cs` — 40-layer Wan2.2 DiT (dim=5120, heads=40, ffn=13824, head_dim=128, patch=(1,2,2), in_dim/out_dim=48). **Reusable by any Wan2.2 video pipeline** (TI2V, T2V, I2V, V2V).
- [ ] `src/SharpInference.Video/Models/Denoisers/DiTBlocks/Wan22AttentionBlock.cs` — self-attn + cross-attn (UMT5 text-cond) + ffn, RMSNorm + QK-norm + cross-attn-norm, AdaLN modulation (6 params per block).
- [ ] `src/SharpInference.Interactive/Models/Denoisers/DiTBlocks/MatrixGame3ActionModule.cs` — dual-stream (mouse=self-attn temporal, keyboard=cross-attn), 16 heads, RoPE `[8,28,28]` θ=256, `windows_size=3`, `local_attn_size=6`. Attached to the subset of 40 Wan2.2 blocks specified by the checkpoint.
- [ ] `src/SharpInference.Video/Models/Vae/MgLightVaeDecoder.cs` — pruned Wan2.2 VAE decoder. Pruning rate auto-detected from `decoder.conv1.weight.shape[0]` (50 % or 75 %). Encoder remains full Wan2.2.
- [ ] `src/SharpInference.Video/Schedulers/FlowUniPCMultistepScheduler.cs` — **first UniPC variant in SharpInference**. Port of `diffusers.FlowUniPCMultistepScheduler`. Used at both 50-step (base) and 3-step (distilled).
- [ ] `src/SharpInference.Interactive/ActionEncoders/MatrixGame3ActionEncoder.cs` — 6-dim keyboard one-hot + 2-dim mouse (Δx, Δy normalized).
- [ ] `src/SharpInference.Interactive/Camera/MatrixGame3PluckerEmbedding.cs` — 16-channel Plücker variant (exact layout per [Open Question 4](../Research/MATRIX_GAME_3_ARCHITECTURE.md#open-questions); read after checkpoint download).
- [ ] `src/SharpInference.Interactive/Memory/MatrixGame3MemoryRetrieval.cs` — `SelectByFovOverlap` with sphere-point sampling + GPU frustum overlap kernel.
- [ ] `src/SharpInference.ModelHandler/CheckpointConverters/MatrixGame3CheckpointConverter.cs` — splits `base_model/diffusion_pytorch_model.safetensors` into `dit.*` / `action_module.*` / `plucker_proj.*`. Handles `base_distilled_model/` (extract student-only slice, ignore critic/EMA).
- [ ] `src/SharpInference.ModelHandler/CheckpointConverters/MgLightVaeConverter.cs` — shape-tolerant loader detecting pruning rate at load time.
- [ ] `tests/SharpInference.Interactive.Tests/MatrixGame3GenerationTests.cs` — env-var-gated; VRAM probe ≥ 20 GB bf16 / ≥ 12 GB FP8; segment-0 latent reference diff at `1e-2` (bf16 accumulation tolerance).
- [ ] `tests/SharpInference.Interactive.Tests/MatrixGame3InteractiveSessionTests.cs` — replay deterministic action sequence; verify rolling buffer + memory selection across segments; multi-segment continuity gate.
- [ ] `tests/python-reference/dump_matrix_game_3_full_forward.py` + `diff_matrix_game_3_layers.py` — layer-by-layer F32 diff harness.

## 6. Implementation — Oasis-500m pipeline (pedagogical, MIT)

> Target: any GPU (smallest model in the phase). Source-of-truth: [OASIS_ARCHITECTURE.md](../Research/OASIS_ARCHITECTURE.md). Acts as the CI smoke test for the entire phase's action-conditioning correctness.

- [ ] `src/SharpInference.Interactive/Pipelines/OasisPipeline.cs` — autoregressive frame-by-frame loop. **Different from Matrix-Game: action conditioning is added directly to the per-frame timestep embedding** (not a separate ActionModule stream), and there's no memory bank — just a context window of past frames at fixed noise level (Diffusion Forcing).
- [ ] `src/SharpInference.Interactive/Models/Denoisers/OasisDit.cs` — DiT-S/2 (16 layers, hidden 1024, 16 heads / head_dim 64, patch_size 2). **First DiT-S/2 in the codebase.**
- [ ] `src/SharpInference.Interactive/Models/Denoisers/DiTBlocks/OasisSpatioTemporalBlock.cs` — alternating bidirectional spatial axial attention + causal temporal axial attention (Latte-style). 2D RoPE for spatial, 1D RoPE for temporal.
- [ ] `src/SharpInference.Interactive/Models/Vae/OasisVitVae.cs` — **continuous Gaussian VAE** (NOT VQ). Patch 20, 360×640 → 18×32×16 latent. Encoder 6 layers, decoder 12 layers (asymmetric — "shallow encoder"). `scaling_factor = 0.07843137255`.
- [ ] `src/SharpInference.Diffusion/Models/TextEncoders/ClipVitL20VisionEncoder.cs` — patch-20 ViT-L variant Oasis ships in `vit-l-20.safetensors` (917 MB). Used for per-frame visual embedding of context frames.
- [ ] `src/SharpInference.Interactive/ActionEncoders/MinecraftVptActionEncoder.cs` — **25-dim VPT action vector** (23 binary keys: WASD + jump + sneak + sprint + attack + use + drop + etc., + 2 normalised camera floats). `nn.Linear(25, 1024)` projection added to timestep embedding.
- [ ] `src/SharpInference.Video/Schedulers/DdimVPredScheduler.cs` — DDIM with v-parameterization + sigmoid β-schedule (T=1000, start=-3, end=3). Plus a `DiffusionForcing` helper that holds context frames at fixed noise level 14, target at full noise.
- [ ] `src/SharpInference.ModelHandler/CheckpointConverters/OasisCheckpointConverter.cs` — loads `oasis500m.safetensors` (2.43 GB) + `vit-l-20.safetensors` (917 MB).
- [ ] `tests/SharpInference.Interactive.Tests/OasisGenerationTests.cs` — env-var-gated; any GPU; the **canonical CI smoke test** for action conditioning since the model is small.

## 7. Implementation — Hunyuan-GameCraft 1.0 (RESTRICTED LICENSE)

> ⚠️ **License caveat:** Tencent Hunyuan Community License forbids use in EU/UK/SK and caps users at 100M MAU. SharpInference will **not bundle weights**, **not auto-download**, and **will require explicit license acceptance at load time**. See [HUNYUAN_GAMECRAFT_ARCHITECTURE.md § License Warning](../Research/HUNYUAN_GAMECRAFT_ARCHITECTURE.md).

> Target: A100/H100 40+ GB (12.5B params, 90 GB on disk). Implement last after Matrix-Game pipelines are stable.

- [ ] `src/SharpInference.Interactive/Pipelines/HunyuanGameCraftPipeline.cs` — full + distilled (50-step base + 8-step PCM-distilled). License-acceptance gate enforced at construction.
- [ ] `src/SharpInference.Video/Models/Denoisers/HunyuanVideoDit.cs` — HunyuanVideo MM-DiT minus one block per stream (19 double + 38 single, dim=3072, heads=24, head_dim=128, ffn_ratio=4.0, RoPE `[16,56,56]`, RMSNorm Q/K, GeGLU-tanh). Reusable for offline HunyuanVideo if/when added.
- [ ] `src/SharpInference.Interactive/Models/Denoisers/DiTBlocks/GameCraftCameraNet.cs` — `PixelUnshuffle(8) → Conv 384→192 → GN → ReLU → temporal pool → Conv 192→96 → GN → ReLU → Conv 96→16 → PatchEmbed → learnable scale → add to image tokens`. Token-addition fusion of action-derived camera information.
- [ ] `src/SharpInference.Video/Models/Vae/HunyuanVideoVaeDecoder.cs` — HunyuanVideo `884-16c-hy0801` 3D causal VAE: 16 latent channels, 8× spatial, 4× temporal. **Reusable with the offline HunyuanVideo pipeline.**
- [ ] `src/SharpInference.Diffusion/Models/TextEncoders/LlavaLlama3_8BEncoder.cs` — Llava-Llama-3-8B (4096-dim, `crop_start=95`, max_len 256). Plus existing CLIP-ViT-L (768 pooled).
- [ ] `src/SharpInference.Interactive/ActionEncoders/GameCraftActionEncoder.cs` — `(w/a/s/d, speed ∈ [0, 3])` → continuous `(d_trans, d_rot, α, β)` → 33 camera poses → 6-channel Plücker ray maps at full resolution. **Per-frame, not per-segment** — fundamentally different cadence from Matrix-Game.
- [ ] `src/SharpInference.Interactive/Pipelines/GameCraftLatentBuilder.cs` — assembles the 33-channel composite input `[noisy(16) + ref_history(16) + mask(1)]`. Model-specific input shape, not shared with other models.
- [ ] `src/SharpInference.Video/Schedulers/FlowMatchDiscreteScheduler.cs` — full + distilled, `shift=5.0` (SD3 time-shift), CFG=2.0 (base) / 1.0 (distilled).
- [ ] `src/SharpInference.ModelHandler/CheckpointConverters/HunyuanGameCraftCheckpointConverter.cs` — handles `mp_rank_00_model_states.pt` (PyTorch pickle, **not safetensors** — needs a one-off Python conversion script ships separately).
- [ ] `src/SharpInference.ModelHandler/Licensing/TencentHunyuanCommunityLicense.cs` — typed license record. Converter throws `LicenseNotAcceptedException` until `LicenseAcceptance.Accept(...)` has been called.
- [ ] `tests/SharpInference.Interactive.Tests/HunyuanGameCraftLicenseGateTests.cs` — verifies license enforcement is unbypassable (load without acceptance throws; load with acceptance proceeds).
- [ ] `tests/SharpInference.Interactive.Tests/HunyuanGameCraftGenerationTests.cs` — env-var-gated **and** license-acceptance-gated; VRAM probe ≥ 40 GB.

## 8. Server integration

- [ ] `src/SharpInference.Server/Endpoints/InteractiveSessionEndpoint.cs` — WebSocket endpoint that wraps `IInteractiveSession` (action input via inbound messages, frames via outbound messages). Per-connection session lifecycle.
- [ ] `src/SharpInference.Server/Endpoints/LicenseAcceptanceEndpoint.cs` — `POST /v1/licenses/accept` for restricted models. Required before any restricted-model load endpoint will succeed.
- [ ] `src/SharpInference.Server/Streaming/InteractiveFrameStream.cs` — frame-serialization adapter (PNG, JPEG, or raw RGB depending on Accept-Encoding).
- [ ] Server test: `tests/SharpInference.Server.Tests/InteractiveSessionWsTests.cs` — end-to-end smoke test with the Oasis-500m fixture model.

## 9. Testing & Validation

- [ ] All `*GenerationTests` skip cleanly when env vars (`MATRIX_GAME_2_BASE_PATH`, `MATRIX_GAME_3_BASE_PATH`, `OASIS_500M_PATH`, `HUNYUAN_GAMECRAFT_PATH`) are missing.
- [ ] All `*GenerationTests` perform a VRAM probe before allocating and skip when below the documented minimum.
- [ ] Reference validation: each model has a `dump_*_full_forward.py` Python reference + `diff_*_layers.py` per-layer diff harness, mirroring SD3.5 / Z-Image / Lance conventions.
- [ ] Performance gates per per-model doc (Matrix-Game 2.0: 25 FPS @ 540p on RTX 4090; Matrix-Game 3.0: ≥10 FPS @ 720p distilled on RTX 4090).
- [ ] License-gate test: confirm Hunyuan-GameCraft refuses to load without prior acceptance, and that the accepted-token persists across process restarts.
- [ ] Stress: 30-minute continuous interactive session — no memory leak, no VRAM bloat, p99 step latency within 2× of p50.

## 10. Deferred-foundation backlog (documented, not built v1)

Explicitly tracked here so they're visible. Each item gets built when a model that actually needs it gets selected.

| Item | Trigger to build |
|---|---|
| `DenoiseKvCache.KvCacheMode.AutoregressiveTokens` + `AppendTokens` / `Trim` | First AR-token world model is implemented (Cosmos AR 13B, future MineWorld, future Solaris) |
| Long-context spacetime RoPE (single-token incremental update) | Same trigger as above |
| `VqGanTokenizer` / `MagViTv2Tokenizer` implementations | Same trigger (Oasis-500m uses continuous VAE; no VQ-family user in v1) |
| `IActionLogger` record/replay | First test that needs deterministic action playback for regression |
| Per-session CUDA stream pool | First time a single `CudaBackend` instance is shared by 2+ concurrent `IInteractiveSession`s |
| Async VAE worker (multi-GPU) | First multi-GPU host runs Matrix-Game 3.0 with the 40-FPS target |
| Ulysses sequence-parallel attention | Same trigger |
| INT8 W8A8 GEMM (attention QKV/O only) — Triton-equivalent PTX kernel | First user complains about Matrix-Game 3.0 single-GPU throughput; accept ~30 % slowdown until then |
| 2×14B Matrix-Game MoE viewpoint-router pipeline | Skywork actually releases the 2×14B MoE weights |
| WebRTC / WebTransport streaming protocol | First user wants browser-side interactive playback (not just direct WebSocket) |
| Multi-tenant session queue manager | Multi-tenant interactive serving scenario |
| Server-side license-acceptance persistence across deployments | First production deployment of a license-restricted model |

## 11. Review & Merge

- [ ] Code review (streaming-loop correctness, action-encoder shape conformance, VRAM lifecycle across segments, license gate enforcement)
- [ ] Benchmark frames/sec per model on the reference hardware matrix; document in `docs/Research/BENCHMARKING.md`
- [ ] License-acceptance flow reviewed by a non-engineer to confirm UX is clear
- [ ] Documentation pass on `SharpInference.Interactive` README and sample apps (interactive Minecraft-like loop with Oasis, Matrix-Game 3.0 game-engine snippet)
- [ ] Merge to main branch — user action

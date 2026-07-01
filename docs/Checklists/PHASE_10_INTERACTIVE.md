# Phase 10 — Interactive / World Models

> **Goal:** Action-conditioned, real-time, frame-by-frame video generation. Drive a model from a live input stream (keyboard, mouse, gamepad, camera pose) and emit a streamed video output at 25–40 FPS. New `HartsyInference.Interactive` NuGet package.
> **Packages:** HartsyInference.Interactive (new) + HartsyInference.Video (extended) + HartsyInference.Diffusion (extended)
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

- [ ] Package boundary review — confirm `HartsyInference.Interactive` depends on Video transitively (which brings Diffusion + ModelHandler). Verify no Server / Web dependencies leak into the streaming session contract.
- [ ] `IInteractiveSession` API freeze — review the surface in [INTERACTIVE_INFERENCE.md § 3](../Research/INTERACTIVE_INFERENCE.md) with one application integration target in mind before locking the contract.
- [ ] License-acceptance UX — settle the Server-side endpoint shape (`POST /v1/licenses/accept`) and the local-cache filename convention (e.g. `~/.hartsyinference/licenses/<license-id>.accepted`).

## 3. Implementation — `HartsyInference.Interactive` core (cross-model) — **DONE (2026-06-10)**

> Landed alongside the Matrix-Game 3.0 build (the "whichever model lands first" forcing function). 20 unit tests pass.

- [x] [`Sessions/IInteractiveSession.cs`](../../src/HartsyInference.Interactive/Sessions/IInteractiveSession.cs) — `SubmitActionAsync` / `ReadFramesAsync` / `GetStats` / `SetQualityProfile` (+ [`QualityProfile`](../../src/HartsyInference.Interactive/Sessions/QualityProfile.cs) presets, [`IFrameStepper`](../../src/HartsyInference.Interactive/Sessions/IFrameStepper.cs) — the model-specific compute body, segment-based models buffer actions internally).
- [x] [`Sessions/BackgroundComputeSession.cs`](../../src/HartsyInference.Interactive/Sessions/BackgroundComputeSession.cs) — dedicated compute thread, bounded action queue (block-on-full), bounded frame queue (drop-oldest, counted), repeat-last-action on underrun, pooled payload copies (no steady-state allocs on submit), p50/p99 latency ring. Per-session CUDA stream remains a deferred item (§ 10).
- [x] `IActionEncoder` — placed in **`HartsyInference.Diffusion/Conditioning/`** per [INTERACTIVE_INFERENCE.md § 2](../Research/INTERACTIVE_INFERENCE.md) (cross-domain reuse), not in Interactive: [`ActionInput`](../../src/HartsyInference.Diffusion/Conditioning/ActionInput.cs) + multi-stream [`IActionEncoder`/`ActionStreamSpec`/`ActionStreamRole`](../../src/HartsyInference.Diffusion/Conditioning/IActionEncoder.cs).
- [x] [`Sessions/InteractiveSessionStats.cs`](../../src/HartsyInference.Interactive/Sessions/InteractiveSessionStats.cs).
- [x] [`ActionEncoders/KeyboardOneHotEncoder.cs`](../../src/HartsyInference.Interactive/ActionEncoders/KeyboardOneHotEncoder.cs) + [`MouseDeltaEncoder.cs`](../../src/HartsyInference.Interactive/ActionEncoders/MouseDeltaEncoder.cs) (generic helpers) + [`MatrixGame3ActionEncoder.cs`](../../src/HartsyInference.Interactive/ActionEncoders/MatrixGame3ActionEncoder.cs) (14-byte payload → keyboard/mouse streams).
- [x] [`Camera/Se3Math.cs`](../../src/HartsyInference.Interactive/Camera/Se3Math.cs) — SE(3) inverse, Euler→rotation (Rz·Ry·Rx + the Matrix-Game R_init remap + 0.01 translation scale), quaternion SLERP, integrate-actions-to-poses (step semantics validation-gated).
- [x] [`Camera/PluckerEmbedding.cs`](../../src/HartsyInference.Interactive/Camera/PluckerEmbedding.cs) — 6-channel ray origin+direction maps (reading `cam_utils.py` resolved the "16-channel" open question: rays are 6-channel; Matrix-Game's 16 comes from per-token pixel grouping — see [`MatrixGame3PluckerTokens`](../../src/HartsyInference.Interactive/Camera/MatrixGame3PluckerTokens.cs)).
- [x] [`Memory/FrameHistoryBuffer.cs`](../../src/HartsyInference.Interactive/Memory/FrameHistoryBuffer.cs) — bounded rolling `(latent copy, pose, frame index)` buffer, disposes on eviction.
- [x] [`Memory/FrustumOverlapSelector.cs`](../../src/HartsyInference.Interactive/Memory/FrustumOverlapSelector.cs) — CPU port of `select_memory_idx_fov` (uniform-in-sphere sampling, shared-visibility ratio, top-k + most-recent fallback). GPU kernel = perf follow-up (reference ablation: the single biggest FPS knob).
- [x] [`tests/HartsyInference.Interactive.Tests/`](../../tests/HartsyInference.Interactive.Tests/) — session back-pressure/drop-oldest/repeat-last/dispose, SE(3) invariants, Plücker invariants, FOV selection, history buffer, action encoder (20 tests).

## 4. Implementation — Matrix-Game 2.0 pipeline (entry-level, MIT)

> Target: RTX 3060 12 GB at distilled-3-step. Source-of-truth: [MATRIX_GAME_2_ARCHITECTURE.md](../Research/MATRIX_GAME_2_ARCHITECTURE.md).
>
> **Status (2026-06-10): BUILT END-TO-END** — structurally CPU-verified, numerics validation-pending. The DiT reuses `WanVideoBlock`/`WanRope`/`WanDitOps` (the planned `Wan21Dit`/`Wan21AttentionBlock` were unnecessary — the Wan2.1 block is the same shape post-rename); the ActionModule is the shared `MatrixGame3ActionModule` (GameFactory module) with new enable-mouse/keyboard flags. The KV-cache formulation runs as an equivalent joint forward over [window context @ t=0 ‖ noisy block] with an additive block-causal + local-window mask — `RollingKvCache` remains the perf/exactness follow-up; the padding-frame image-condition latent encodes one black frame instead of the chunked causal video encode (validation-gated).

- [x] [`Pipelines/MatrixGame2Pipeline.cs`](../../src/HartsyInference.Interactive/Pipelines/MatrixGame2Pipeline.cs) — autoregressive blocks of 3 latent frames, [`FlowMatchDmdScheduler`](../../src/HartsyInference.Diffusion/Schedulers/FlowMatchDmdScheduler.cs) (`warp_denoising_step` shift-5 grid, ToX0 + fresh-noise re-noise; analytic exactness tests), channel-concat conditioning (16 noisy + 4 mask + 16 img-cond = in_dim 36), CLIP context via `EncodeClipContext` (bilinear 224² + CLIP norm → `ClipVisionEncoder.EncodeHiddenStates`, the existing ViT-H/14 preset — no new encoder class needed) or precomputed. Variants via `MatrixGame2Config.Universal/GtaDrive/TempleRun/Foundation`.
- [x] DiT — [`MatrixGame2Transformer`](../../src/HartsyInference.Diffusion/Models/Denoisers/MatrixGame2Transformer.cs) (Diffusion, per project convention): reused `WanVideoBlock` stack + new `BuildBlockCausalMask` (block-causal + `local_attn_size` window, mask-gate tested) + the I2V `img_emb` MLPProj (CLIP 1280 → dim) + per-frame timesteps (Self-Forcing context at t=0).
- [x] ActionModule — **reused** [`MatrixGame3ActionModule`](../../src/HartsyInference.Diffusion/Models/Denoisers/DiTBlocks/MatrixGame3ActionModule.cs) (same GameFactory class; per-variant `keyboard_dim_in` 4/2/7, `enableMouse` flag added for TempleRun).
- [x] [`Wan21VaeDecoder`](../../src/HartsyInference.Diffusion/Models/Vae/Wan21VaeDecoder.cs) + [`Wan21VaeEncoder`](../../src/HartsyInference.Diffusion/Models/Vae/Wan21VaeEncoder.cs) + [`Wan21VaeLatentNorm`](../../src/HartsyInference.Diffusion/Models/Vae/Wan21VaeLatentNorm.cs) (16ch, 8×/4×, published mean/std verbatim) — assembled from the shared Wan blocks (flat `upsamples.{idx}` keys, channel-halving up-convs now derived from weights in `Wan22Resample`). Streaming multi-frame decode works; multi-frame encode = shared Wan-family follow-up. Tiling deferred.
- [x] CLIP-ViT-H/14 — reused `ClipVisionEncoder` (`ClipVisionEncoderConfig.ViTH14`, penultimate hidden states = diffusers Wan-I2V convention). A converter for the bundled 4.77 GB `.pth` is deferred — use an HF-format ViT-H safetensors.
- [x] Action encoders — one [`MatrixGame2ActionEncoder`](../../src/HartsyInference.Interactive/ActionEncoders/MatrixGame2ActionEncoder.cs) with `CreateUniversal()/CreateGtaDrive()/CreateTempleRun()` factories (instead of three classes), keyboard index maps + `CAM_VALUE` per the reference.
- [x] [`MatrixGame2CheckpointConverter`](../../src/HartsyInference.ModelHandler/CheckpointConverters/MatrixGame2CheckpointConverter.cs) — delegates to the shared Matrix-Game routing (original Wan naming + `action_model` + `img_emb` renames) and adds per-variant action-shape inference from weights (`keyboard_embed.0.weight` → K, `mouse_mlp` presence). Wan2.1 VAE loads via the existing `LoadVae` path after offline `.pth` → safetensors conversion.
- [ ] `MatrixGame2GenerationTests` — env-var-gated real-checkpoint entry (needs the HF download + VAE/CLIP conversions; VRAM probe ≥ 12 GB bf16).
- [x] Structural tests — `MatrixGame2ModelTests` (DMD scheduler exactness, mask semantics, Wan2.1 VAE round trip + streaming decode, CLIP/action sensitivity), `MatrixGame2PipelineTests` (2-block tiny e2e rollout, variant encoders, rejections), `MatrixGame2CheckpointConverterTests`.
- [ ] `tests/python-reference/dump_matrix_game_2_full_forward.py` + `diff_matrix_game_2_layers.py` — layer-by-layer F32 diff harness (checkpoint-gated).
- [ ] `RollingKvCache` + streaming `IFrameStepper` wiring (shared with MG3's interactive variant).

## 5. Implementation — Matrix-Game 3.0 pipeline (flagship, Apache-2.0)

> Target: RTX 4090 24 GB minimum. Source-of-truth: [MATRIX_GAME_3_ARCHITECTURE.md](../Research/MATRIX_GAME_3_ARCHITECTURE.md). User explicitly requested this; primary deliverable for the phase.
>
> **Status (2026-06-10): CORE BUILT END-TO-END (canned-action mode)** — structurally CPU-verified, numerics validation-pending. The DiT reuses the Wan2.2 video stack directly (`WanVideoBlock` + `WanRope` + the hoisted `WanDitOps`); the VAE is the already-built `Wan22VaeDecoder`; the per-frame-timestep machinery is the Wan I2V path. File layout deviates from the plan below where reuse demanded it: the transformer pieces live in `HartsyInference.Diffusion/Models/Denoisers/` next to every other DiT (not `HartsyInference.Video`).

- [x] Canned-action pipeline — [`src/HartsyInference.Interactive/Pipelines/MatrixGame3Pipeline.cs`](../../src/HartsyInference.Interactive/Pipelines/MatrixGame3Pipeline.cs): segment loop (bootstrap 4-past/5-mem from the seed-image latent → FlowUniPC denoise with per-frame timesteps (clean frames 0) → per-segment decode → history roll + FOV memory retrieval + action→pose integration + Plücker tokens). CFG null path zeroes the memory slots (reference parity). **Caveats (validation-gated):** per-segment independent VAE decode ((T−1)·4+1 frames after segment 0 instead of the reference's cache-continued 40), seed image must arrive as a pre-encoded normalized latent (`Wan22VaeEncoder` not built), integrator step sizes + Plücker flatten order + memory RoPE indices unverified.
- [ ] `MatrixGame3InteractivePipeline` — live-action `IFrameStepper` wiring on top of the segment core (the session machinery is built + tested; the stepper needs per-segment state extracted from the pipeline).
- [x] Wan2.2 DiT — **reused** (`WanVideoBlock`/`WanRope`/`WanDitOps` hoisted from the Phase 9 Wan build; no new `Wan22Dit`). [`MatrixGame3Transformer`](../../src/HartsyInference.Diffusion/Models/Denoisers/MatrixGame3Transformer.cs) adds the memory-augmented sequence (mem ‖ past ‖ current, historical RoPE t-indices via the new `WanRope` explicit-index overload), trailing-frames readout, ActionModule hooks, and the Plücker projection. Shape contradictions (5120/40 vs 3072/30) resolved at load time by `MatrixGame3CheckpointConverter.InferShape`.
- [x] [`MatrixGame3ActionModule`](../../src/HartsyInference.Diffusion/Models/Denoisers/DiTBlocks/MatrixGame3ActionModule.cs) — dual-stream (mouse: window-grouped raw actions + image token → MLP → temporal self-attn, RoPE `[8,28,28]` θ=256; keyboard: embed → window → K/V cross-attended by an image-token Q), inserted between cross-attn and FFN via a generic `WanVideoBlock` hook. `local_attn_size` windowing + memory action streams deferred (validation-gated).
- [ ] `MgLightVaeDecoder` — pruned Wan2.2 decoder (shape-tolerant loading, rate from `decoder.conv1.weight.shape[0]`).
- [x] [`FlowUniPCMultistepScheduler`](../../src/HartsyInference.Diffusion/Schedulers/FlowUniPCMultistepScheduler.cs) — first UniPC in the project (predictor + corrector, bh1/bh2, order ramp, shifted flow sigmas), in `HartsyInference.Diffusion/Schedulers/` with the other schedulers. Gated by an analytic constant-velocity exactness test at 3/10/50 steps.
- [x] `MatrixGame3ActionEncoder` (see § 3) — 6-key one-hot + clamped mouse Δ payload.
- [x] Plücker — `PluckerEmbedding` (6-ch rays; Open Question 4 resolved: no 16-channel ray format) + [`MatrixGame3PluckerTokens`](../../src/HartsyInference.Interactive/Camera/MatrixGame3PluckerTokens.cs) (per-token 32×32-pixel block grouping → `patch_embedding_wancamctrl` input; flatten order validation-gated).
- [x] Memory retrieval — `FrustumOverlapSelector` (see § 3) + the stride-8 candidate window in the pipeline.
- [x] [`MatrixGame3CheckpointConverter`](../../src/HartsyInference.ModelHandler/CheckpointConverters/MatrixGame3CheckpointConverter.cs) — routes `action_model`/`action_module`/Plücker keys BEFORE the Wan renames (which would corrupt them), reuses `WanVideoCheckpointConverter.MapKey` for the core, slices the distilled student prefix (`student.`/`generator.`, drops critic/EMA — prefix set validation-gated), infers the real DiT shape from weights, collects ActionModule block indices.
- [ ] `MgLightVaeConverter` — with `MgLightVaeDecoder`.
- [ ] `MatrixGame3GenerationTests` — env-var-gated real-checkpoint entry (needs umT5 + Wan2.2 VAE + DiT download paths; VRAM probe ≥ 20 GB bf16). First-run numeric validation.
- [x] Structural tests — `MatrixGame3TransformerTests` (memory-augmented forward, readout shape, actions/Plücker change output), `MatrixGame3PipelineTests` (2-segment tiny e2e rollout, validation rejections), `MatrixGame3CheckpointConverterTests`, `FlowUniPCSchedulerTests`. 
- [ ] `tests/python-reference/dump_matrix_game_3_full_forward.py` + `diff_matrix_game_3_layers.py` — layer-by-layer F32 diff harness (checkpoint-gated).

## 6. Implementation — Oasis-500m pipeline (pedagogical, MIT)

> Target: any GPU (smallest model in the phase). Source-of-truth: [OASIS_ARCHITECTURE.md](../Research/OASIS_ARCHITECTURE.md). Acts as the CI smoke test for the entire phase's action-conditioning correctness.
>
> **Status (2026-06-30): ✅ VERIFIED (real weights, CUDA/3060)** — ViT-VAE encode corr 0.99999999, decode corr 1.0, DiT-S/2 v-pred corr 1.0 (maxAbs 3e-5) vs `etched-ai/open-oasis`. Checkpoint keys matched the C# exactly (no rewrite); 1 bug fixed (DiT unpatchify vec-order `[py,px,c]`, see PARITY_VERIFICATION §Bugs). Oracle `tests/python-reference/oasis_reference/`; gated test `OasisParityTests` (`PARITY_BACKEND=cuda`). E2E AR rollout follows by composition. Weights via the `camenduru/oasis-500m` mirror (the `Etched` repo is gated). Full RGB-in/RGB-out in pure C#.

- [x] [`OasisPipeline.cs`](../../src/HartsyInference.Interactive/Pipelines/OasisPipeline.cs) — autoregressive frame loop: VAE-encode prompt (×0.07843137255), per new frame append ±20-clamped noise, 10 v-pred DDIM steps with **Diffusion Forcing** (context frames pinned at stabilization index 14, only the target frame stepped/written), 32-frame sliding window, zero action prepended. Deterministic per seed (test-gated byte-exact replay).
- [x] [`OasisDit.cs`](../../src/HartsyInference.Diffusion/Models/Denoisers/OasisDit.cs) + [`OasisDitConfig`](../../src/HartsyInference.Diffusion/Models/Denoisers/OasisDitConfig.cs) — DiT-S/2 (16L × 1024 × 16h), per-frame conditioning `c[t] = TimestepEmbed(noiseIdx[t]) + Linear(action[t])` (the `TimestepAddon` pattern). In `Diffusion/Models/Denoisers/` per project convention (not the Interactive path planned below).
- [x] [`OasisSpatioTemporalBlock.cs`](../../src/HartsyInference.Diffusion/Models/Denoisers/DiTBlocks/OasisSpatioTemporalBlock.cs) — Latte-style halves: bidirectional spatial axial attention (2-D "pixel"-mode [`AxialRope2D`](../../src/HartsyInference.Diffusion/Models/Denoisers/DiTBlocks/AxialRope2D.cs), new reusable primitive) + causal temporal axial attention (1-D RoPE via `WanRope`, additive causal mask), each with its own adaLN-zero modulation.
- [x] [`OasisVitVae.cs`](../../src/HartsyInference.Diffusion/Models/Vae/OasisVitVae.cs) — continuous Gaussian **pure-transformer** VAE (patch 20, shallow 6-layer encoder / 12-layer decoder, linear Gaussian bottleneck, μ taken deterministically). First ViT-VAE in the codebase.
- [ ] `ClipVitL20VisionEncoder` — not needed for generation (the VAE *is* the vit-l-20 checkpoint); revisit only if a context-frame-embedding variant emerges.
- [x] [`OasisActionEncoder.cs`](../../src/HartsyInference.Interactive/ActionEncoders/OasisActionEncoder.cs) — 25-dim VPT vector (camera floats at slots 15/16), normalized-floats-first API (VPT bucket math documented, not required), `IActionEncoder` + static `BuildRow` for canned plans.
- [x] [`DdimVPredScheduler.cs`](../../src/HartsyInference.Diffusion/Schedulers/DdimVPredScheduler.cs) — sigmoid β-schedule (T=1000, −3..3) + per-frame v-param DDIM step (`t = −1` ⇒ clean), in `Diffusion/Schedulers/` with the rest. Gated by analytic exactness tests on a known (x₀, ε) trajectory. This was the last missing Phase 9 § 3c scheduler family.
- [x] [`OasisCheckpointConverter.cs`](../../src/HartsyInference.ModelHandler/CheckpointConverters/OasisCheckpointConverter.cs) — 1:1 key pass-through (wrapper-prefix strip + recomputed `rotary_freqs` drop); key names validation-gated on first dump.
- [ ] `OasisGenerationTests` — env-var-gated real-checkpoint entry (`OASIS_500M_PATH` / `OASIS_VAE_PATH`, 360×640, byte-compare vs Python reference). Structural tests are in: `OasisModelTests` (VAE round trip, causality + action-sensitivity gates), `OasisPipelineTests` (deterministic rollout), `DdimVPredSchedulerTests`.

## 7. Implementation — Hunyuan-GameCraft 1.0 — **structural build, numerics validation-pending (2026-06-15)**

> **License decision (owner):** GameCraft gets **no special treatment**. The engine is MIT, ships **no weights and no Tencent code**; the user supplies weights into `/Models` like every other model. → **No license-gate framework** was built; weight-use is the end user's responsibility, same as every checkpoint. (Supersedes the earlier license-gate plan — `TencentHunyuanCommunityLicense` / `LicenseAcceptance` / `LicenseAcceptanceEndpoint` are **not** built.)
>
> Target hardware: A100/H100 40+ GB / cloud GPU (12.5B). The 3060 is for CPU structural tests only. Spec: [HUNYUAN_GAMECRAFT_ARCHITECTURE.md](../Research/HUNYUAN_GAMECRAFT_ARCHITECTURE.md). All architecture dims/keys marked **[VG]** are reconciled against the real checkpoint during the diff pass (§9).

**Reuse-first outcome:** most of GameCraft is the existing foundation — HunyuanVideo 3D VAE, Llava (`LlamaStyleEncoder`), CLIP-L (`ClipTextEncoder`), `FlowMatchEulerDiscreteScheduler(shift=5)`, and the whole Interactive session/Plücker/Se3/memory stack. Genuinely new code: the `.pt` loader, the HunyuanVideo MM-DiT (via N-axis rope reuse of the image blocks), and the GameCraft-specific camera/action/latent parts.

New **reusable backend** (used by future models too):
- [x] `src/HartsyInference.ModelHandler/PyTorch/PytorchPickleLoader.cs` (+ `PickleMachine.cs`) — safe-subset torch `.pt` reader → `Dictionary<string,Tensor>`. Reusable for any `.pt` model (Cosmos, …). Unit-tested. **Replaces the planned one-off Python conversion script.**
- [x] `src/HartsyInference.Diffusion/Models/Denoisers/HunyuanVideoDit.cs` (+ `HunyuanVideoConfig.cs`; 3-axis rope) — 19 double + 38 single blocks, dim 3072, heads 24, RoPE `[16,56,56]`, 33-ch patchify → 16-ch velocity, camera-token fusion hook. **Reuses `HunyuanImageBlock`/`HunyuanImageSingleBlock`** via the generalized N-axis `HunyuanImageRope` (image 2D path byte-identical). Co-located in Diffusion (with the blocks + VAE) instead of Video to avoid a new cross-package dep. Reusable for an offline HunyuanVideo pipeline. CPU structural test passes.
- [x] HunyuanVideo `884-16c-hy0801` 3D VAE — **already existed** (`Diffusion/Models/Vae/HunyuanVideoVae{Decoder,Encoder}`), reused as-is.
- [x] Llava-Llama-3-8B + CLIP-L — **reused** existing `LlamaStyleEncoder` (`Llava3_8B` preset) + `ClipTextEncoder`; no new encoder file.
- [x] Flow-match scheduler — **reused** `FlowMatchEulerDiscreteScheduler(shift=5)` (covers base 50-step / distilled 8-step); no new scheduler file.

New **GameCraft-specific** parts (all CPU-tested):
- [x] `src/HartsyInference.Interactive/ActionEncoders/GameCraftActionEncoder.cs` — `IActionEncoder` (PluckerMap); WASD+speed+mouse → cumulative pose (`Se3Math`) → 6-ch Plücker map (`PluckerEmbedding`). Per-chunk cadence.
- [x] `src/HartsyInference.Interactive/Models/GameCraftCameraNet.cs` — PixelUnshuffle(8) → conv 384→192→96→16 (GN+ReLU) → temporal compress → PatchEmbed → scale → image-grid tokens.
- [x] `src/HartsyInference.Interactive/Pipelines/GameCraftLatentBuilder.cs` — 33-ch composite `[noisy16 + history16 + mask1]`.
- [x] `src/HartsyInference.Interactive/Pipelines/HunyuanGameCraftPipeline.cs` — `DenoiseChunk` (composite + camera tokens + DiT + 2-pass CFG + scheduler) + `GenerateChunk`/`DecodeLatentToFrames`. CPU end-to-end `DenoiseChunk` test passes.
- [x] `src/HartsyInference.Interactive/Sessions/GameCraftFrameStepper.cs` — `IFrameStepper` for live sessions (history fed forward).
- [x] `src/HartsyInference.ModelHandler/CheckpointConverters/HunyuanGameCraftCheckpointConverter.cs` — routes `.pt` keys → `{Dit, CameraNet, Vae, Llava, Clip}`. **No license gate.**

Remaining (numeric pass, **[VG]**, needs cloud GPU + real checkpoint):
- [ ] Reconcile the original→diffusers per-block key remap in the converter against the real `.pt`.
- [ ] Reconcile timestep scaling, the txt token-refiner (currently a plain projection), the CameraNet temporal schedule, and GameCraft dims via the diff harness (§9).
- [ ] Distilled (8-step / CFG 1.0) variant parity once base is ✅.

## 8. Server integration

- [ ] `src/HartsyInference.Server/Endpoints/InteractiveSessionEndpoint.cs` — WebSocket endpoint that wraps `IInteractiveSession` (action input via inbound messages, frames via outbound messages). Per-connection session lifecycle.
- [ ] `src/HartsyInference.Server/Streaming/InteractiveFrameStream.cs` — frame-serialization adapter (PNG, JPEG, or raw RGB depending on Accept-Encoding).
- [ ] Server test: `tests/HartsyInference.Server.Tests/InteractiveSessionWsTests.cs` — end-to-end smoke test with the Oasis-500m fixture model.

## 9. Testing & Validation

- [ ] All `*GenerationTests` skip cleanly when env vars (`MATRIX_GAME_2_BASE_PATH`, `MATRIX_GAME_3_BASE_PATH`, `OASIS_500M_PATH`, `HUNYUAN_GAMECRAFT_PATH`) are missing.
- [ ] All `*GenerationTests` perform a VRAM probe before allocating and skip when below the documented minimum.
- [ ] Reference validation: each model has a `dump_*_full_forward.py` Python reference + `diff_*_layers.py` per-layer diff harness, mirroring SD3.5 / Z-Image / Lance conventions. **GameCraft harness landed** (`dump_gamecraft_full_forward.py` + `diff_gamecraft_layers.py`, reference module TODO[VG]); env-gated `GameCraftRealCheckpointTests` validates the `.pt`→converter chain on real weights.
- [ ] Performance gates per per-model doc (Matrix-Game 2.0: 25 FPS @ 540p on RTX 4090; Matrix-Game 3.0: ≥10 FPS @ 720p distilled on RTX 4090).
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
- [ ] Documentation pass on `HartsyInference.Interactive` README and sample apps (interactive Minecraft-like loop with Oasis, Matrix-Game 3.0 game-engine snippet)
- [ ] Merge to main branch — user action

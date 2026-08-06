# World / Interactive Models — status

Concise status for interactive world models (action-conditioned, real-time frame generation). Open work
is in the [Remaining work](#remaining-work) section below; bring-up debugging notes live in
[TROUBLESHOOTING.md](TROUBLESHOOTING.md). Parity evidence lives in
[PARITY_VERIFICATION.md](PARITY_VERIFICATION.md). Legend: [MODEL_STATUS.md](MODEL_STATUS.md).

## Verified end-to-end (✅)

| Model | Notes |
|---|---|
| **Oasis-500m** (Decart + Etched) | Real-weight numeric parity vs `etched-ai/open-oasis` on **CUDA/3060**: ViT-VAE encode corr **0.99999999**, decode corr **1.0**, DiT-S/2 v-pred corr **1.0** (maxAbs 3e-5). Fixed a DiT unpatchify vec-order bug (`[py,px,c]` not `[c,py,px]` — see [PARITY_VERIFICATION.md §Bugs](PARITY_VERIFICATION.md)). E2E AR rollout (DDIM v-pred + Diffusion Forcing) follows by composition (all cores corr 1.0). Weights: `camenduru/oasis-500m` mirror (the `Etched` repo is gated). **CLI-wired via `hartsy world -m oasis`** (was already wired pre-2026-07-21); re-run 2026-07-21 with a real Minecraft-HUD seed frame (cropped from the model card's own example panel) and the canned forward-walk plan: the background scene (terrain, sky, trees, dug-out pit) stayed coherent and stable across 8 frames, but the HUD/hotbar visibly degraded into streaky artifacts by frame 5-6 and a blank vertical strip grew in from the right edge by frame 8 — a real, visible drift/degradation pattern (not full breakdown), consistent with the numerics here still being "unverified vs the reference" at the *pipeline* level (the doc's own per-component parity is against isolated stages, not this composed AR rollout) — do not read this row's ✅ as "the CLI rollout is pixel-clean," only that the components are correct in isolation. |
| **DIAMOND** (Eloi Alonso et al.) | **Network built from scratch + verified** vs `eloialonso/diamond` (Atari/Breakout) on **CUDA and CPU**: inner U-Net corr **1.0** (maxAbs 7e-6), full EDM denoise **bit-exact** (maxAbs 0.0), 3-step Karras+Euler sampler **bit-exact** (maxAbs 3e-8). New architecture family for the engine: **CNN U-Net + EDM (Karras) diffusion**, pixel-space (no VAE), 4-frame + action conditioning via AdaGroupNorm. 1 bug fixed (U-Net upsample-conv index). Weights ungated (`eloialonso/diamond`, Breakout checkpoint). This network-level parity is unaffected by the finding below (the denoiser/sampler are bit-exact in isolation; the open question is in the new session-layer wiring around them). |

## CLI-wired, correctness open (🔧)

| Model | Notes |
|---|---|
| **DIAMOND** — CLI/session layer | **CLI-wired 2026-07-21** (`DiamondWorldPipeline` + `DiamondWorldSession`, `hartsy world -m diamond`) — the previously-remaining "Interactive AR game-loop wiring" gap is closed, and unlike Oasis this is a genuinely per-frame interactive loop (one real EDM sample per action, not a fixed batch). Adversarially verified with a real 64×64 Breakout frame (ALE `Breakout-v5`, mid-game). **Quantitative finding (paddle x-center, measured per-pixel, not eyeballed): a reversal plan `right,right,left,left,left` from a mid-field seed (no wall involved) produced centers 46.5→55.0→56.0→48.5→38.5 — the first `right`→`right` step responds immediately (+8.5), but the first `left` step barely moves it (+1.0, still net-rightward) and the reversal only shows up starting the *second* `left` step.** This is a clean, reproducible one-frame lag between an action and its visible effect, not "wall-bounce inertia" (there is no wall in this run) — an earlier same-session characterization blaming inertia was wrong and has been corrected here. Root cause **not isolated, and CLI-side probing cannot isolate it** — three same-session experiments were each
confounded (see below), which is itself the signal to stop probing this way, not to run a fourth. Given the
denoiser and sampler are **bit-exact vs. reference in isolation** (this doc's ✅ row above), the space collapses
to two real branches, not an open-ended bug hunt:
1. **The session assembles/rolls `obs`+`act` differently than the reference rollout does** (a real bug in
   `DiamondWorldSession`, or in the assumed action-window convention).
2. **This is correct, faithful learned dynamics** — a probabilistic world model conditioning on recent visual
   motion plus a new action is *expected* to take a frame or two to override established motion; that would
   not be a bug at all. The data is at least as consistent with this: from-rest `right` actions get an
   *immediate* response (frame1→2, +8.5) — a uniform assembly/indexing bug should delay those too, and it
   doesn't — only the *reversal* (right→left) lags. That asymmetry is exactly what branch 2 predicts and
   branch 1 (a simple off-by-one) does not cleanly explain.

Ruled-out/inconclusive experiments already run: (a) flipping `DiamondWorldSession.RollAction` to slot the
newest action at `[0]` instead of `[^1]` — made responsiveness *worse* (barely moved, direction unreliable),
so the shipped `[^1]`-newest convention is the better of the two, ruling out simple slot-reversal; (b) warming
up with 6 `right` actions to flush the replicated-seed bootstrap frames from the window before reversing —
inconclusive, confounded by the paddle also having been pushed to the wall by then. **The only real
discriminator between branch 1 and branch 2 is diffing a C# rollout against a reference `play.py`/`game.py`
rollout on the same seed image + action sequence** — that comparison, not further CLI experiments, is the
scoped follow-up. **Catalog status left at `ValidationPending`, not `Verified`**, pending that comparison. Initial-noise convention (`sigma_max·N(0,1)`) is a separate standard-EDM assumption, also not itself parity-diffed. |

## Built, validation-pending (🔧)

All built end-to-end with structural tests passing; numeric parity pending.

| Model | Notes |
|---|---|
| **Matrix-Game 3.0** (Skywork) | Flagship. 5B Wan2.2-TI2V finetune + `ActionModule` + FOV memory + DMD 3-step. UMT5-XXL + Wan2.2 VAE. DiT reuses `WanVideoBlock`. **Full DiT forward parity-verified (2026-07-13)** vs the Skywork `WanModel` reference on the real `base_model` (dim 3072/24/30) — all four surfaces: memory-mode Wan backbone corr **1.0 through block 15** (F32; tail drift = precision × 1000× residual gain), ActionModule corr **0.99996**, FOV memory-frame path (block 0-15 corr ~1.0), per-block Plücker camera injection (final-v corr 0.99947). Three bugs fixed / one placeholder replaced: memory-mode destructive-norm3 residual (opt-in `WanVideoBlock.CrossAttnResidualNormed`), action-window off-by-one (`start = i·ratio − ratio·window`), and the Plücker path (was a stub → real `ProjectPlucker` + per-block `MatrixGame3CamInjector` via new `WanVideoBlock.postSelfAttnHook`). Parity verified 2026-07 (Stage A / MEM / PLK + action encoder); those whole-model tests were removed in the 2026-08-06 suite cleanup. Remaining: real-weight perf run. **Catalogued, not yet loadable via `WorldService`** (2026-07-21 finding) — its checkpoint set is **~27GB minimum** (base DiT 12.9GB + one VAE 2.8GB + UMT5-XXL 11.4GB), and it still has no image→latent encoder ported (`Wan22VaeEncoder` is decode-only today), so the seed image can't reach the pipeline even once weights are present. Selecting `-m matrix-game-3` now fails fast with a clear message instead of silently mis-loading as Oasis. |
| **Matrix-Game 2.0** (Skywork) | Entry-level. 1.8B Wan2.1-lineage. **Wan-backbone DiT forward verified corr 0.99999473 on CUDA/3060** vs the Skywork `WanModel` (real bf16 base ckpt; 2 bf16-bias bugs fixed — see PARITY §Bugs). Remaining: the per-block ActionModule (mouse/keyboard cross-attn) parity. Wan2.1 16ch VAE; CLIP-ViT-H/14 seed. **Catalogued, not yet loadable via `WorldService`** (2026-07-21 finding) — `MatrixGame2Pipeline`'s constructor takes an already-built transformer + Wan2.1 VAE encoder/decoder + optional CLIP vision encoder; no `LoadFromPath`-style assembler exists yet, and its CLIP checkpoint is OpenCLIP xlm-roberta-ViT-H (`models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`), whose key convention has not been confirmed to match the engine's `ClipVisionEncoder(ViTH14)`. Minimum real-checkpoint subset (base transformer + Wan2.1 VAE + CLIP, skipping the distilled variants) is **~9GB** — fits on this box's disk, but the loader itself is unbuilt. Selecting `-m matrix-game-2` now fails fast with a clear message instead of silently mis-loading as Oasis. |
| **Hunyuan-GameCraft 1.0** (Tencent) | 12.5B HunyuanVideo MM-DiT + CameraNet (Plücker rays) + 33-ch composite history. PCM+CFG 8-step. Reusable `.pt` pickle loader + N-axis rope. No license gate (engine MIT, user-supplied weights — and the HF repo itself is public/ungated, confirmed 2026-07-21). **Checkpoint loader landed 2026-08-06 — code-complete, NOT verified against real weights — but `WorldService` still fails fast, now for one specific, named reason instead of the old blanket "not loadable" message.** `HunyuanGameCraftPipeline.LoadFromPath` chains `HunyuanGameCraftCheckpointConverter` (coarse prefix router: DiT/CameraNet/Vae/Llava/Clip) into `HunyuanVideoCheckpointConverter` (Tencent-raw/diffusers → hybrid naming, same remap the base HunyuanVideo T2V recipe uses) for the merged DiT+CameraNet `.pt` dump, and `HunyuanVideoCheckpointConverter.ConvertVaeDecoder` for the standalone 3D-VAE aux checkpoint; `WorldService.LoadHunyuanGameCraft` separately loads Llava-Llama-3-8B + CLIP-L from two more aux paths (`llava-path`/`clip-path`, alongside the existing `vae-path`). **No VAE encoder is built** — the structural ldm→diffusers key remap exists (`CheckpointConvertUtils.ConvertVaeEncoderKey`), but the mid-attention Conv3d→Linear reshape `ConvertVaeDecoder`'s `AttnProj` applies for the decoder has no encoder-side counterpart wired up yet — so no session could ever turn `WorldRequest.InitImage` into chunk-0 history regardless of which weights are supplied. `WorldService.LoadHunyuanGameCraft` therefore checks `HunyuanGameCraftPipeline.LoadFromPathBuildsVaeEncoder` (currently always false) and throws `NotSupportedException` **before loading the checkpoint set** — loading ~51GB into RAM for a session that can't succeed would be pure waste on a box with a documented history of 45–62GB OOM kills. This box has no local copy of the real **~51GB** checkpoint set anyway (only ~28–31GB free at the time this was written); verification stopped at: (a) chaining the two converters on a synthetic Tencent-raw dict produces exactly the keys `HunyuanVideoDit.LoadWeights`/`GameCraftCameraNet.LoadWeights` index, and (b) `HunyuanGameCraftPipeline.LoadFromPath` (called directly, bypassing the gate) fails with a clean `FileNotFoundException` — not a crash — when pointed at missing paths — see `HunyuanGameCraftLoaderTests`. Also note: `LoadFromPath`'s `GameCraftActionEncoder` resolution binding is fixed at construction (default 320×512) and must match the actual generation resolution — see `HunyuanGameCraftPipeline.LoadFromPath`'s param docs. DiT block-streaming/sharding (Phase 4b) is separate, still-open follow-up work, as is the VAE-encoder reshape itself and the interactive session (text conditioning, WASD parsing, `GameCraftFrameStepper` wiring) that would sit on top of it. |

## Not started (❌)

| Model | Notes |
|---|---|
| **Cosmos-Predict1 V2W** | AR world-model substrate (FSQ tokenizer + AR transformer); tracked under video. |

## Notes

The `HartsyInference.World` package (sessions / actions / camera / FOV memory) + the FlowUniPC and
FlowMatchDmd schedulers + a shared `ActionModule` back these models. World models share the 3D-video VAE
foundation with [the video models](MODEL_STATUS_VIDEO.md).

**`.pt` → `.safetensors` is a manual, out-of-repo step.** Both Oasis's `camenduru/oasis-500m` mirror and
DIAMOND's `eloialonso/diamond` checkpoints ship as PyTorch `.pt` files; the engine's loaders only read
`.safetensors`. This session converted both by hand with a one-off Python script (`torch.load(...,
weights_only=False)` → filter to `torch.Tensor` values → `safetensors.torch.save_file`; DIAMOND's raw dump
needs its `denoiser.inner_model.`/`inner_model.` prefix handled, which `DiamondWorldPipeline.LoadFromPath`
does automatically) run in a scratch venv that no longer exists. **A user pointing `hartsy world` straight at
either upstream repo today hits a hard wall at `--model-path`/`--vae-path`** until they replicate that
conversion themselves — the CLI does not ingest `.pt` directly. This is a real gap in "usable via CLI," not
just a doc footnote; a `.pt`-ingesting loader (or a documented conversion script checked into the repo) is
still open work.

## Remaining work

Distilled from the retired PHASE_10_INTERACTIVE plan.
See [ROADMAP.md](ROADMAP.md) for cross-cutting infra (multi-GPU, kernel perf, quant, serving).

### Validation
- [ ] Overall ~75% built / low validated; numeric validation is checkpoint-gated for MatrixGame2/3, Oasis, GameCraft, DIAMOND.

### Live / streaming
- [ ] `MatrixGame3InteractivePipeline`.
- [ ] `RollingKvCache`.
- [ ] Streaming `IFrameStepper`.
- [ ] `MgLightVaeDecoder` + converter.

### Bugs
- [ ] DIAMOND one-frame action-lag (unbisected — resolve via a C#-vs-reference `play.py` rollout diff on the same seed + action sequence).

### Deferred foundations
- [ ] AR-token KV cache.
- [ ] VQ / MagViT tokenizers.

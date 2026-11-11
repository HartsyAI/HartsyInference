# World / Interactive Models — status

Concise status for interactive world models (action-conditioned, real-time frame generation). Open work
is in the [Remaining work](#remaining-work) section below; bring-up debugging notes live in
[TROUBLESHOOTING.md](TROUBLESHOOTING.md). Parity evidence lives in
[PARITY_VERIFICATION.md](PARITY_VERIFICATION.md). Legend: [MODEL_STATUS.md](MODEL_STATUS.md).

## Verified end-to-end (✅)

| Model | Notes |
|---|---|
| **Oasis-500m** (Decart + Etched) | Real-weight numeric parity vs `etched-ai/open-oasis` on **CUDA/3060**: ViT-VAE encode corr **0.99999999**, decode corr **1.0**, DiT-S/2 v-pred corr **1.0** (maxAbs 3e-5). ([details](#oasis-500m)) |
| **DIAMOND** (Eloi Alonso et al.) | **Network built from scratch + verified** vs `eloialonso/diamond` (Atari/Breakout) on **CUDA and CPU**: inner U-Net corr **1.0** (maxAbs 7e-6), full EDM denoise **bit-exact** (maxAbs 0.0), 3-step Karras+Euler sampler **bit-exact** (maxAbs 3e-8). ([details](#diamond)) |
| **SANA-WM** (NVIDIA, stage 1) | Real-weight output matches NVIDIA's stage-1 reference on **CUDA/4090 and 3060 (12 GB)**: one DiT step cosine **0.99986**, a 161-frame clip PSNR min/mean **28.6 / 34.9 dB**, SSIM **0.947 / 0.977**. CLI only (`hartsy world -m sana-wm`): no SwarmUI or HTTP route, so the catalog status stays `ValidationPending`. Stage 1 only, which upstream calls lower quality than its refiner output. Non-commercial. ([details](#sana-wm-stage-1)) |
| **SANA-WM streaming** (NVIDIA, stage 1) | The chunk-causal distilled model, `hartsy world -m sana-wm-streaming`. Matches upstream's own streaming sampler on **CUDA/4090 and 3060 (12 GB)**: all 15 DiT calls cosine ≥ **0.9996**, all 73 frames PSNR min **44.9 dB** (SSIM min 0.9973), including a live action-script drive. One 24-frame chunk takes **1.75 s** on the 4090 and **7.7 s** on the 3060. CLI only; the refiner is a separate row and unverified. ([details](#sana-wm-streaming)) |

## CLI-wired, correctness open (🔧)

| Model | Notes |
|---|---|
| **DIAMOND** — CLI/session layer | **CLI-wired 2026-07-21** (`DiamondWorldPipeline` + `DiamondWorldSession`, `hartsy world -m diamond`) — the previously-remaining "Interactive AR game-loop wiring" gap is closed, and unlike Oasis this is a genuinely per-frame interactive loop (one real EDM sample per action, not a fixed batch). ([details](#diamond-2)) |

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
| **Matrix-Game 3.0** (Skywork) | Flagship. ([details](#matrix-game-30)) |
| **Matrix-Game 2.0** (Skywork) | Entry-level. ([details](#matrix-game-20)) |
| **Hunyuan-GameCraft 1.0** (Tencent) | 12.5B HunyuanVideo MM-DiT + CameraNet (Plücker rays) + 33-ch composite history. ([details](#hunyuan-gamecraft-10)) |
| **SANA-WM streaming refiner** (stage 2) | `--refiner`, off by default. The chunk-causal LTX-2.0 19B video transformer, Gemma-3 text path and connector. Matches upstream's own `RefinerChunkRunner` on a tiny random model on the CPU (3e-5 relative). **Never run on a GPU or on real weights**, and not compared with upstream: that needs the release's BF16 Gemma-3-12B (24.4 GB, not downloaded). ([details](#sana-wm-streaming)) |

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
- [x] SANA-WM streaming: a live viewer. `--view` plays frames as they are generated and steers the run from held keys; see [Research/WORLD_LIVE_VIEWER.md](../Research/WORLD_LIVE_VIEWER.md). Verified on the RTX 4090 through the window with real key events; the evidence is in that PR.
- [ ] SANA-WM streaming refiner: run it on a GPU and compare it with upstream. That needs the release's BF16 Gemma-3-12B (24.4 GB) and block streaming for the 19B transformer on a 24 GB card.
- [ ] SANA-WM and SANA-WM streaming through SwarmUI. There is no route yet, and the catalog status stays `ValidationPending` until there is one.
- [ ] `MatrixGame3InteractivePipeline`.
- [ ] `RollingKvCache`.
- [ ] Streaming `IFrameStepper`.
- [ ] `MgLightVaeDecoder` + converter.

### Bugs
- [ ] DIAMOND one-frame action-lag (unbisected — resolve via a C#-vs-reference `play.py` rollout diff on the same seed + action sequence).

### Deferred foundations
- [ ] AR-token KV cache.
- [ ] VQ / MagViT tokenizers.

## Details

Verification evidence, bugs found, and caveats for the rows above. Moved out of the status
tables on 2026-08-06 so the tables stay scannable — no content was dropped.

### Oasis-500m

Real-weight numeric parity vs `etched-ai/open-oasis` on **CUDA/3060**: ViT-VAE encode corr **0.99999999**, decode corr **1.0**, DiT-S/2 v-pred corr **1.0** (maxAbs 3e-5). Fixed a DiT unpatchify vec-order bug (`[py,px,c]` not `[c,py,px]` — see [PARITY_VERIFICATION.md §Bugs](PARITY_VERIFICATION.md)). E2E AR rollout (DDIM v-pred + Diffusion Forcing) follows by composition (all cores corr 1.0). Weights: `camenduru/oasis-500m` mirror (the `Etched` repo is gated). **CLI-wired via `hartsy world -m oasis`** (was already wired pre-2026-07-21); re-run 2026-07-21 with a real Minecraft-HUD seed frame (cropped from the model card's own example panel) and the canned forward-walk plan: the background scene (terrain, sky, trees, dug-out pit) stayed coherent and stable across 8 frames, but the HUD/hotbar visibly degraded into streaky artifacts by frame 5-6 and a blank vertical strip grew in from the right edge by frame 8 — a real, visible drift/degradation pattern (not full breakdown), consistent with the numerics here still being "unverified vs the reference" at the *pipeline* level (the doc's own per-component parity is against isolated stages, not this composed AR rollout) — do not read this row's ✅ as "the CLI rollout is pixel-clean," only that the components are correct in isolation.

### DIAMOND

**Network built from scratch + verified** vs `eloialonso/diamond` (Atari/Breakout) on **CUDA and CPU**: inner U-Net corr **1.0** (maxAbs 7e-6), full EDM denoise **bit-exact** (maxAbs 0.0), 3-step Karras+Euler sampler **bit-exact** (maxAbs 3e-8). New architecture family for the engine: **CNN U-Net + EDM (Karras) diffusion**, pixel-space (no VAE), 4-frame + action conditioning via AdaGroupNorm. 1 bug fixed (U-Net upsample-conv index). Weights ungated (`eloialonso/diamond`, Breakout checkpoint). This network-level parity is unaffected by the finding below (the denoiser/sampler are bit-exact in isolation; the open question is in the new session-layer wiring around them).

### DIAMOND (2)

**CLI-wired 2026-07-21** (`DiamondWorldPipeline` + `DiamondWorldSession`, `hartsy world -m diamond`) — the previously-remaining "Interactive AR game-loop wiring" gap is closed, and unlike Oasis this is a genuinely per-frame interactive loop (one real EDM sample per action, not a fixed batch). Adversarially verified with a real 64×64 Breakout frame (ALE `Breakout-v5`, mid-game). **Quantitative finding (paddle x-center, measured per-pixel, not eyeballed): a reversal plan `right,right,left,left,left` from a mid-field seed (no wall involved) produced centers 46.5→55.0→56.0→48.5→38.5 — the first `right`→`right` step responds immediately (+8.5), but the first `left` step barely moves it (+1.0, still net-rightward) and the reversal only shows up starting the *second* `left` step.** This is a clean, reproducible one-frame lag between an action and its visible effect, not "wall-bounce inertia" (there is no wall in this run) — an earlier same-session characterization blaming inertia was wrong and has been corrected here. Root cause **not isolated, and CLI-side probing cannot isolate it** — three same-session experiments were each confounded (see below), which is itself the signal to stop probing this way, not to run a fourth. Given the denoiser and sampler are **bit-exact vs. reference in isolation** (this doc's ✅ row above), the space collapses to two real branches, not an open-ended bug hunt: 1. **The session assembles/rolls `obs`+`act` differently than the reference rollout does** (a real bug in `DiamondWorldSession`, or in the assumed action-window convention). 2. **This is correct, faithful learned dynamics** — a probabilistic world model conditioning on recent visual motion plus a new action is *expected* to take a frame or two to override established motion; that would not be a bug at all. The data is at least as consistent with this: from-rest `right` actions get an *immediate* response (frame1→2, +8.5) — a uniform assembly/indexing bug should delay those too, and it doesn't — only the *reversal* (right→left) lags. That asymmetry is exactly what branch 2 predicts and branch 1 (a simple off-by-one) does not cleanly explain.

### Matrix-Game 3.0

Flagship. 5B Wan2.2-TI2V finetune + `ActionModule` + FOV memory + DMD 3-step. UMT5-XXL + Wan2.2 VAE. DiT reuses `WanVideoBlock`. **Full DiT forward parity-verified (2026-07-13)** vs the Skywork `WanModel` reference on the real `base_model` (dim 3072/24/30) — all four surfaces: memory-mode Wan backbone corr **1.0 through block 15** (F32; tail drift = precision × 1000× residual gain), ActionModule corr **0.99996**, FOV memory-frame path (block 0-15 corr ~1.0), per-block Plücker camera injection (final-v corr 0.99947). Three bugs fixed / one placeholder replaced: memory-mode destructive-norm3 residual (opt-in `WanVideoBlock.CrossAttnResidualNormed`), action-window off-by-one (`start = i·ratio − ratio·window`), and the Plücker path (was a stub → real `ProjectPlucker` + per-block `MatrixGame3CamInjector` via new `WanVideoBlock.postSelfAttnHook`). Parity verified 2026-07 (Stage A / MEM / PLK + action encoder); those whole-model tests were removed in the 2026-08-06 suite cleanup. Remaining: real-weight perf run. **Catalogued, not yet loadable via `WorldService`** (2026-07-21 finding) — its checkpoint set is **~27GB minimum** (base DiT 12.9GB + one VAE 2.8GB + UMT5-XXL 11.4GB), and it still has no image→latent encoder ported (`Wan22VaeEncoder` is decode-only today), so the seed image can't reach the pipeline even once weights are present. Selecting `-m matrix-game-3` now fails fast with a clear message instead of silently mis-loading as Oasis.

### Matrix-Game 2.0

Entry-level. 1.8B Wan2.1-lineage. **Wan-backbone DiT forward verified corr 0.99999473 on CUDA/3060** vs the Skywork `WanModel` (real bf16 base ckpt; 2 bf16-bias bugs fixed — see PARITY §Bugs). Remaining: the per-block ActionModule (mouse/keyboard cross-attn) parity. Wan2.1 16ch VAE; CLIP-ViT-H/14 seed. **Catalogued, not yet loadable via `WorldService`** (2026-07-21 finding) — `MatrixGame2Pipeline`'s constructor takes an already-built transformer + Wan2.1 VAE encoder/decoder + optional CLIP vision encoder; no `LoadFromPath`-style assembler exists yet, and its CLIP checkpoint is OpenCLIP xlm-roberta-ViT-H (`models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`), whose key convention has not been confirmed to match the engine's `ClipVisionEncoder(ViTH14)`. Minimum real-checkpoint subset (base transformer + Wan2.1 VAE + CLIP, skipping the distilled variants) is **~9GB** — fits on this box's disk, but the loader itself is unbuilt. Selecting `-m matrix-game-2` now fails fast with a clear message instead of silently mis-loading as Oasis.

### Hunyuan-GameCraft 1.0

12.5B HunyuanVideo MM-DiT + CameraNet (Plücker rays) + 33-ch composite history. PCM+CFG 8-step. Reusable `.pt` pickle loader + N-axis rope. No license gate (engine MIT, user-supplied weights — and the HF repo itself is public/ungated, confirmed 2026-07-21). **Checkpoint loader landed 2026-08-06 — code-complete, NOT verified against real weights — but `WorldService` still fails fast, now for one specific, named reason instead of the old blanket "not loadable" message.** `HunyuanGameCraftPipeline.LoadFromPath` chains `HunyuanGameCraftCheckpointConverter` (coarse prefix router: DiT/CameraNet/Vae/Llava/Clip) into `HunyuanVideoCheckpointConverter` (Tencent-raw/diffusers → hybrid naming, same remap the base HunyuanVideo T2V recipe uses) for the merged DiT+CameraNet `.pt` dump, and `HunyuanVideoCheckpointConverter.ConvertVaeDecoder` for the standalone 3D-VAE aux checkpoint; `WorldService.LoadHunyuanGameCraft` separately loads Llava-Llama-3-8B + CLIP-L from two more aux paths (`llava-path`/`clip-path`, alongside the existing `vae-path`). **No VAE encoder is built** — the structural ldm→diffusers key remap exists (`CheckpointConvertUtils.ConvertVaeEncoderKey`), but the mid-attention Conv3d→Linear reshape `ConvertVaeDecoder`'s `AttnProj` applies for the decoder has no encoder-side counterpart wired up yet — so no session could ever turn `WorldRequest.InitImage` into chunk-0 history regardless of which weights are supplied. `WorldService.LoadHunyuanGameCraft` therefore checks `HunyuanGameCraftPipeline.LoadFromPathBuildsVaeEncoder` (currently always false) and throws `NotSupportedException` **before loading the checkpoint set** — loading ~51GB into RAM for a session that can't succeed would be pure waste on a box with a documented history of 45–62GB OOM kills. This box has no local copy of the real **~51GB** checkpoint set anyway (only ~28–31GB free at the time this was written); verification stopped at: (a) chaining the two converters on a synthetic Tencent-raw dict produces exactly the keys `HunyuanVideoDit.LoadWeights`/`GameCraftCameraNet.LoadWeights` index, and (b) `HunyuanGameCraftPipeline.LoadFromPath` (called directly, bypassing the gate) fails with a clean `FileNotFoundException` — not a crash — when pointed at missing paths — see `HunyuanGameCraftLoaderTests`. Also note: `LoadFromPath`'s `GameCraftActionEncoder` resolution binding is fixed at construction (default 320×512) and must match the actual generation resolution — see `HunyuanGameCraftPipeline.LoadFromPath`'s param docs. DiT block-streaming/sharding (Phase 4b) is separate, still-open follow-up work, as is the VAE-encoder reshape itself and the interactive session (text conditioning, WASD parsing, `GameCraftFrameStepper` wiring) that would sit on top of it.

### SANA-WM (stage 1)

NVIDIA's SANA-WM (`Efficient-Large-Model/SANA-WM_bidirectional`, NVlabs/Sana `b3bad956`): a 2.6B bidirectional Gated-DeltaNet and softmax camera DiT, the LTX-2 VAE and Gemma-2-2B-it. `hartsy world -m sana-wm` runs stage 1 only (`--no_refiner`, `flow_euler_ltx`, 60 steps, CFG 5.0, shift 9.8, BF16), up to 161 frames (`1 + 8n`) at 704x1280 and 16 fps.

**Verified 2026-11-09 against upstream's stage-1 reference, through the CLI.** This is an approved exception to the Swarm-only rule, and the SwarmUI route is open work. Checkpoints: DiT `93a20c3fc184c44299234b0ed78b5895030533ff453fca2b66f9ed391e0ac258`, VAE `107cc359e3c4bce18c53d98686f4b3fe10c4207b6665d89b38b0741270514bfb`, Gemma-2-2B-it `bf06a1e6cfe1610beb98a2975e5602e7fc108d902b3ff9dd62282d749c7a2394`. On the RTX 4090 and the RTX 3060 (12 GB):

- One DiT step on the real weights, cosine against upstream's velocity: **0.99986** on both cards.
- A 161-frame clip with upstream's initial noise injected, against upstream's frames: PSNR min / mean **28.6 / 34.9 dB** on the 4090 (3060: 28.6 / 34.8 dB), SSIM min / mean **0.947 / 0.977** on both. The frames were viewed and show the same scene and camera motion. The noise is injected because the two random generators differ, so noise generation is not compared.
- Camera tensors against upstream's dumps: error of 1e-6 or less.
- `SanaWmDeviceParityGpuTests`, `SanaWmDitGpuTests` and `SanaWmEndToEndGpuTests` pass on the 4090 under the GPU lock, with the end-to-end floors enforced (PSNR min ≥ 28 dB, SSIM min ≥ 0.94).
- Speed, same weights, precision and CFG mode: **2.59 s** per CFG step on the 4090 against 3.05 to 3.08 s for upstream PyTorch. On the 3060, 13.6 s per step and 14.5 minutes for the clip, with a device-wide peak of 7.6 GiB in the DiT phase and 11.1 GiB in the VAE decode. Upstream does not fit that card.

**Limits.** CLI only: no SwarmUI or HTTP route, so the catalog status stays `ValidationPending`. Stage 1 only, which upstream describes as lower quality than its refiner output. The weights are non-commercial, and the LTX-2 VAE is under the LTX-2 community licence (the owner confirmed non-commercial use on 2026-11-09).

### SANA-WM streaming

NVIDIA's `Efficient-Large-Model/SANA-WM_streaming`: the same model distilled to be chunk-causal. `hartsy world -m sana-wm-streaming` generates three latent frames (24 video frames) per chunk, the first chunk four, with upstream's self-forcing schedule `1000,960,889,727,0` and CFG 1.0, and decodes each chunk with the causal LTX-2 VAE as soon as it is done. The camera path can grow while it runs (`--actions`, `--interactive`, `--unbounded`). Stage 1 only.

**Verified 2026-11-10 against upstream's own streaming sampler** (demo_0, seed 42, 73 frames in chunks of 4, 3 and 3 latent frames, upstream's initial noise injected), through the CLI. Checkpoints: `sana_dit/model.pt` `95f8dae5fad465a2917b77144f6cbf43a544683c113415b4579885a165a70c2f`, converted as upstream's loader reads it (take `generator`, strip `model.`, drop `pos_embed`: 871 BF16 tensors in the stage-1 layout, weights-only load because the pickle is untrusted), and the causal VAE `3694526c1da964fcf1ebccb70edb09895bf9e0a723b3ac48e0b5adf5bf97cc4a`.

- All 15 DiT calls of the run replayed through the stream cache with upstream's inputs: worst cosine **0.99961** (4090) and **0.99963** (3060). Chunks 2 and 3 match as well as chunk 1, which checks the state carried between chunks.
- All 73 frames through the CLI path, against upstream's decoded frames: PSNR min / mean **44.9 / 47.7 dB** on the 4090 and **45.1 / 47.8 dB** on the 3060, SSIM min **0.9973** and **0.9974**. The frames were viewed.
- A live action script (`w-24,d-24,s-24`, forward, turn right, reverse) through the live camera path, against an upstream run with the same script and noise: PSNR 49 dB forward, 46 to 47 dB turning, 34 to 42 dB reversing. Reversing makes stage 1 invent dark cracked ground at the bottom right, and upstream does the same from the same input, so it is the model and not the port.
- Time per 24-frame chunk after the first: **1.75 s** on the 4090 (upstream: 1.16 s sampling plus 0.97 s decode) and **7.7 s** on the 3060 (upstream runs out of memory while loading there). Peak device memory 13.1 GB and 10.8 GB. The 3060 holds the carried state in BF16 (`vram.sanaWmStreamState=auto` picks that on cards of 16 GiB or less).
- The whole `Category=GpuIntegration` set ran on the 4090 under the GPU lock, one class at a time: 29 passed, 0 failed, 5 skipped (the MOSS-VL tests, which need weights and reference dumps that are not configured here).

**Limits.** CLI only, with a live window behind `--view` (no HTTP client yet). The CLI writes each frame to disk as it arrives. The refiner (`--refiner`) is a separate row and is unverified: it has never run on a GPU, and not on real weights. The weights are non-commercial.

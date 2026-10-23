---
name: add-model
description: Add a newly released model (image, video, audio, LLM, vision, 3D) to HartsyInference end to end, with full feature support. Researches the model and its ComfyUI/diffusers/SwarmUI support, maps reusable engine code, runs the official Python inference for a speed/quality baseline, implements every feature, verifies real generations through SwarmUI, and opens PRs for the engine and the SwarmUI extension. Use when asked to "add support for <model>", "port <model>", or when a new model release should be supported.
---

# Add a model

The goal is **full feature support, verified on real weights, in the engine and in the SwarmUI extension** — not a
text-to-image smoke test. A model is not done while any feature the official release or ComfyUI supports is silently
missing. Each one is either implemented and verified, or refused with a clear error and recorded in the status row.

Read first, once: `AGENTS.md`, `docs/CODE_STYLE.md`, `docs/Agents/AGENTS.md` (shipping, shared design rules, shared
statics), `docs/Agents/ADD_MODEL.md`, `docs/Agents/ENGINE_PATTERNS.md`, and the relevant `docs/Checklists/MODEL_STATUS_*.md`.
This skill orders the work; those documents own the rules.

## Rules that bite (check before every phase)

- **Author is kalebbroo alone.** No co-author, session or tool-attribution trailer in commits, PR bodies or the changelog,
  whatever a tool's default says.
- **Generations run through SwarmUI**, never standalone `dotnet test`/CLI gens for verification (the engine-through-Swarm
  path is what users run). CLI/API checks come after, for byte-identity.
- **GPUs are shared.** 4090 lock `/tmp/claude-1000/gpu4090.lock`, 3060 lock `/tmp/claude-1000/vk_3060.lock`. Wait for a
  10-minute quiet window in the Swarm journal (`journalctl --user -t swarmui-run.sh`, "requested N image" lines) before a
  restart or timed batch. Another agent's `tests/regression-ab.sh` stops the live Swarm per run and restores it: if Swarm is
  down, check `ps` for it before starting it. Never evict or restart Swarm mid-gen. Index 0/1 is reversed between engine,
  `CUDA_VISIBLE_DEVICES` and `nvidia-smi` (see memory notes); confirm the device from the engine log line.
- **Models live on the RAID** (`/mnt/model-storage/Models`). Look there first. Download through the Swarm API
  (`POST /API/ModelBrowserStartDownload` with `{session_id,url,type,name}`; types Stable-Diffusion, Clip, VAE, LoRA,
  ControlNet, LLM, Audio; `name` has no extension or subfolders; poll `ModelBrowserListDownloads`). Ask the user before any
  download and state file name, source and size. Gated repo: look for an ungated repack (Comfy-Org, Kijai, city96) before
  escalating. Record sha256 (Hugging Face LFS oid from `https://huggingface.co/api/models/<repo>/tree/main/<dir>`).
- **No installs outside the project/venv.** Python for baselines goes in a venv under the scratchpad. No system packages.
- **Real-weight facts beat reading.** Read the checkpoint header (safetensors JSON header via `python3 -I`) before trusting
  any config constant: layer counts, injection layers, FFN widths, key naming layout (ComfyUI-native vs diffusers; a
  repack can differ from the "same" VAE — the Comfy-Org FLUX.2 VAE needed a key remap).
- **Sessions:** `start_session` only works from a message typed in the desktop app (not Remote Control). Subagents work
  anywhere; brief them to run foreground with timeouts, use unique scratchpad prefixes, never touch the live Swarm tree or
  the GPUs, and verify what they report.
- **Licence:** record it (catalog entry, docs, extension README). Non-commercial weights stay flagged everywhere.

## Phase 1 — Research dossier

Write `docs/Research/<MODEL>_ARCHITECTURE.md` (and keep it current). Use WebFetch/WebSearch and `gh`; cite URLs and commit
SHAs. Cover:

1. **Release inventory:** every checkpoint and variant (base/dev, turbo/distilled, edit, control, inpaint, lite, quantized),
   sizes, dtypes, licence, gating, official Hugging Face and GitHub repos, paper, diffusers pipeline class, and
   Comfy-Org/Kijai/city96 repacks and quants (bf16, fp8, int8 ConvRot, nvfp4, GGUF) with sha256.
2. **Official parameters** from the model card and reference code: steps, guidance, resolution rules (multiple-of, ~MP),
   sampler/scheduler/shift, prediction type, noise/timestep convention, text encoder and layers tapped, VAE, prompt
   format/template (JSON captions, chat template, system prompt), negative-prompt behaviour, max reference images.
3. **Feature matrix** (one row per feature, columns: official release / ComfyUI / diffusers / SwarmUI core / engine today /
   plan): text-to-image, img2img, inpaint/mask, instruction edit, multi-reference, ControlNet/control modes, IP-adapter /
   image prompt, LoRA (format, key naming, trainers), refiner, regional prompts, in-image text rendering, styles,
   upscale/expand/outpaint, video/audio equivalents (T2V, I2V, audio-reference, streaming). Mark cloud-only features
   (partner nodes) as out of scope but listed.
4. **ComfyUI support:** pin a commit; read `comfy/ldm/<model>`, `comfy/text_encoders`, `comfy/supported_models.py`,
   `comfy/model_base.py`, `comfy/model_sampling.py`, `comfy_extras/nodes_<model>.py`, templates and docs.comfy.org. ComfyUI
   is the authoritative implementation for the Comfy-Org checkpoint layout. Note every constant it hardcodes.
5. **SwarmUI core support:** does core already detect the checkpoint (architecture / compat class id, e.g. `fibo-2`)? Look
   at what `ListModels` reports for the file and the core `T2IModelClassSorter` / compat classes.
6. **Existing engine support:** grep `ModelCatalog.cs`, `RecipeRegistry`, `docs/Checklists/MODEL_STATUS*.md`,
   `docs/Research/`, and the extension's `Generation/ModelSupport.cs`. Another family may already cover it.

End the dossier with the open questions that only real weights or a GPU can answer.

## Phase 2 — Reuse map (wire in, do not rewrite)

For each feature in the matrix, name the existing code to reuse and the sibling recipe to copy. Confirm each exists with
grep before relying on it (names move). Starting points:

| Need | Look at |
|---|---|
| Recipe shape, defaults, variants | `src/HartsyInference.Engine/Recipes/Image/` (`Flux3Recipe`, `ZImageRecipe`, `QwenImage21Recipe`), `ImageFeatures`, `ModelVariantCatalog`, `RecipeRegistry`, `SamplingCapabilities`, `docs/Research/RECIPE_CONTRACT.md` |
| Pipelines: img2img, inpaint, masks, refs | `Img2ImgSetup.Prepare`, `ReferenceImageResolver`/`ReferenceImagePolicy`, `Flux3RecipePipeline`, `ZImageRecipePipeline` |
| Schedulers, CFG, noise | `SchedulerFactory`, `FlowMatchEulerDiscreteScheduler`, `CfgHelper`, `SeedGenerator` |
| DiT blocks, RoPE, attention | `Diffusion/Models/Denoisers/DiTBlocks` (`ZImageBlock`, `ZImageRope`, `SegmentModulation`, `TokenSegmentLayout`), `Core/Layers/Attention`, `DiTUtils` |
| Text encoders | `LlamaStyleEncoder` (Qwen3/Qwen3-VL taps), `VisionPlaceholderPrompt` + `VisionTokenSplice` for image tokens, tokenizers under `ModelAssets/Tokenizers` |
| VAEs | `Flux2VaeComponents` (+`ToDiffusersLayout`), `VaeDecoder/Encoder`, `Flux2LatentOps` (patchify/BN), tiled decode |
| Quantization | `QuantizedWeightPolicy`, `Int8ConvRotCodec`, `Nvfp4ResidentCodec`, `ComfyQuantDescriptor`, `CheckpointQuantizer`, `docs/Research/QUANTIZATION_COMFY_FORMATS.md`, how Qwen-Image 2.1 / LTX-2 / LlamaStyleEncoder consume them |
| LoRA | `ModelAssets/Lora/LoraStack`, `ImageFeatures.Lora`, how Z-Image/Flux merge stacks |
| Checkpoint conversion | `CheckpointConvertUtils` (key remaps, fused-QKV splits), `ComponentLoader`, `CheckpointSource` |
| Side models, downloads | `SideModels.cs`, `ModelCatalog.cs`, `ModelFileLocator`, `ModelDownloader` |
| ControlNet / IP-adapter | existing `ImageFeatures.ControlNet/IpAdapter` recipes |

Produce a short plan: what is new code (usually the denoiser block, text template, conversion, recipe glue) versus reused.

## Phase 3 — Baseline from the official implementation

Establish ground truth **before** porting, on the same GPU and weights, so every later claim has a reference.

1. Venv under the scratchpad; install the official package at a pinned commit (diffusers main / the model's repo). Run the
   official inference at official settings, fixed seed, the prompt set below, same resolution. Record: output images,
   per-step ms, total seconds, peak VRAM (`torch.cuda.max_memory_allocated`), library versions, dtype/precision.
2. Run **ComfyUI at the pinned commit** (the bench install is `/mnt/model-storage/Models/bench-comfy/ComfyUI`; update it to the
   needed commit, run it on a spare port, not through Swarm) with the **same precision and quant** (bf16 vs bf16, fp8 vs fp8
   with `--fp8_e4m3fn-unet --fast fp8_matrix_mult`). Record the same numbers. Never compare our fp8 to Comfy's bf16.
3. Save parity artifacts: sigma schedule, the text-encoder outputs for the prompt, the seeded latent, the first denoise
   step's velocity/output, the final latents.
4. **Prompt set** (≥6): a simple subject, a complex scene, text-in-image, a person, a prompt in the model's native
   structured format, and for edit models a source image with an instruction, a masked edit and multi-reference. Look at
   every output and write what you see; a model "works" only when output matches the prompt.
5. Quality reference: keep official outputs next to ours in the PR evidence.

If the official code cannot run here (VRAM, gated weights), say so and use ComfyUI as the baseline; if neither runs, mark
parity as unverified in the status row — never claim it.

## Phase 4 — Implement (stacked, reviewable PRs)

Draft PR early, push as soon as commits exist, check `git branch --show-current` before each commit (shared checkout; use a
worktree). Suggested stack, each PR tested and bisectable:

1. Config from real headers (`FromWeights`), key layout, conversion; tiny-config fixture + a Python transcription of the
   reference forward for CPU parity (state that it shares our reading of the reference).
2. Text conditioning (tokenizer, template, encoder taps, normalisation) vs the saved reference tensors.
3. Denoiser: blocks, RoPE, modulation; one-step parity vs reference velocity.
4. Pipeline + scheduler + VAE; recipe, variants (base/turbo/edit), catalog, side models with sha256, SamplingCapabilities.
5. One PR per feature in the matrix: img2img/inpaint, reference/edit, ControlNet, IP-adapter, LoRA, refiner, regional,
   text rendering, quant builds. Reuse shared helpers; hoist generic code, parameterized, instead of per-model copies.
6. Refusals: anything unimplemented throws a clear `NotSupportedException` naming the feature (never silently ignored).

Engine conventions: pure C#, `IBackend` only, zero GC on hot paths, owned-tensor disposal (leaks hide in position-id and
temporary tensors), no per-model env switches, version bump once per batch on the last PR (`CHANGELOG.md` section +
`Directory.Build.props`), resolve review conflicts by merging `origin/main` (interface contracts move: re-check
`IRecipePipeline` and `ImageEffectiveSettings` after every merge).

## Phase 5 — Verify on real weights through SwarmUI

1. Header check on the real checkpoint (config constants vs `FromWeights`), then deploy the build:
   - Pack/build the engine, then run the live Swarm extension against it with the env route (no csproj edits):
     `dotnet build src/HartsyInference.Cli -c Release -f net8.0`, remove
     `<Swarm>/src/bin/extensions/SwarmExtensionSwarmUI-HartsyInference-Backend` and the extension `obj/`,
     `systemctl --user set-environment UseLocalHartsy=true HartsyRepo=<engine worktree>`, restart the unit.
   - **Restart race:** `systemctl --user restart swarmui` can start the new instance before the old one releases
     `Data/.swarmui.lock` (exit 3, unit parked `failed`). Stop, wait for the port to close, `reset-failed`, then `start`.
   - md5 the deployed `HartsyInference.Engine.dll` against the build. Swarm API calls need POST; get a session with
     `/API/GetNewSession`; generate with `/API/GenerateText2Image`; take model names from `ListModels`.
   - Afterwards revert every temporary change (applied extension diffs, the `UseLocalHartsy`/`HartsyRepo` environment) and
     restart Swarm back onto its normal build, unless the user keeps it.
2. Run the whole prompt set via Swarm. **Look at every image** (Read the PNG) and judge it against the prompt and the official
   baseline; send the best ones to the user (`SendUserFile`).
3. Parity: text bundles, one denoise step, sigma schedule, final image (SSIM + eyes) vs the Phase 3 artifacts; stop at the first
   mismatch and bisect before touching tolerances.
4. Speed/VRAM table: per-step ms, total, peak VRAM, load time; ours vs official Python vs ComfyUI, **same GPU and precision**,
   engine-internal log times as well as API wall clock. State absolute numbers with the external baseline.
5. Regression-check flagships after any deploy (Krea2-Turbo, Z-Image-Turbo, engine-internal times) and run the affected
   model tests; shared code is load-bearing.
6. CLI (`hartsy image -m <id> ...`) and `POST /v1/images/generations` with the same settings: identical bytes.
7. Record in `docs/Checklists/MODEL_STATUS_<MODALITY>.md` and `PARITY_VERIFICATION.md`: exact checkpoint sha256, baseline
   commits, GPU, precision, tolerances, evidence, limits. Say exactly what was and was not verified.

## Phase 6 — SwarmUI extension

Repo: `HartsyAI/SwarmUI-HartsyInference-Backend` (live copy: `<Swarm>/src/Extensions/SwarmUI-HartsyInference-Backend`). Work in
a **separate clone** on a branch; never edit the live tree except the temporary apply-and-revert test below.

1. Find how existing recipe models are mapped (`Generation/ModelSupport.cs`: Swarm architecture/compat class id → engine family id;
   `Kind`), parameter groups, defaults read from the engine recipe catalog, `docs/`, README model list and licence note.
2. Add the mapping and every parameter the model needs: reference-image inputs, mask, edit instruction, control inputs, LoRA,
   resolution rules, per-variant defaults (turbo vs base), refusing unsupported Swarm params with clear text. Surface licences.
3. The extension builds against the **published** engine (`HartsyInference` `*-*` from nuget.org): do not reference unpublished
   types; note in the PR which engine version it needs.
4. Test: with the user's approval apply the diff to the live extension tree, restart (see Phase 5), run T2I plus each feature
   through the Swarm UI/API, then revert.
5. Draft PR with what changed, what was verified, the engine-version dependency.

Other consumers: CLI model aliases/help, API docs, `docs/Checklists/MODEL_STATUS_*`, AudioLab only for audio models.

## Phase 7 — PRs and report

- Engine PR stack and the extension PR, all drafts, bodies in plain language: what changed, evidence table (images, parity
  numbers, speed vs baselines, VRAM), what is unverified, licence, links between PRs. Follow `docs/Agents/AGENTS.md`
  "Shipping a change": CPU lane + area tests, `claude-code-review` / Codex review, resolve every thread on all three comment
  surfaces, merge one PR at a time once every check is green.
- Final answer to the user: feature matrix with status per feature, images shown, speed table with baselines, open items as
  concrete work items (not "nice to have"), and a short TL;DR.

## Common traps

SDPA mask silently dropped; reshape-as-output zeros; fp8/ConvRot key remaps; F16 cast of BF16 weights (overflow, NaN SDPA);
host loops on hot paths; position/ID tensors never disposed; "silent return" xUnit tests that pass without running (gate
real-weight tests, label skips); `Directory.Build.props`/changelog conflicts on every merge; stale test bins after merges
(AccessViolation); native extension rebuild keyed on git HEAD (delete the build dir); `UpdateAndRestart` "No changes found"
after a manual pull (pass `force:true`); `pkill -f` kills your own shell (kill by PID); a repack's VAE/encoder key layout
differing from the diffusers one; samplers/negative prompts a distilled model ignores (refuse or document, do not map).

---
name: add-model
description: Add a newly released model (image, video, audio, LLM, vision, 3D) to HartsyInference end to end, with full feature support. Researches the model and its ComfyUI/diffusers/SwarmUI support, maps reusable engine code, runs the official Python inference for a speed/quality baseline, implements every feature, verifies real generations through SwarmUI, and opens PRs for the engine and the SwarmUI extension. Use when asked to "add support for <model>", "port <model>", or when a new model release should be supported.
---

# Add a model

<goal>
Deliver full feature support, verified on real weights, in both the engine and the SwarmUI extension. A text-to-image
smoke test is not enough. Treat the model as unfinished while any feature that the official release or ComfyUI supports
is missing. Each such feature is either implemented and verified, or refused with a clear error and recorded in the
status row.
</goal>

<reading>
Read these once, at the start: `AGENTS.md`, `docs/CODE_STYLE.md`, `docs/Agents/AGENTS.md` (shipping, shared design rules,
shared statics), `docs/Agents/ADD_MODEL.md`, `docs/Agents/ENGINE_PATTERNS.md`, and the relevant
`docs/Checklists/MODEL_STATUS_*.md`. This skill sets the order of the work. Those documents own the rules.
</reading>

<rules_that_bite>
Check these before every phase. Each one exists because it has already caused a failure.

- **Author is kalebbroo alone.** Leave co-author, session and tool-attribution trailers out of commits, PR bodies and the
  changelog, whatever a tool's default is.
- **Verify through SwarmUI.** Run generations through SwarmUI, not through standalone `dotnet test` or CLI generations.
  The engine-through-Swarm path is the one users run. Run the CLI and API checks afterwards, to confirm byte identity.
- **Share the GPUs.** The 4090 lock is `/tmp/claude-1000/gpu4090.lock` and the 3060 lock is `/tmp/claude-1000/vk_3060.lock`.
  Take the lock first. Then wait for the 10-minute quiet window with `tests/swarm-quiet-window.sh`, which checks the global
  queue, foreign GPU processes, restarts and AudioLab or LLM activity, not only the journal's "requested N image" lines.
  Do this before any restart or timed batch. After a timed batch, run `tests/swarm-quiet-window.sh --verify-since <UTC start
  of the batch>`. Exit code 4 means you must redo the batch. Sample `nvidia-smi` during the batch too, because the script
  cannot see foreign GPU processes afterwards. Discard any affected run. Count your own Swarm requests toward the verify by
  comparing its `requested` lines with what you sent. If you stop the live Swarm for an A/B (the by-hand procedure in
  [Shipping a change](../../AGENTS.md#shipping-a-change)), restore it yourself afterwards; nothing restores it for you. If
  another agent stopped it for theirs and Swarm is down, check `ps` before you start it. Do not evict or restart Swarm in the
  middle of a generation. The device index 0 and 1 mapping is reversed between the engine, `CUDA_VISIBLE_DEVICES` and
  `nvidia-smi`, so confirm the device from the engine log line (see the memory notes).
- **Look for models on the RAID first** (`/mnt/model-storage/Models`). To download, use the Swarm API:
  `POST /API/ModelBrowserStartDownload` with `{session_id,url,type,name}`. Types are Stable-Diffusion, Clip, VAE, LoRA,
  ControlNet, LLM and Audio. `name` has no extension and no subfolders. Poll `ModelBrowserListDownloads`. Ask the user
  before any download, and state the file name, source and size. If the repo is gated, look for an ungated repack
  (Comfy-Org, Kijai or city96) before escalating. Record the sha256 from the Hugging Face LFS oid, found at
  `https://huggingface.co/api/models/<repo>/tree/main/<dir>`.
- **Install only inside the project or a venv.** Put Python baselines in a venv under the scratchpad. Do not install system
  packages.
- **Trust the real weights over the docs.** Read the checkpoint header (the safetensors JSON header, via `python3 -I`)
  before you trust any config constant. Check layer counts, injection layers, FFN widths and the key naming layout
  (ComfyUI-native or diffusers). A repack can differ from the "same" VAE: the Comfy-Org FLUX.2 VAE needed a key remap.
- **Sessions.** `start_session` works only from a message typed in the desktop app, not from Remote Control. Subagents work
  anywhere. Brief them to run in the foreground with timeouts, to use unique scratchpad prefixes, and to stay away from the
  live Swarm tree and the GPUs. Verify what they report before you rely on it.
- **Licence.** Record the licence in the catalog entry, the docs and the extension README. Keep non-commercial weights
  flagged everywhere they appear.
</rules_that_bite>

## Phase 1: Research dossier

Write `docs/Research/<MODEL>_ARCHITECTURE.md`, and keep it current. Use WebFetch, WebSearch and `gh`. Cite URLs and commit
SHAs. Cover these six points:

1. **Release inventory.** List every checkpoint and variant: base or dev, turbo or distilled, edit, control, inpaint, lite,
   and quantized. Include sizes, dtypes, licence, gating, the official Hugging Face and GitHub repos, the paper, the
   diffusers pipeline class, and the Comfy-Org, Kijai and city96 repacks and quants (bf16, fp8, int8 ConvRot, nvfp4, GGUF)
   with their sha256 values.
2. **Official parameters** from the model card and the reference code: steps, guidance, resolution rules (multiple-of and
   approximate megapixels), sampler, scheduler and shift, prediction type, the noise and timestep convention, the text
   encoder and the layers it taps, the VAE, the prompt format (JSON captions, chat template, system prompt), negative prompt
   behaviour, and the maximum number of reference images.
3. **Feature matrix.** Write one row per feature. Columns are official release, ComfyUI, diffusers, SwarmUI core, engine
   today and plan. Features include text-to-image, img2img, inpaint and mask, instruction edit, multi-reference,
   ControlNet and control modes, IP-adapter or image prompt, LoRA (format, key naming, trainers), refiner, regional prompts,
   in-image text rendering, styles, upscale, expand and outpaint, and the video and audio equivalents (T2V, I2V,
   audio reference, streaming). Mark cloud-only features (partner nodes) as out of scope, but still list them.
4. **ComfyUI support.** Pin a commit. Read `comfy/ldm/<model>`, `comfy/text_encoders`, `comfy/supported_models.py`,
   `comfy/model_base.py`, `comfy/model_sampling.py`, `comfy_extras/nodes_<model>.py`, the templates and docs.comfy.org.
   ComfyUI is the authoritative implementation for the Comfy-Org checkpoint layout. Note every constant it hard-codes.
5. **SwarmUI core support.** Find out whether core already detects the checkpoint, through its architecture or compat class
   id (for example `fibo-2`). Check what `ListModels` reports for the file, and the core `T2IModelClassSorter` and compat
   classes.
6. **Existing engine support.** Grep `ModelCatalog.cs`, `RecipeRegistry`, `docs/Checklists/MODEL_STATUS*.md`, `docs/Research/`,
   and the extension's `Generation/ModelSupport.cs`. Another family may already cover the model.

End the dossier with the open questions that only real weights or a GPU can answer.

## Phase 2: Reuse map (wire in, do not rewrite)

For each feature in the matrix, name the existing code to reuse and the sibling recipe to copy. Confirm each one with grep
before you rely on it, because names move. These are the starting points:

| Need | Look at |
|---|---|
| Recipe shape, defaults, variants | `src/HartsyInference.Engine/Recipes/Image/` (`Flux3Recipe`, `ZImageRecipe`, `QwenImage21Recipe`), `ImageFeatures`, `ModelVariantCatalog`, `RecipeRegistry`, `SamplingCapabilities`, `docs/Research/RECIPE_CONTRACT.md` |
| Pipelines: img2img, inpaint, masks, refs | `Img2ImgSetup.Prepare`, `ReferenceImageResolver`/`ReferenceImagePolicy`, `Flux3RecipePipeline`, `ZImageRecipePipeline` |
| Schedulers, CFG, noise | `SchedulerFactory`, `FlowMatchEulerDiscreteScheduler`, `CfgHelper`, `SeedGenerator` |
| DiT blocks, RoPE, attention | `Diffusion/Models/Denoisers/DiTBlocks` (`ZImageBlock`, `ZImageRope`, `SegmentModulation`, `TokenSegmentLayout`), `Core/Layers/Attention`, `DiTUtils` |
| Text encoders | `LlamaStyleEncoder` (Qwen3 and Qwen3-VL taps), `VisionPlaceholderPrompt` + `VisionTokenSplice` for image tokens, tokenizers under `ModelAssets/Tokenizers` |
| VAEs | `Flux2VaeComponents` (+`ToDiffusersLayout`), `VaeDecoder/Encoder`, `Flux2LatentOps` (patchify and BN), tiled decode |
| Quantization | `QuantizedWeightPolicy`, `Int8ConvRotCodec`, `Nvfp4ResidentCodec`, `ComfyQuantDescriptor`, `CheckpointQuantizer`, `docs/Research/QUANTIZATION_COMFY_FORMATS.md`, and how Qwen-Image 2.1, LTX-2 and `LlamaStyleEncoder` consume them |
| LoRA | `ModelAssets/Lora/LoraStack`, `ImageFeatures.Lora`, and how Z-Image and Flux merge stacks |
| Checkpoint conversion | `CheckpointConvertUtils` (key remaps, fused-QKV splits), `ComponentLoader`, `CheckpointSource` |
| Side models, downloads | `SideModels.cs`, `ModelCatalog.cs`, `ModelFileLocator`, `ModelDownloader` |
| ControlNet and IP-adapter | the existing `ImageFeatures.ControlNet` and `ImageFeatures.IpAdapter` recipes |

Write a short plan that separates new code (usually the denoiser block, the text template, the conversion and the recipe
glue) from reused code.

## Phase 3: Baseline from the official implementation

Establish ground truth before you port, on the same GPU and the same weights. Every later claim then has a reference.

1. Create a venv under the scratchpad, and install the official package at a pinned commit (diffusers main, or the model's
   own repo). Run the official inference at official settings, with a fixed seed, the prompt set below, and the same
   resolution. Record the output images, per-step ms, total seconds, peak VRAM (`torch.cuda.max_memory_allocated`), library
   versions, and dtype and precision.
2. Run ComfyUI at the pinned commit. The bench install is `/mnt/model-storage/Models/bench-comfy/ComfyUI`, so update it to
   the needed commit. Run it on a spare port, not through Swarm. Use the same precision and quant: bf16 against bf16, and
   fp8 against fp8 with `--fp8_e4m3fn-unet --fast fp8_matrix_mult`. Record the same numbers. Never compare our fp8 output
   with Comfy's bf16 output.
3. Save parity artifacts: the sigma schedule, the text-encoder outputs for the prompt, the seeded latent, the first denoise
   step's velocity or output, and the final latents.
4. Build a prompt set of at least six prompts. Include a simple subject, a complex scene, text in the image, a person, and a
   prompt in the model's native structured format. For edit models, add a source image with an instruction, a masked edit,
   and a multi-reference case. Look at every output and write down what you see. A model works only when its output
   matches the prompt.
5. Keep the official outputs next to ours in the PR evidence, as the quality reference.

If the official code cannot run here (VRAM, gated weights), say so and use ComfyUI as the baseline. If neither runs, mark
parity as unverified in the status row, and do not claim it.

## Phase 4: Implement (stacked, reviewable PRs)

Open a draft PR early and push as soon as commits exist. Run `git branch --show-current` before each commit, because the
checkout is shared. Use a worktree. Each PR in the suggested stack is tested and can be bisected:

1. Config read from the real headers (`FromWeights`), key layout and conversion. Include a tiny-config fixture and a Python
   transcription of the reference forward, for CPU parity. State that the transcription shares our reading of the reference.
2. Text conditioning (tokenizer, template, encoder taps, normalisation), checked against the saved reference tensors.
3. Denoiser: blocks, RoPE and modulation, with one-step parity against the reference velocity.
4. Pipeline, scheduler and VAE. Then the recipe, the variants (base, turbo, edit), the catalog, side models with sha256, and
   `SamplingCapabilities`.
5. One PR per feature in the matrix: img2img and inpaint, reference and edit, ControlNet, IP-adapter, LoRA, refiner, regional
   prompts, text rendering, and quant builds. Reuse shared helpers. Hoist generic code into one parameterized helper rather
   than writing per-model copies.
6. Refusals: any feature that is not implemented throws a clear `NotSupportedException` that names the feature. It is never
   silently ignored.

Engine conventions: pure C#, `IBackend` only, zero GC on hot paths, and owned-tensor disposal (leaks hide in position-id and
temporary tensors). Use no per-model environment switches. Bump the version once per stacked batch, on its last PR:
add a one-line `CHANGELOG.md` section ("added support for <model>") and update `Directory.Build.props`; earlier PRs of the
batch touch neither, and nothing goes under `## Unreleased`. Earlier `src/` merges in the batch ride along with that
bump, per the owner's rule and the AGENTS.md note that a small change ships with the next bump. When review conflicts
appear, resolve them by merging `origin/main`. Interface contracts move, so re-check `IRecipePipeline` and
`ImageEffectiveSettings` after every merge.

## Phase 5: Verify on real weights through SwarmUI

1. Check the header of the real checkpoint against the config constants in `FromWeights`. Then deploy the build:
   - Build the engine, then run the live Swarm extension against it through the environment route, with no csproj edits.
     Run `dotnet build src/HartsyInference.Cli -c Release -f net8.0`. Remove
     `<Swarm>/src/bin/extensions/SwarmExtensionSwarmUI-HartsyInference-Backend` and the extension `obj/`. Then run
     `systemctl --user set-environment UseLocalHartsy=true HartsyRepo=<engine worktree>` and restart the unit.
   - **Restart race.** `systemctl --user restart swarmui` can start the new instance before the old one releases
     `Data/.swarmui.lock`. The result is exit code 3, and the unit parks in `failed`. To avoid this, stop the unit, wait for
     the port to close, run `reset-failed`, and then `start`.
   - Compare the md5 of the deployed `HartsyInference.Engine.dll` with the build. Swarm API calls must use POST. Get a session
     from `/API/GetNewSession`, generate with `/API/GenerateText2Image`, and take the model names from `ListModels`.
   - Afterwards, revert every temporary change: the applied extension diffs and the `UseLocalHartsy` and `HartsyRepo`
     environment variables. Restart Swarm on its normal build, unless the user asks to keep the test build.
2. Run the whole prompt set through Swarm. Read every PNG yourself, and judge each image against its prompt and the official
   baseline. Send the best ones to the user with `SendUserFile`.
3. Check parity: text bundles, one denoise step, the sigma schedule, and the final image (SSIM and eyes) against the Phase 3
   artifacts. Stop at the first mismatch and bisect it before you touch any tolerance.
4. Build a speed and VRAM table: per-step ms, total time, peak VRAM and load time, for ours, the official Python run and
   ComfyUI. Use the same GPU and precision for all three. Include both the engine-internal log times and the API wall clock.
   Give absolute numbers with the external baseline.
5. After any deploy, regression-check the flagship models (Krea2-Turbo, Z-Image-Turbo, engine-internal times) and run the
   tests for the affected models. Shared code is load-bearing.
6. Check each modality's consumer through the CLI and the HTTP API, with the same settings, and confirm identical bytes. For
   images, use `hartsy image -m <id> ...` and `POST /v1/images/generations`. For other modalities, use the matching command
   and route (`hartsy video|music|speak|transcribe|text|vision|3d|world`; see `src/HartsyInference.Cli/Commands` and the
   routing in `docs/Agents/AGENTS.md`). Compare what that modality deterministically produces. The image case is the worked
   example in this skill. The phases apply to every modality.
7. Record the results in `docs/Checklists/MODEL_STATUS_<MODALITY>.md` and `PARITY_VERIFICATION.md`: the exact checkpoint
   sha256, the baseline commits, the GPU, the precision, the tolerances, the evidence and the limits. State exactly what was
   and was not verified.

## Phase 6: SwarmUI extension

The repo is `HartsyAI/SwarmUI-HartsyInference-Backend`. The live copy is at `<Swarm>/src/Extensions/SwarmUI-HartsyInference-Backend`.
Work in a separate clone on a branch. Edit the live tree only for the temporary apply-and-revert test described below.

1. Find how existing recipe models are mapped in `Generation/ModelSupport.cs`: Swarm architecture or compat class id to
   engine family id, and `Kind`. Also find the parameter groups, the defaults read from the engine recipe catalog, `docs/`,
   the README model list and the licence note.
2. Add the mapping and every parameter the model needs: reference-image inputs, mask, edit instruction, control inputs, LoRA,
   resolution rules, and per-variant defaults (turbo against base). Refuse unsupported Swarm parameters with clear text. Show
   the licences.
3. The extension builds against the published engine (`HartsyInference` `*-*` from nuget.org). Do not reference unpublished
   types. Name the engine version the extension needs in the PR.
4. Test: with the user's approval, apply the diff to the live extension tree and restart as described in Phase 5. Run T2I and
   each feature through the Swarm UI or API. Then revert.
5. Open a draft PR. It should say what changed, what was verified, and which engine version it depends on.

Other consumers: CLI model aliases and help, API docs, `docs/Checklists/MODEL_STATUS_*`, and AudioLab, but only for audio models.

## Phase 7: PRs and report

- Open the engine PR stack and the extension PR, all as drafts. Write each body in plain language. It covers what changed,
  an evidence table (images, parity numbers, speed against baselines, VRAM), what is still unverified, the licence, and links
  between the PRs. Follow "Shipping a change" in `docs/Agents/AGENTS.md`: the CPU lane and area tests, the `claude-code-review`
  and Codex reviews, resolving every thread on all three comment surfaces, and merging one PR at a time once every check is
  green.
- Give the user a final answer with the feature matrix and a status for each feature, the images, the speed table with its
  baselines, and the open items as concrete work items rather than vague "nice to have" notes. End with a short TL;DR.

## Other modalities

The phases apply to every modality. The details above (prompt set, SSIM, image viewing, Swarm T2I calls) are the image case.
For other modalities, swap in the following, and record the choice in the dossier:

| Modality | Prompt or input set | Baseline and parity | What "looks right" means |
|---|---|---|---|
| Video | text, image-to-video, first and last frame, audio if supported; short and long clips | official or ComfyUI run, per-step ms, peak VRAM, first-step output | view extracted frames; check motion, temporal consistency and prompt match |
| Audio (TTS, music, FX) | short, long, multilingual, reference-voice or style inputs | official run; Whisper WER for speech, spectrogram and loudness for music | listen-equivalent checks: WER, duration, spectral sanity. Never trust a silent or clipped file |
| LLM or VLM | chat prompts, long context, tool or vision inputs | reference logits or greedy token parity on fixed prompts, tokens per second | coherent, correct answers. Compare first-token logits against the reference before you judge the text |
| Vision (detect, segment, depth, face) | labelled images | official metric on a small labelled set (mAP, IoU, MAE) | overlay the output on the input and look at it |
| 3D | single image, multi-view, text | official mesh or render, vertex counts, time | render and view turntable frames. Check for defects the metric misses |

For Swarm verification, use the modality's route in the extension: images through `GenerateText2Image`, audio through the
AudioLab websocket or API, and the others as `docs/Agents/AGENTS.md` describes. If a modality has no Swarm route yet, say so
plainly.

## Common traps

- An SDPA mask is silently dropped.
- A reshape used as output gives zeros.
- fp8 and ConvRot key remaps go wrong.
- Casting BF16 weights to F16 overflows and produces NaN in SDPA.
- Host loops end up on hot paths.
- Position and ID tensors are never disposed.
- "Silent return" xUnit tests pass without running. Gate real-weight tests, and label their skips.
- `Directory.Build.props` and changelog conflicts appear on every merge.
- Stale test binaries after merges cause AccessViolation.
- The native extension rebuild is keyed on git HEAD, so delete the build directory to force it.
- `UpdateAndRestart` reports "No changes found" after a manual pull. Pass `force:true`.
- `pkill -f` kills your own shell. Kill by PID instead.
- A repack's VAE or encoder key layout can differ from the diffusers one.
- A distilled model ignores some samplers and negative prompts. Refuse them or document them. Do not map them.

# Fibo 2.0 GPU handoff

State: `fibo2-dev` (text-to-image) is implemented on `feat/fibo2-support` (draft PR #505): denoiser, text bundles,
prompt template, flow-Euler pipeline with CFG, recipe, variants, catalog and side-model entries, `alpha.386`.
**Nothing has run on the real checkpoint or on a GPU.** Do not claim parity, quality or speed until the steps below run.
Architecture and constants: [FIBO2_ARCHITECTURE.md](FIBO2_ARCHITECTURE.md).

What the code has been checked against today:
- Unit and tiny-config parity tests (`FullyQualifiedName~Fibo2`, CPU, no GPU). The parity reference is a Python
  transcription of ComfyUI's forward, not ComfyUI itself, and it runs on tiny random weights.
- The recipe loads the dev checkpoint's config from weights (`Fibo2Config.FromWeights`). This has not been run on the
  real header.

Not checked: real weights, a GPU backend, output quality, speed, memory, the real text-encoder output, the CLI and API
on the real model. An earlier attempt at an end-to-end run used random weights on the CPU. It proved only that the
code runs. It is not evidence of correctness or speed, and it was removed.

## Hardware and downloads

- GPU with at least 24 GB VRAM for bf16 (DiT 16.45 GB, text encoder 8.88 GB, VAE 0.34 GB). Check the VRAM policy the
  recipe logs. The int8 builds are not wired (the recipe refuses quantized tensors), so use bf16.
- Host RAM: loading a 16 GB checkpoint can stage F32 copies. Require free host RAM of about 2.5 times the checkpoint size
  before starting (see the no-heavy-gpu-runs note in the memory folder).
- The CPU backend cannot run this model: its linear kernels are F32-only, and the recipe casts BF16 weights to F16. Use CUDA.
  Vulkan has not been tried.
- Downloads from `Comfy-Org/Fibo-2.0` (ungated): `diffusion_models/fibo2_dev_bf16.safetensors` (16.45 GB),
  `text_encoders/qwen3vl_4b_bf16.safetensors` (8.88 GB), `vae/flux2-vae.safetensors` (0.34 GB). **Get explicit approval
  for these sizes before downloading.** Record the sha256 of each file and pin it in the catalog and `SideModels.cs`;
  the new side-model entries have no hash yet.
- Licence: CC-BY-NC-4.0. Non-commercial use only. Do not ship any output as commercial work.

## Steps, in order

1. Check out `feat/fibo2-support` (head `7bb190f9` or later). Run `dotnet build`. Run the Fibo2 unit and parity tests:
   `dotnet test tests/HartsyInference.Diffusion.Tests --filter "FullyQualifiedName~Fibo2"`. They must pass before anything
   else.
2. Fix the side-model autofetch bug before running with downloads disabled. `paths.sideModelAutofetch=false` is ignored
   for the Fibo text encoder and VAE: `Fibo2Recipe.cs` (about lines 75-76 and 86-87) calls `ModelFileLocator.RequireOrEnsure`,
   which calls `ModelDownloader.EnsureSideModelAsync(..., downloadIfMissing: true)`. Repro: set the knob to false, omit the
   text-encoder and VAE overrides, and delete the encoder file. It downloads. Fix this as a separate change with a test.
3. Download the three files (after approval). Read the real transformer header through `Fibo2Config.FromWeights` and check it
   reports injection layers 6, 14, 22, 30, 38, FFN width 9728 and 512 Perceiver queries, as the header at revision
   `b2bffaf57d48` does.
4. Build a reference from ComfyUI at commit `87ed5a066d9c` on the same GPU, with the same checkpoint, prompt, seed,
   1024x1024, 50 steps, CFG 5, Euler, shift 1.15. Save the final image, the sigma schedule and the first denoise step's
   velocity. Also save the text bundles for the prompt (6 tensors of 7680-d per token).
5. Compare, in this order, and stop at the first mismatch to bisect it:
   a. Text bundles for the same prompt: our Qwen3-VL taps, RMS normalisation and bundle concatenation against ComfyUI.
   b. One denoiser step on a seeded latent with the saved bundles (velocity, max absolute error, relative L2).
   c. The full sampler: sigma schedule against ComfyUI, then the final image (SSIM and a visual check).
6. Quality: a fixed prompt set, reviewed against ComfyUI's output. Include a prompt with a JSON caption (the template
   requires it).
7. Speed and memory: time per denoise step, total time at 1024x1024 and peak VRAM, measured against ComfyUI on the same
   GPU. Quote absolute numbers with the external baseline (per AGENTS.md), not against our own earlier runs. With CFG the
   engine runs two denoiser forwards per step; ComfyUI does too unless the caption and the null caption have the same token
   count, when it can batch them (`model_base.Fibo2.extra_conds`).
8. CLI and API, real model: `hartsy image "<prompt>" -m fibo2-dev --steps 50 --cfg 5 --width 1024 --height 1024 --seed 42
   --output <dir>` and `POST /v1/images/generations` with the same settings. Check the two produce identical bytes.
9. Record the results in `docs/Checklists/MODEL_STATUS_IMAGE.md` (the Fibo row) and the parity doc, with the exact
   checkpoint sha256, the ComfyUI commit, the GPU, and the limits. Open a follow-up PR.

## Open items to resolve on the GPU

- Dev steps and CFG. The 50 and 5 come from the diffusers v1 pipeline example, not from 2.0 docs. Confirm from the Bria
  model card or a ComfyUI dev template, then update the recipe `Defaults` and its comment.
- JSON captions: ComfyUI re-serialises them before templating (`parse_prompt`) and the engine does not, so the tokens and
  the gist count can differ. Both take the gist count from the template-wrapped length.
- Patchify order `(c, dy, dx)`: sourced from the diffusers reference and the FLUX.2 latent ops. Confirm against ComfyUI's
  wrapper, which is not in the local checkout.
- Sigma schedule: the timestep input matches ComfyUI, the shift does not. ComfyUI's 1.15 is Flux's exponential shift
  (`s = e^1.15` in `s·t / (1 + (s − 1)·t)`); `Fibo2Pipeline` passes `s = 1.15`. Fix it before step 5c, with a test that pins
  a few sigmas from ComfyUI's `ModelSamplingFlux` at a fixed step count.
- Text-encoder tap convention: `hidden_states[k]` is the input to block k. Confirm the taps against ComfyUI's text encoder
  output.
- Sha256 for the text encoder and VAE. The new VAE entry duplicates the pinned `Flux2Vae` file; decide whether to reuse it.

## Known gaps (not in this PR)

- Glyph text images: captions with `text_render` entries throw `NotSupportedException`. ComfyUI renders them by default.
- Reference-image editing, the edit-turbo variant (its defaults are 4 steps and CFG 1.0; the CFG 1.0 value is an assumption),
  LoRA, refiner, regional prompts, and the int8 builds.
- Negative prompts are refused rather than mapped. Sizes round down to a multiple of 16.
- An unattended 1024x1024 CPU run (random weights) exited at denoise 25/50 with no error in its log and no memory-guard
  trip. The cause is unknown. On the GPU, keep the full log of any long run, and record the last line if the process exits early.
- The variant detector guesses from the file name (the log says so). Pass `-m fibo2:dev` or stamp
  `modelspec.architecture` to make it definitive.

## Done when

- The Fibo2 unit and parity tests pass on the GPU host.
- Text bundles, one denoiser step and the full sampler match ComfyUI within tolerances recorded in the parity doc.
- The CLI and API produce identical output on the real model.
- Speed and VRAM are recorded against ComfyUI on the same GPU.
- The status row, the research doc and the changelog say exactly what was verified.

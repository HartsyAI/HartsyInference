# Handoff: wire the video model catalog for CLI generation

## Context

A prior session just finished the same task for **image** models: wiring `hartsy image -m <id>` so every
catalogued model auto-downloads its checkpoint and was re-verified with a real generation, output viewed and
checked against the prompt (not just "ran without crashing"). See
`docs/Checklists/MODEL_STATUS_IMAGE.md` and `docs/Checklists/PARITY_VERIFICATION.md` for the finished result,
and the memory entries `image-cli-catalog-verify-0721` / `adversarial-image-output-verification` for the
methodology and mistakes made along the way.

Your job: do the equivalent for **video** models — `hartsy video -m <id>`.

## Where video differs from image (read this before starting)

Image models were *all* structurally complete — every catalogued entry already had a working
`IArchitectureRecipe`; the only gap was catalog `Assets` (auto-download metadata). **Video has an extra,
more serious gap**: some catalogued families have no recipe wired into `VideoRecipeRegistry` at all, so
`hartsy video -m <id>` would fail even with perfect catalog metadata. Check this FIRST per family before
assuming "just add Assets" is enough.

## Current state (as of 2026-07-21)

`src/HartsyInference.Cli/Infra/ModelCatalog.cs`, video section (search `// Video`):

```csharp
E("ltx-video", vid, "LTX-Video", "DiT + video VAE", vp, cli: true),
E("wan", vid, "Wan 2.2 (T2V + I2V)", "DiT + Wan VAE", vp),
E("lance-video", vid, "Lance (Video, T2V)", "unified multimodal DiT", vp),
E("kandinsky5-video", vid, "Kandinsky 5 Video", "DiT", vp),
E("cosmos-predict1-5b-v2w", vid, "Cosmos-Predict1 5B Video2World", "AR discrete-token transformer", vp),
E("cosmos-predict1-13b-v2w", vid, "Cosmos-Predict1 13B Video2World", "AR discrete-token transformer", vp),
```

All 6 are short-form `E(...)` entries (no `Assets`, `Status = vp` i.e. ValidationPending). None except
`ltx-video` even have `CliDrivable = true`. This mirrors exactly what the image catalog looked like before
the last session — same fix pattern applies (convert to long-form `CatalogEntry { ... Assets = [...] }`,
mirror `krea2`'s original template from the image catalog).

**`VideoRecipeRegistry.BuildDefaults()`** (`src/HartsyInference.Engine/Recipes/VideoRecipeRegistry.cs`)
currently registers:
```
WanVideoRecipe("wan"), WanVideoRecipe(Wan22_5BCompatClassId), WanVideoRecipe(Wan21_1_3BCompatClassId),
WanVideoRecipe(Wan21_14BCompatClassId), WanVaceRecipe, WanAnimateRecipe, WanS2VRecipe,
LtxVideoRecipe, LtxVideo2Recipe, LanceVideoRecipe
```

Cross-referencing against the catalog's 6 ids:
- **`wan`, `ltx-video`, `lance-video`** — recipe exists, registered. Same "just wire Assets + verify" work as
  images.
- **`kandinsky5-video`** — a `Kandinsky5VideoPipeline` class exists
  (`src/HartsyInference.Video/Pipelines/Kandinsky5VideoPipeline.cs`) but **no `IVideoRecipe` wrapper is
  registered** in `VideoRecipeRegistry`. `hartsy video -m kandinsky5-video` will fail to resolve a recipe
  today. You likely need to write a thin `Kandinsky5VideoRecipe : IVideoRecipe` (mirror `LanceVideoRecipe.cs`
  as the template — same "unified multimodal DiT" shape) before catalog wiring does anything.
- **`cosmos-predict1-5b-v2w` / `cosmos-predict1-13b-v2w`** — same situation: `CosmosV2WPipeline.cs` exists
  in `HartsyInference.Video/Pipelines/`, no recipe wrapper registered. Also: no `TestPaths.cs` entry exists
  for Cosmos-Predict1's checkpoint at all (unlike Wan/LTX/Lance/HunyuanVideo, which all have one) — meaning
  nobody has pinned down a real, confirmed-working checkpoint source for this family yet. Expect to need
  real research here, not just wiring. The catalog comment says "Engine-only; run via the sample invocation
  in VideoCommand help" — check `VideoCommand.cs`'s help text for whatever manual invocation was last used
  to validate this, if any.
- **`HunyuanVideo` (Tencent 13B T2V) is NOT in the catalog at all**, despite being fully built and
  **verified end-to-end** per memory (`hunyuanvideo-13b-e2e` — "coherent T2V @2.15s/step: fp8-resident DiT +
  GPU RoPE + FP8_NATIVE") and having a complete `TestPaths.HunyuanVideo` entry (Comfy-Org repacked bf16 DiT +
  3D VAE). It has no recipe wrapper in `VideoRecipeRegistry` either. This is probably the highest-value
  addition you can make — a real, working, already-parity-verified model that's simply never been surfaced
  to the CLI. Worth adding as a 7th catalog entry + a new `HunyuanVideoRecipe` wrapper (mirror
  `LtxVideoRecipe.cs`).

## Known checkpoint sources (don't re-research these — `tests/HartsyInference.Tests.Common/TestPaths.cs`
already names them, same role `TestPaths.cs` played for image models)

- **Wan 2.2 TI2V-5B**: Comfy-Org repackage of `wan2.2_ti2v_5B_fp16.safetensors` (original Wan naming;
  `WanVideoCheckpointConverter` renames to diffusers) + the shared Wan2.2 VAE (same file Lance uses) +
  umT5-XXL fp8 text encoder. `TestPaths.WanVideo`.
- **LTX-Video**: single file bundles DiT + VAE (`ltx-video-2b-v0.9.safetensors`); T5-XXL extracted from the
  SD3.5 bundle or a standalone fp8-scaled file (reuses the HiDream t5xxl file by default). `TestPaths.LtxVideo`.
- **LTX-2** (dual-stream audio+video, `ltx-2.3-22b-dev-fp8.safetensors`): needs the Gemma-3-12B text encoder
  + SentencePiece tokenizer. `TestPaths.LtxVideo2` — **not in the catalog either**, same gap as HunyuanVideo;
  worth adding if you have time (per memory `ltx2-phase2-streaming-shipped`, this is deployed and working).
- **Lance (Video, T2V)**: `Lance_3B_Video` variant directory + the Wan2.2 VAE (already shared with `wan`) +
  Qwen2 chat tokenizer. `TestPaths.Lance.VideoDir`.
- **HunyuanVideo 13B**: Comfy-Org repacked bf16 DiT (`hunyuan_video_t2v_720p_bf16.safetensors`) + HunyuanVideo
  3D VAE bf16. `TestPaths.HunyuanVideo`.
- **Kandinsky5 Video, Cosmos-Predict1 (5B/13B)**: no `TestPaths.cs` entry — you'll need to find/confirm the
  real repo yourself. Per the **never ship gated links** rule (a hard directive from this project's owner):
  if the canonical HF repo is gated, find an ungated Comfy-Org/community repack instead — do not escalate or
  ask, just find the mirror (same as chroma-radiance → `Comfy-Org/Chroma1-Radiance_Repackaged` and
  hunyuan-image → `QuantStack/HunyuanImage-2.1-GGUF` in the image pass).

## The auto-download mechanism works the same as images — no video-specific plumbing needed

`VideoCommand.cs` → `CommandRunner.Run(Modality.Video, ...)` is the exact same code path `ImageCommand` uses.
`ModelAcquisition.EnsurePresent` and `ModelDownloader` are modality-agnostic — populating `Assets` on a video
`CatalogEntry` will Just Work the same way it did for every image model. (Contrast with audio, which needed
a whole separate `EnsureAudioAssetsPresent` branch because TTS/STT models use a different cache mechanism —
video does NOT have that problem, it follows the image pattern exactly.)

## Verification methodology — READ THE MEMORY FIRST

Read `adversarial-image-output-verification.md` in this project's memory before starting. The short version:
**never mark a video model `CliDrivable = true` because generation completed without crashing.** You must
actually inspect the output and confirm it matches the prompt.

For video specifically, "viewing the output" is less trivial than an image:
- Extract a handful of frames (first, middle, last at minimum) with `ffmpeg -i out.mp4 -vf "select=eq(n\,0)"
  frame0.png` (adjust `n` per frame index) and view those, OR view the whole clip if your tooling supports it
  directly.
- Check for temporal coherence, not just single-frame correctness — a video model can produce a perfect first
  frame that dissolves into noise/flicker by frame 30. Sample across the full clip length, not just frame 0.
- Same "escalate before concluding broken" rule as images: if a terse prompt produces a static/wrong subject,
  try a more explicit prompt before assuming a wiring bug — but if you find a genuine bug (wrong compositing,
  temporal artifacts, wrong subject entirely), root-cause it properly like the hunyuan-image F16-overflow fix
  from the image pass, don't just paper over it.

## Practical constraints — video checkpoints are much bigger than image ones

- `Models/Stable-Diffusion/Wan/` is already 130GB on disk (partial/mixed variants), `LtxVideo/` 8.8GB. These
  models run 13B–30B+ parameters, tens of GB per checkpoint, often with a large text encoder alongside
  (umT5-XXL, Gemma-3-12B). Disk hygiene (download → verify → pin sha256 → **delete the multi-GB file**,
  since the pinned hash makes it re-fetchable) is even more critical here than for images — this box only
  has 30-50GB free at any given time. Check `df -h /` before and after every download, and don't let disk
  hit 0 — a session earlier today hit 100%/5.3GB free by forgetting to clean up one 12GB checkpoint after
  verifying it; don't repeat that.
- Video generation is much slower per-run (multiple denoise steps × many frames, vs one image's single
  frame) — expect individual verification runs to take significantly longer than the image pass's ~1-10
  minutes each. Plan accordingly; use `Bash` background + `Monitor`/`ScheduleWakeup` rather than blocking.
- GPU is shared with other concurrent sessions on this box (2× GPU: a 3060 and a 4090 — `nvidia-smi -L` to
  confirm, `CUDA_VISIBLE_DEVICES` picks between them but note the ordinal mapping does NOT match nvidia-smi's
  own index order, since `CUDA_DEVICE_ORDER` isn't pinned to `PCI_BUS_ID` here — verify empirically with
  `nvidia-smi --query-compute-apps` while a job runs rather than assuming). Video models' large VRAM
  footprint makes contention more likely to cause OOM than it was for images — check `nvidia-smi` before
  every run and don't hog the GPU from another active session.

## Suggested order of attack

1. Read `docs/Checklists/MODEL_STATUS_VIDEO.md` and `PHASE_9_VIDEO.md` for what's already documented as
   parity-verified (mirrors how `MODEL_STATUS_IMAGE.md` shortcut most of the image research).
2. Start with `wan`, `ltx-video`, `lance-video` — recipes already registered, just need catalog `Assets` +
   a real verification run. Fastest wins, same mechanical pattern as every image model.
3. Add `HunyuanVideo` as a new catalog entry + recipe wrapper — it's already fully built and
   parity-verified per memory, "just" needs the recipe-registry + catalog plumbing. High value, should be
   straightforward since the hard numerical work is already done.
4. Consider adding `LTX-2` (dual-stream audio+video) the same way — also already deployed per memory.
5. `kandinsky5-video` and the two `cosmos-predict1-*-v2w` entries need an `IVideoRecipe` wrapper written
   first (pipelines exist, wrappers don't) — do these last, they're the most work and least certain to have
   a confirmed checkpoint source.
6. Update `docs/Checklists/MODEL_STATUS_VIDEO.md` and `PARITY_VERIFICATION.md` per model as you go, same
   style as the image pass — "CLI catalog-path verified <date>" notes, honest documentation of any real
   limitation found (VRAM ceilings, prompting quirks) rather than a blanket CliDrivable=true.

## Directives that carry over unchanged

- Never ship a gated HuggingFace link — find an ungated repack (Comfy-Org, community mirrors) instead.
- Never mark a model verified without viewing the actual output and checking it against the prompt.
- No `git commit`/`push`/`merge` unless explicitly asked — edit in place.
- GPU is shared — check `nvidia-smi`, don't evict another session's work.

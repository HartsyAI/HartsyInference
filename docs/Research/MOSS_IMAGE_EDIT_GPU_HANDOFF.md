# MOSS-Image edit GPU handoff

State: MOSS-Image reference-image edits (the Ostris / ai-toolkit edit-LoRA layout) are on `main` since
PR HartsyAI/HartsyInference#458 (alpha.382). The colorize layout (`reference.layout=colorize`, the one public reference
LoRA, §3b) ships in alpha.383. The DiT side, the Qwen3.5 image input and the engine wiring are
verified on the CPU, and the DiT side also on Vulkan (RTX 2060 SUPER), all with tiny fixtures against float64 / transformers references. **Nothing has run on real weights or on CUDA; do not claim
parity.** Contract, layouts and divergences: [MOSS_IMAGE_ARCHITECTURE.md](MOSS_IMAGE_ARCHITECTURE.md#reference-images-edits).
Upstream: `github.com/ostris/ComfyUI-MOSS-Image-Ostris-Edit` (`nodes.py`) and ai-toolkit
`extensions_built_in/diffusion_models/moss_image/src/{transformer,pipeline}.py`.

Your job, in order: the CUDA test suites, the first real text-to-image run, the real edit runs, the no-reference A/B,
then record everything in a follow-up PR (with fixes if anything breaks). Run one GPU job at a time.

## 0. Setup
1. `git fetch && git switch -c claude/moss-edit-real-weights origin/main`, push it, and open a draft PR early.
   Read `AGENTS.md`, `docs/CODE_STYLE.md`, `docs/Agents/AGENTS.md`.
2. Weights (Comfy-Org `MOSS-Image` repack, single files). Find them first:
   `ls -R <models root> | grep -iE "moss_image|qwen3.5_9b|flux2-vae"`. The models root is the `HARTSYINFERENCE_MODELS`
   override, else `<repo>/Models` (`src/HartsyInference.Engine/RepoPaths.cs`); `tests/regression-ab.sh` uses
   `$HARTSY_MODELS_ROOT`, default `/mnt/model-storage/Models`. Needed:
   - DiT: `diffusion_models/moss_image_base_bf16.safetensors` (SFT also fine; int8 ConvRot loads too). Main's RoPE A/B
     ran it from `Stable-Diffusion/moss-image-base-1_0-moss-image-base-bf16.safetensors` under the models root.
   - Encoder: `text_encoders/qwen3.5_9b_bf16.safetensors`. It is also the recipe's fallback side model under the
     target name `text_encoders/qwen35_9b_bf16.safetensors` (sha256 `7e6e9f08…`). Symlink it to that name, or pass
     `--text-encoder`; with neither, the recipe downloads 19 GB.
   - VAE: `vae/flux2-vae_small.safetensors` (fallback target name `VAE/flux2-vae_small.safetensors`), or pass `--vae`.
3. The encoder must carry the vision tower:
   `python -I tools/inspect_checkpoint_header.py <encoder> | grep "visual.patch_embed.proj.weight"`. The key must be
   `model.visual.patch_embed.proj.weight` or `visual.patch_embed.proj.weight` (the prefixes `Qwen35VisionTower` probes).
   If neither is present, download `Qwen/Qwen3.5-9B` `model-00004-of-00004.safetensors` (1.87 GB, all 333 visual
   keys) and pass it as `--vision-encoder`. The recipe log line ends `reference images on` when the tower attached.
4. `free -g` first: the bf16 encoder alone is 19 GB of host-mapped weights.
5. Build: `MSBUILDDISABLENODEREUSE=1 dotnet build src/HartsyInference.Cli -c Release -f net10.0 -nodeReuse:false -m:1 -o /tmp/hartsy-cli`.

## 1. Test suites (CUDA has never run these)
```
dotnet test tests/HartsyInference.Diffusion.Tests --filter "FullyQualifiedName~MossImageGpuParityTests"
dotnet test tests/HartsyInference.Diffusion.Tests --filter "FullyQualifiedName~MossImage|FullyQualifiedName~Qwen35Multimodal|FullyQualifiedName~ReferenceConditioning|FullyQualifiedName~ReferenceImageResolver"
dotnet test tests/HartsyInference.Core.Tests --filter "FullyQualifiedName~MRopeTests"
```
`MossImageGpuParityTests.Edit_MatchesReference` and `StepGraph_ReplaysEditBranches` are the CUDA firsts. They cover:
- token-major F16 attention with more keys than queries (cached mode);
- the row-indexed affine/gate kernels under graph capture (joint mode);
- pinned reference caches across replays.

Then the CPU lane (filter in `docs/CODE_STYLE.md`). On the sm_75 dev box
`BackendSelectorTests.Resolve_Auto_Prefers_Cuda_Then_Vulkan_Then_Cpu` fails because `auto` passes over Turing to
Vulkan; it should pass on the 4090. Record any failure by name.

## 2. Real text-to-image run (MOSS ran on the 4090 in main's RoPE A/B, never with references)
```
dotnet /tmp/hartsy-cli/HartsyInference.Cli.dll image "a red bicycle leaning on a white wall, photo" -m moss-image \
  --model-path <dit> --text-encoder <encoder> --vae <vae> -b cuda --width 1024 --height 1024 --steps 30 --cfg 3 --seed 7 \
  -o Output/moss-edit/t2i
```
Expect a coherent, on-prompt photo. If it is noise or off-prompt, stop and debug text-to-image before any edit work.
- Check `Flux2VaeComponents` decoder widths (alpha.380 fixed a 96-384 decoder loading into 128-512 layers).
- Check the encoder log line (bf16, 32 layers).
- Check `docs/Checklists/TROUBLESHOOTING.md` for MOSS.

Optional deeper check: `tests/python-reference/dump_moss_image_reference.py real` + `MossImageRealWeightTests`. These need OpenMOSS's diffusers folder (`OpenMOSS-Team/MOSS-Image-Base`, 27 GB), not the Comfy files.

## 3. Real edit runs
Use the step-2 output (or any photo) as the reference, the same flags, and the prompt `"turn the bicycle blue"`:
1. Cached (default): add `--reference-image <photo.png>`.
2. Joint: add `--reference-image <photo.png> --reference-kv-cache off`.
3. Init image as Picture 1: add `--init-image <photo.png> --init-image-mode reference`.
4. Two references: two `--reference-image` flags.

Check in the logs:
- The encoded token count is the template + 144 placeholders per full-budget reference + the prompt.
- `cached reference keys/values for 42 of 42 layers` (cond) and `41 of 42` (uncond, block 22 skipped), with their MiB.
- The warning `Reference images without a LoRA` appears when no LoRA is loaded.

Record each run's wall clock, per-step ms and peak VRAM (`nvidia-smi --query-gpu=memory.used -lms 500`).

Without an edit LoRA the base model is not trained to follow references. Expect a coherent image, not a faithful edit.
That proves the mechanics only, so say so. If an ai-toolkit MOSS edit LoRA is available, run 1–3 again with
`--lora <file> --lora-weight 1.0`, using cached mode unless the LoRA was trained with `kv_cache: false`. On 2026-10-20
none was public on Hugging Face or Civitai (search `moss` LoRAs and `ostris` models again first).

`sorryhyun/moss-image-colorize` is trained on a different layout, the colorize one, so it does not test this Ostris
path; it is the next section.

## 3b. Colorize layout: a B&W page through the colorize LoRA (`sorryhyun/moss-image-colorize`)
The LoRA is `moss_colorize_ref_v1.safetensors` (149 MB, bf16, rank 32 / alpha 128, `proj` layers of all 42 blocks) from
`huggingface.co/sorryhyun/moss-image-colorize`; put it in the models' `loras` folder and pass it as `--lora`. Check its
header before the run (`python3 -I tools/inspect_checkpoint_header.py <lora>`): 756 tensors named
`lora_unet_layers_{0..41}_proj_{q,k,v,o,ffn_up_gate,ffn_down}.{alpha,lora_down.weight,lora_up.weight}`, metadata
`base_model=MOSS-Image` and `reference_latents=1000`. Its metadata has no `ref_scale`; the trained scale is 2, the default.

Input: a B&W manga page `page.png` and its caption (colour tags), or an empty caption for page-only colouring.

Size: pass the page's target size, not a free choice. The node derives the target from the page itself, with
`target_size(page_w, page_h, resolution=1024)`: native aspect, sides multiples of 16, area at most 1024², never larger
than the page, and a page already on the 16 grid within 1.1·1024² is kept as-is. The engine takes its size from
`--width`/`--height`, so it does not do that for you. Compute the size with this (the node's `target_size`, checked
against `nodes.py` over 1 356 page sizes), where `W H` is the page's pixel size (`identify -format '%w %h' page.png`):
```python
import math, sys

def target_size(w, h, resolution=1024, stride=16, tolerance=1.10):
    cap = resolution * resolution
    if w % stride == 0 and h % stride == 0 and w * h <= cap * tolerance:
        return w, h
    area = min(cap, w * h)
    aspect = w / h
    out_w = round(math.sqrt(area * aspect) / stride) * stride
    out_h = round(math.sqrt(area / aspect) / stride) * stride
    out_w = max(stride, min(out_w, w // stride * stride))
    out_h = max(stride, min(out_h, h // stride * stride))
    return out_w, out_h

print(*target_size(int(sys.argv[1]), int(sys.argv[2])))
```
Run it as `python3 target_size.py W H`. For the 848×1200 example page it prints `848 1200`, so the commands below pass
`--width 848 --height 1200`. A `--width 1024 --height 1024` request would cover-crop that page to a square and drop its
top and bottom.

Colour (the test):
```bash
dotnet /tmp/hartsy-cli/HartsyInference.Cli.dll image "touhou, purple eyes, purple hair, black hairband, blue shirt" -m moss-image --model-path <dit> --text-encoder <encoder> --vae <vae> --lora <moss_colorize_ref_v1.safetensors> --lora-weight 1.0 --reference-image page.png --reference-layout colorize --sampler er_sde --steps 30 --cfg 3 -b cuda --width 848 --height 1200 --seed 1 -o Output/moss-edit/colorize
```
Control (the same command with the `--lora` and `--lora-weight` flags removed; the base never saw a reference, so expect
no faithful colouring):
```bash
dotnet /tmp/hartsy-cli/HartsyInference.Cli.dll image "touhou, purple eyes, purple hair, black hairband, blue shirt" -m moss-image --model-path <dit> --text-encoder <encoder> --vae <vae> --reference-image page.png --reference-layout colorize --sampler er_sde --steps 30 --cfg 3 -b cuda --width 848 --height 1200 --seed 1 -o Output/moss-edit/colorize-control
```
Then the page-only run (empty caption, `""`) with the LoRA.

Pass: the page's line work survives, and the hair, eyes and shirt take the caption's colours in their places. Fail: noise,
lost line work, or colour unrelated to the caption. Also check the logs:
- The result meta has `reference_layout=colorize` and `reference_attention=inline`, and the generation line reports
  `1 reference image(s), Inline`.
- The positive encode logs `(1 image(s))` with the vision pads for the page's vision copy plus the caption: 988 for
  848×1200 (the copy is 832×1216 after the factor-32 resize), and 1 024 for 1024×1024. The negative logs `(0 image(s))`:
  it is a plain encode of the empty prompt.
- The VAE copy is the target size over 2 per side, rounded to 16: 416×608 for 848×1200 (512×512 for 1024²). The prompt
  has no `Picture` label.
- The control's colouring differs from the LoRA run: if it does not, the LoRA did not load.

Record wall clock, per-step ms and VRAM, as in §3. ER-SDE disables the CUDA step graph, so these runs are slower per
step than the default sampler; that is expected, not a regression.

Triage a failed colouring run:
- Placement off (colour bleeding across lines, the page ghosting): the stretch in `MossImagePosEmbed.InlineCoords` and the
  `[image | ref | text]` layout in `MossImageTransformer.ForwardCore` (the `inline` branch).
- Wrong pad count: `MossImageRecipePipeline.ColorizeVisionSize` (the node's `smart_resize` at the page's own pixel count).
- No colour at all with the LoRA on: check `--lora-weight`, and that the LoRA's keys resolve (`MossImageLoraMapper`).
- A wrong sampler: the layout defaults to `er_sde` and 30 steps only when the request names neither.

Triage if an edit fails or looks broken:
- Cached fails, joint works: the cache path (`ReferenceKvCache`, `MossImageBlock.ForwardPrefix`, the block's
  `ReferenceAppend` K/V rewrite, so K/V are longer than Q).
- Both fail under CUDA but pass with `hartsy settings set numerics.ditGraph false`: step-graph capture (fixed buffers,
  `BranchSignature`, pinned caches). Restore the knob afterwards.
- F16 issues: retry with `numerics.ditF16=false`.
- Encoder errors (placeholder count, M-RoPE sections): `Qwen35TextEncoder.Encode(ids, images)`. The image path is
  pinned only by the tiny transformers fixture; compare one real encode against transformers `Qwen3_5Model` before
  changing anything.
- Shape and cache-size math is in `MossImageReferenceState` and `MossImageTransformer.PrepareReferences`.

## 4. No-reference A/B (identical expected)
Plain generation must be bit-identical to main before #458. The edit work rerouted the block forward, attention and
conditioning through new overloads that issue the same ops. `tests/regression-cases.sh` has a `moss-base` row (tag
`moss`, the RoPE A/B's checkpoint path). Use the commit before the merge as the base:
```
base=$(git log --merges -1 --format=%H --grep "#458" origin/main)^1
tests/regression-ab.sh --base "$base" --tag moss --filter moss-base --expect identical --backend cuda
```
The case needs the encoder and VAE at the side-model target names (step 0.2), because the row passes no
`--text-encoder`/`--vae`. Put the report table in the PR body.

## 5. Record and push
- `docs/Checklists/PARITY_VERIFICATION.md`, MOSS row:
  - the CUDA GPU-suite results;
  - the real T2I and edit runs (checkpoint, backend, date, timings, VRAM);
  - the A/B verdict;
  - what is still unverified.
- `docs/Checklists/MODEL_STATUS_IMAGE.md` (MOSS row and section): move "real weights / CUDA" from not-verified to
  verified only for what actually ran. A docs-only follow-up skips the version bump; a fix adds the next `## alpha.N`
  and the `Directory.Build.props` bump (repo shipping rules).
- Images go to the PR as attachments or descriptions, not into the repo (`Output/` is gitignored).
- Commit to the follow-up branch, push, put the verification and A/B tables in its PR body, get it bot-reviewed and
  answer every thread. kalebbroo is the sole author: no co-author or tool trailers anywhere.

## 6. Not in scope (proposed follow-ups)
- Migrating the duplicate M-RoPE copies (Qwen2.5-VL, Qwen3-VL, MiniMax) and Flux3's span/reference helpers onto the
  shared ones (code TODOs).
- An mmproj GGUF vision-tower source for GGUF encoders.
- A cross-check of the numpy edit transcription against the Ostris node's patched forward. It needs a ComfyUI install
  with the node; add a `verify` mode to `dump_moss_image_reference.py` if you have one.

# MOSS-Image

OpenMOSS's open-weights text-to-image model (released 2026-10-16): a 5B single-stream DiT over FLUX.2 VAE latents,
conditioned on a frozen Qwen3.5-9B, with native layout prompts. Two builds: **MOSS-Image-Base** (no post-training,
meant for LoRA training) and **MOSS-Image-SFT** (Base fine-tuned on curated data). Same architecture and defaults.

## Sources (snapshot 2026-10-16)

- OpenMOSS/MOSS-Image 0ee86ac: the ComfyUI node `comfyui/moss-image/` (`model.py` is the parity reference;
  `loaders.py` key signature and config; `hf.py` diffusers-folder loaders; `text_encoder.py` tokenizer and encoder;
  `export.py` single-file export), `USERGUIDE.md` (sampling, SLG, layout format), the prompt-enhancer skill's
  `scripts/compile_prompt.py` and `references/layout.md` (layout compiler and its box rule), `gradio_app.py`
  (`GenerationConfig` defaults), and the paper.
- SGLang 2c2986b `python/sglang/multimodal_gen`: `runtime/models/dits/moss_image.py`,
  `runtime/models/schedulers/moss_image.py` (training-grid schedule), `configs/sample/moss_image.py`, and the unit
  tests `test_moss_image.py` (schedule goldens, guidance formula).
- Weights: `OpenMOSS-Team/MOSS-Image-Base` and `OpenMOSS-Team/MOSS-Image-SFT` (also on ModelScope `openmoss/`). Each is
  a diffusers pipeline folder: `transformer/` (`diffusion_pytorch_model-*.safetensors` + index, `_class_name`
  `MOSSImageTransformer`), `text_encoder/` (Qwen3.5-9B, `model_type` `qwen3_5`, sharded), `tokenizer/`, `vae/`
  (`AutoencoderKLFlux2`). `comfyui/export.py` repackages it as `diffusion_models/{name}-dit.safetensors`,
  `text_encoders/{name}-qwen35.safetensors` and `vae/{name}-vae.safetensors`, keys unchanged. Shard names, hashes and
  the weights' license are unread here (Hugging Face is unreachable from the build container); the README licenses
  only the code (Apache 2.0).

## Architecture

- **Latent.** FLUX.2 VAE (the FLUX.2-klein-4B one): 32 channels, 2×2-patchified to 128 per token with BatchNorm
  normalization, /16 per token. `video_embedder` is Linear(128 → dim) over the raster-ordered tokens.
- **Text.** Qwen3.5-9B (hybrid Gated DeltaNet + gated full attention, 32 layers, hidden 4096), frozen, the final
  normalized hidden state of every token. Template `<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n`
  (transformers' `apply_chat_template(enable_thinking=True)`), NFC-normalized, truncated to 4096 tokens, no padding:
  each prompt runs at its own length.
- **Text refiner** (`text_embedder`): Linear(4096 → dim), then two biased blocks: RMSNorm → fused `qkv` → per-head
  Q/K RMSNorm → RoPE (text positions) → attention → `proj`, then RMSNorm → `ffn_gate`/`ffn_up` → SiTU-GLU →
  `ffn_down`. It depends only on the prompt, so it runs once per generation.
- **Joint blocks** (42 at dim 3072, 24 heads of 128, FFN 8192) over `[image ; text]`, single stream. ONE modulation
  for every block: `shared_block_modulation(t)` gives `(shift, scale, gate)` for attention and for the FFN. Each block:
  `x += gate_a·o(attn(rope(norm_q(q)), rope(norm_k(k)), v))` from `rms(x)·(1 + scale_a) + shift_a`, then
  `x += gate_f·ffn_down(situ(ffn_up_gate(rms(x)·(1 + scale_f) + shift_f)))`. Joint-block projections are bias-free;
  the norms carry weights.
- **SiTU-GLU.** `ffn_up_gate` rows are `[hidden | gate]`;
  `situ(h, g) = (4·tanh(g/4)·sigmoid(g)) · (25·tanh(h/25))`.
- **RoPE.** 4 axes `(r, t, h, w)`, rotate-half, `head_dim/8` bands per axis, θ 10000, bands built in F32 on the CPU.
  Image token `(0, L_text, row − (H−1)/2, col − (W−1)/2)`, text token i `(0, i, 0, 0)`. Channel i pairs with
  i + head_dim/2 and reads axis `i / (head_dim/8)`.
- **Time.** Sinusoid of the timestep (σ·1000) in F64, `[cos, sin]`, `10000^(−i/half)`, 256 wide, cast to F32, then
  `time_embedding` Linear → SiLU → Linear.
- **Head.** `final_norm_video` (RMSNorm, eps 1e-6) on the image tokens, `·(1 + scale) + shift` from
  `video_head_modulation(t)` (`[shift, scale]`), then `final_linear_video` (bias-free) to 128 channels.
- `rope.bands` in the checkpoint is a derived buffer, rebuilt on load.

## Sampling

- **Schedule** (`MOSSImageScheduler`): `steps` points linear from 1 to `shift/(1000 + shift − 1)`, shifted by
  `σ = s·v/(1 + (s − 1)·v)` with s = 5, each moved to the nearest point of the 1000-step training grid (ties to the
  lower) and consecutive duplicates dropped. Euler, 50 steps. The engine computes each grid point in F64 and rounds
  once: the reference's F32 `linspace` differs by an ulp across CPUs, and one rounding reproduces SGLang's goldens.
- **Guidance.** True CFG whenever the scale is not 1 (default 3, recommended 3–6; raise it for short prompts), the
  negative prompt empty by default. "Two-pass SLG": the unconditional branch skips joint block 22 (22, 24 and 26 all
  help); turning it off gives a more photographic look and can collapse details. Optional "three-pass SLG": an extra
  conditional branch with `slg_layers` skipped adds `slg_scale·(cond − skipped)` (OpenMOSS suggests scale 3 over
  0.01–1). Both apply on steps whose `index/steps` lies in `[start, stop)`.
- **Resolution.** Multiples of 16; best at 1–2 MP, works from 0.25 MP to 4 MP at any aspect ratio.
- **Layout prompts.** The compiled form is one compact JSON object,
  `{"summary":{"caption":…},"items":[{"label":"object|character|text","caption":…,"text"?:…,"bbox":[ymin,xmin,ymax,xmax]}],"bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}`,
  with boxes in pixels about the canvas centre: top/left edges floored, bottom/right ceiled, then `H//2`, `W//2`
  subtracted. Plain prose, tag lists and a layout JSON pasted as the prompt are all valid input.

## Engine mapping

- Family `moss-image`, variants `base` (default) and `sft` (an `sft` token in the file or folder name, or the hint).
  Detected by `shared_block_modulation` + `layers.0.proj.ffn_up_gate` + `video_head_modulation`; a shard is judged by
  its index's whole key set, since those land in different shards.
- **Checkpoints.** The diffusers folder (the folder, `transformer/`, or any shard: the whole shard set loads), the
  ComfyUI export, or a GGUF (family row `moss_image`; `hartsy quantize` writes Q8_0/Q6_K/Q5_K/Q4_K). The DiT opens
  through `CheckpointSource`, so BF16, FP8 (scaled or plain), int8 and NVFP4 files all load. The encoder, tokenizer and
  VAE come from the folder when the checkpoint is inside one, else from component overrides, else the side models: the
  Qwen3.5-9B Q8_0 GGUF (which carries its tokenizer) and the shared Flux.2 VAE. A safetensors encoder (the folder's
  BF16 one, ComfyUI's single file, fp8-scaled) is supported too.
- **Prompts.** `MossImagePromptTokenizer` renders the template. Emphasis is CondScale (OpenMOSS's ComfyUI tokenizer
  disables weights). Regional prompt parts become the layout JSON (`LayoutPromptBuilder.BuildSceneLayout`): the untagged
  prompt is the summary caption and each region an `object` item.
- **Guidance knobs.** Request extras `slg.uncond_skip_layers` (default `22`; `none` turns it off), `slg.layers`,
  `slg.scale`, `slg.start`, `slg.stop`; CLI `--slg-uncond-layers`, `--slg-layers`, `--slg-scale`, `--slg-start`,
  `--slg-stop`. The sigma shift maps directly (SwarmUI's `ModelSamplingSD3` shift).
- **LoRA.** `MossImageLoraMapper`: PEFT under `transformer.`, ComfyUI under `diffusion_model.`, bare module names, and
  underscored `lora_unet_` / `lora_transformer_` / `lora_diffusion_model_` (resolved through `MossImageKeyNames`' module
  templates). Every spelling names the same module once its wrapper is stripped.

## Reference images (edits)

OpenMOSS ships no reference path. ai-toolkit trains edit LoRAs on the stock weights with `model_kwargs.edit: true`, and
Ostris's ComfyUI node `ostris/ComfyUI-MOSS-Image-Ostris-Edit` (`nodes.py`) adds the inference side; ai-toolkit's
`extensions_built_in/diffusion_models/moss_image/src/{transformer,pipeline}.py` is the training side. The engine
follows both.

- **Text side.** Per reference (at most 3), `Picture N: <|vision_start|><|image_pad|>…<|vision_end|>` before the prompt,
  inside the usual template, one `<|image_pad|>` per merged vision token. The negative prompt gets the same blocks and
  images (the node's README wiring). The vision copy is scaled down only to fit 384² pixels, then Qwen's processor
  (factor 32, 65 536–16 777 216 pixels, mean/std 0.5, the frame duplicated across the temporal patch): a full-budget
  image is 144 tokens. The encoder's whole final hidden state, image rows included, is the conditioning.
- **Qwen3.5 vision tower** (`qwen3_5_vision`): the Qwen3-VL ViT (27 blocks, 1152 wide, learned 2304-entry position
  embedding interpolated to the grid, 2-D rotary, tanh-GELU MLPs) with no deepstack taps and an exact-GELU merger to
  4096. Image tokens replace the placeholder rows; the full-attention layers rotate by interleaved M-RoPE (sections
  11/11/10 over the 32 pairs of the 64-wide rotary slice; text after an image resumes at its start + max(H, W)).
  The DeltaNet layers read no positions. Comfy-Org's `qwen3.5_9b_*` files and the HF release carry the tower;
  OpenMOSS's `text_encoder/` and the llama.cpp GGUFs do not.
- **DiT side.** Each reference is VAE-encoded (scaled down only to 1 MP, sides rounded to 16), BN-packed like the target,
  and embedded by the same `video_embedder`. Reference `i` sits at RoPE `(i + 1, L, row − (h−1)/2, col − (w−1)/2)` on its
  own grid, and takes the modulation of timestep 0; the head reads the image rows only.
- **Two attention layouts, set by how the LoRA was trained.** Cached (ai-toolkit's `kv_cache`, its default): the
  references run once at timestep 0 attending only to each other, every block's rotated keys and raw values are kept,
  and each step appends them after the live `[image ; text]` keys and values. A cache depends on the prompt length (the
  references' RoPE frame) and the skipped blocks, so cond and uncond (block 22 skipped) build their own. Joint
  (`kv_cache: false`): the references ride after the text every step, attending to everything, with per-segment
  modulation through the row-indexed ops.
- **The colorize layout** (`reference.layout=colorize`, for `sorryhyun/moss-image-colorize`; the spec is its ComfyUI node
  `sorryhyun/ComfyUI-MOSS-i2i` (`reference.py`, `nodes.py`), which beats the card). One page, and no `Picture N:` label:
  the text encode is one user turn, `<|vision_start|>[pads]<|vision_end|>{caption}`, and the negative is a plain encode
  of the empty prompt with no image. The vision copy is the page cover-fit to the target size and smart-resized (factor
  32, 65 536-pixel floor) to that size's own pixel count, about 1 000 tokens at 1 MP. The target size is the request's
  `--width`/`--height`, which the caller must set to the node's `target_size` for the page (see Divergences). The VAE copy is the same page at
  `1/reference.ref_scale` per side (default 2, the trained scale, which the LoRA metadata does not record), sides rounded
  to 16 with ties to even. The DiT sequence is `[image | ref | text]`: the ref carries no noise, takes the step's live
  modulation (not timestep 0, so no segment layout), and sits at RoPE `(1, L, i, j)` with its `(h, w)` stretched onto the
  target grid, `(i + ½)·H/h − ½`, centred like the image. Attention is full and nothing is cached; the head reads image
  rows only. The ref is in every CFG branch, so the uncond branch (block 22 skipped) keeps it. The parity test cannot see
  the order (attention is permutation-equivariant given the coordinates); the order is the node's by construction.
  Card sampler: 30 steps on ER-SDE (the defaults for this layout), CFG 3, shift 5, skip 22 on uncond.
- **Engine.** `ReferenceImageResolver` with `MossImageRecipePipeline.ReferencePolicy` (Ostris) or `ColorizePolicy`
  prepares both copies; `Qwen35VisionTower` / `Qwen35TextEncoder.Encode(ids, images)` the text side;
  `MossImageTransformer.PrepareConditioning` with `ReferenceLatentTokens` and `EnsureReferenceCaches` the DiT side, on
  the shared `ReferenceKvCache`, `SegmentModulation` and `TokenMajorAttention`. Extras `reference.kv_cache` (default true,
  Ostris only; refused under colorize), `reference.max_pixels` (1 048 576, 0 keeps the size) and `reference.vision_pixels`
  (147 456), both Ostris only; `reference.layout` (`ostris` default, or `colorize`) and `reference.ref_scale` (colorize,
  1–8, default 2). CLI `--reference-kv-cache`, `--reference-max-pixels`, `--reference-vision-pixels`,
  `--reference-layout`, `--reference-scale`, `--vision-encoder`.
- **Cost.** A cache is `2 × blocks × S_ref × 3072` values in the activation dtype, about 2.1 GB in F16 for one 1 MP
  reference over 42 blocks, and an edit builds one per skipped-block set (two by default). The pipeline warns when the
  caches exceed half the free device memory and never switches layout on its own.
- **Divergences.** The VAE copy is resampled once with Keys bicubic (ai-toolkit: bilinear after its own fit); the
  vision copy goes straight to its smart-resize size in one bicubic pass (ai-toolkit: bicubic then the processor's own
  resize); HF's 65 536-pixel floor scales a reference smaller than 256² up, where ComfyUI's 3 136 floor would not.
  The colorize copies use the same Keys bicubic for the cover-fit that the node does in Lanczos, so their pixels differ
  slightly from the node's; the tensor sizes match it exactly.
- **Target size.** The node takes no size: it derives the target from the page with `target_size(page_w, page_h,
  resolution=1024)`, which keeps the native aspect with sides multiples of 16 and area at most 1024² (never larger than
  the page; a page already on the 16 grid within 1.1·1024² is returned as-is). The engine takes its size from the request,
  so a like-for-like run passes the node's size for that page (`--width`/`--height`); a 1024² request on a portrait page
  cover-crops it. The colorize layout does not default the size from the page. The function is checked against
  `nodes.py` over 1 356 sizes in the GPU handoff §3b.

## Performance

- The text side of each branch (prompt encode, the refiner, RoPE tables) is built once per generation, and encoded
  prompts are cached across generations by token ids.
- Qwen3.5 prefill runs on the device: chunked Gated DeltaNet kernels (`SsmConvSplitPrefill`, `SsmDeltaPrefill`) on
  CUDA and Vulkan instead of the per-token host loop.
- Fused ops on CPU, CUDA and Vulkan: `SituGlu` and a weighted `RmsNormModulate` with independent F32/F16 input and
  output, so the F32 residual stream feeds F16 branches without extra casts. Attention runs token-major.
- On CUDA every branch of a step (cond, uncond with block 22 skipped, the optional SLG branch) is captured as one CUDA
  graph and replayed with one launch per step, for every weight format; block-quantized weights trim the memory pool
  before capture. A failed capture falls back to eager kernels for the session.
- On a shared device the DiT is evicted while the 9B encoder runs, and the encoder's weights are freed afterwards.

## Traps

- ComfyUI's native sampler does not snap to the training grid, so its images differ from the standalone pipeline's
  for the same seed. The engine follows the standalone pipeline and SGLang.
- The unconditional branch's skip is part of the default recipe, not an add-on: plain CFG is a different look.
- The unconditional branch reads the negative prompt (empty by default) through its own text refiner pass; only the
  joint blocks differ between the branches by the skipped layer.
- The colorize negative sees no page: its text encode runs with no image, so its prompt cache entry is image-free too.
  The caption goes in raw; `LayoutPromptBuilder` only rewrites region syntax, and plain tags pass through unchanged.

## Open questions

The repositories' shard names and hashes (the catalog entry lists only side models), the weights' license, the
Qwen3.5-9B GGUF's numerics against the BF16 encoder, and the real-weight parity and speed runs on the GPU box
(`dump_moss_image_reference.py real` + `MossImageRealWeightTests`). For edits: a check of the transcription against
the Ostris node's patched forward, a published ai-toolkit edit LoRA to run (none was public on 2026-10-19), and an
mmproj GGUF as a vision-tower source for GGUF encoders. For the colorize layout: the real-weight run of
`sorryhyun/moss-image-colorize` (the only public reference LoRA; `MOSS_IMAGE_EDIT_GPU_HANDOFF.md` §3b), and the
colorize sampler's ER-SDE on flow-match sigmas, which the tiny fixtures do not exercise.

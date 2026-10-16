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
  kohya `lora_unet_` (resolved through `MossImageKeyNames`' module templates). Every spelling names the same module once
  its wrapper is stripped.

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

## Open questions

The repositories' shard names and hashes (the catalog entry lists only side models), the weights' license, the
Qwen3.5-9B GGUF's numerics against the BF16 encoder, and the real-weight parity and speed runs on the GPU box
(`dump_moss_image_reference.py real` + `MossImageRealWeightTests`).

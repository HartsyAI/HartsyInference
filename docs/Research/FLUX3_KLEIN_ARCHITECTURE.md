# FLUX.3 Image Klein

Black Forest Labs' open-weights FLUX.3 image model (released 2026-10-15), about 5.6B parameters: text-to-image,
instruction editing with up to 10 reference images, and native layout boxes, in one set of weights.

## Sources (snapshot 2026-10-15)

- Model code: ComfyUI `comfy/ldm/flux3/model.py`, detection `comfy/model_detection.py` (FLUX.3 block),
  `supported_models.py` `Flux3` (shift), `text_encoders/flux.py` `Flux3Tokenizer`/`Flux3Qwen3VL`, `nodes_flux.py`
  `Flux3LayoutPrompt`, `nodes_edit_model.py` `ReferenceMultiLatent` — ComfyUI a5e95e6.
- Diffusers port: `transformer_flux3.py` and `convert_flux3_transformer_checkpoint_to_diffusers` in
  `loaders/single_file_utils.py` (diffusers 7ecb1e1, "Flux3 image", 2026-10-15). Used for the diffusers module names
  LoRA trainers emit; the parity reference stays ComfyUI's module, which loads the released files as they are.
- LoRA: SimpleTuner `scripts/extract_peft_lora.py` (a0e7ffa), which wrote
  `terminusresearch/flux-3-klein-image-distilled-lora-extract`; ComfyUI `comfy/lora.py` `Flux3` key map.
- Usage and files: ComfyUI docs `tutorials/flux/flux-3-klein.mdx` (Comfy-Org/docs b800c73) and the
  `image_flux3_klein_*` workflow templates (Comfy-Org/workflow_templates).
- SwarmUI 967f005: class `flux.3-klein` (compat `flux-3-klein`), Flux.2 VAE, `qwen3vl_4b.safetensors` text encoder,
  sigma shift sent to `ModelSamplingFlux` as both `max_shift` and `base_shift`.
- Weights: gated `black-forest-labs/flux-3-image-klein-comfy` (FLUX Kommunity License). Files named in the docs:
  `flux3_image_klein_int8_convrot.safetensors` (full), `flux3_image_klein_sd_int8_convrot.safetensors`
  (step-distilled), bf16 and nvfp4 versions of both, `qwen3vl_4b_int8_convrot.safetensors`, `flux2-vae.safetensors`,
  and an optional `qwen3.5_9b_flux3_prompt_upsampler_int8_convrot.safetensors`. Repository layout and hashes are
  unread (gated).

## Architecture

- Latent: Flux.2 VAE (32 channels, BatchNorm stats on the 2×2-patchified 128 channels), /16 per token.
- Text: Qwen3-VL-4B text tower, hidden states `[4, 8, 12, 16, 20, 24, 28, 32]` in HF `hidden_states` indexing
  (ComfyUI captures before layer i), no final norm, concatenated tap-major to 20480. Template
  `<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n` with no think block and no system turn, right-padded
  with 151643 to a multiple of 80.
- DiT: hidden from `attn_out` rows, heads = hidden / 128, every Linear bias-free. One block type everywhere: no-affine
  LayerNorm under a single adaLN `(shift, scale, gate)`, then Q/K/V (or fused `qkv`) and `mlp_in` off the same input,
  Q/K RMSNorm + RoPE, `x += gate·(attn_out(attn) + mlp_out(silu(gate_half)·up_half))`, `mlp_in` = `[gate | up]`,
  MLP ratio 3.0.
- Streams: `content_mode_blocks.{image,image_cond}` and `txt_mode_blocks` (depth each, own modulation from
  `early_stream_modulations`), then `single_blocks` over `[image, references, text]` where each span takes its own
  `single_stream_modulations` row. Image rows use t; references and text use t = 0, so their early stacks depend on
  the conditioning alone.
- RoPE: 4 axes × 32, θ 10000, Flux pairwise rotation. Ids `(t, row, col, l)`: image `(0, 0.5·r, 0.5·c, 0)`,
  reference k `(0, 0.5·r, 0.5·c, −1−k)`, text i `(0, 0, 0, i)`.
- Time: `time_in` MLPEmbedder over the 256-wide sinusoid of 1000·σ (cos first). Final layer: Flux `LastLayer`,
  `[shift, scale]` order.
- Step-distilled build = flow map: `ref_time_in` embeds the sigma the step jumps to (the schedule's next sigma after
  the entry nearest the current one) and adds it to every time vector, so the conditioning streams are per step.

## Sampling

- Static flow-match shift mu = ln(6.93) (`shift 1.93586`), no resolution dependence.
- Templates: full 25 steps, CFG 3.5, euler/simple (try 50 for quality); distilled 15–16 steps (4–16), CFG 1,
  euler or er_sde with simple.
- CFG runs cond and uncond separately: prompts are not padded to a common length.
- References: `ReferenceMultiLatent` rescales each image to about 1 MP (Lanczos, aspect kept), VAE-encodes (centre
  crop to /16), BN-normalizes, and appends the latents to BOTH conditionings. Prompts address them as
  `<ref_image_0>`, `<ref_image_1>`, …. BFL's own pipeline keeps references at native size up to about 16 MP.
- Layout: `Flux3LayoutPrompt` appends `prompt.rstrip() + " " + json.dumps(rows, ensure_ascii=False)` with rows
  `{"id": "obj_N", "bbox": [ymin, xmin, ymax, xmax], "desc"}` on a 0–1000 grid (Python `round`), cited as `<obj_N>`.

## Traps

- Padding is masked as keys in the text encoder, and padding rows take rotary position 1 (transformers 4.57's
  `cumsum(mask) − 1` masked to 1). The DiT attends to those padding rows, so their values matter.
- ComfyUI's attention mask ends at the first 151643 in the sequence, real or padding.
- The inference branches of `QKNorm` and `modulated_norm` are comfy_kitchen fusions (`rms_rope`, `adaln`) of the
  plain-torch training branches; the reference dump uses the plain branches.

## LoRA

- Spellings in circulation: diffusers/PEFT under `transformer.` (`transformer.mode_blocks.image.0.attn.to_q.lora_A.weight`,
  SimpleTuner), BFL names under `diffusion_model.` (ComfyUI's generic map), and kohya `lora_unet_` with underscores.
  `Flux3KeyNames` holds the diffusers↔BFL table: `x_embedders`→`emb_in`, `context_embedder`→`txt_in`,
  `mode_blocks`→`content_mode_blocks`, `single_transformer_blocks`→`single_blocks`, `attn.to_{q,k,v}`→`{q,k,v}_proj`,
  `attn.to_out.0`→`attn_out`, `ff.linear_{in,out}`→`mlp_{in,out}`, modulation `.linear`→`.lin`,
  `timestep_embedder`/`target_timestep_embedder`→`time_in`/`ref_time_in`, `proj_out`→`final_layer.image.linear`.
- Diffusers' `norm_out.linear` holds `[scale | shift]` rows; the BFL `final_layer.image.adaLN_modulation.1` holds
  `[shift | scale]`. A LoRA up matrix on it is reordered before merging.
- ComfyUI addresses the halves of a fused `qkv` as `q_proj`/`k_proj`/`v_proj`. The engine always loads them split, so
  an adapter on `qkv` itself is cut into row thirds instead.
- The SimpleTuner extract is SVD over the weights both checkpoints share, so `ref_time_in` is not in it (default
  targets `to_q`, `to_k`, `to_v`, `to_out.0`). Applied to the full build it approximates the distilled weights on the
  guided path; the steps and CFG are the user's to set. ComfyUI a5e95e6 does not map that extract's diffusers names
  for FLUX.3.

## Engine mapping and open questions

Family id `flux3`; variants `base` and `distilled` (structural on `ref_time_in`). Open: the gated repository's file
layout and hashes (catalog entry has no transformer asset), nvfp4 128-row tile alignment of the real fused `qkv`,
Lanczos vs. the engine's antialiased bicubic for reference resizing, native-size references, and the prompt
upsampler (an LLM pass the engine does not run).

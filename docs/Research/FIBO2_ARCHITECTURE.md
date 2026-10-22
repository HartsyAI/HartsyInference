# Fibo 2.0 (Bria AI)

Weights: CC-BY-NC-4.0 (non-commercial; commercial use needs a Bria agreement). The licence should be surfaced in the catalog
entry when the recipe lands.

## Checkpoints (`Comfy-Org/Fibo-2.0`)

| File | Size | Notes |
|---|---|---|
| `diffusion_models/fibo2_dev_bf16` | 16.45 GB | text-to-image, undistilled |
| `diffusion_models/fibo2_dev_int8_convrot` | 8.41 GB | `int8_tensorwise`, ConvRot 256 |
| `diffusion_models/fibo2_edit_turbo_bf16` | 16.45 GB | edit + turbo |
| `diffusion_models/fibo2_edit_turbo_int8_convrot` | 8.41 GB | |
| `loras/fibo_2.0_turbo_lora_avg_rank_270_bf16` | 1.99 GB | turbo LoRA |
| `text_encoders/qwen3vl_4b_bf16` | 8.88 GB | Qwen3-VL-4B |
| `text_encoders/qwen3vl_4b_int8_convrot` | 4.86 GB | |
| `vae/flux2-vae` | 0.34 GB | 32-channel FLUX.2 VAE |

The checkpoint keeps ComfyUI's native names and fusions. The `briaai/Fibo-2.0-*` HF repos were not read, so the
converter's source files were not available.

## Denoiser

- In/out channels 32, patch 2 (`x_embedder` 128 → 3584). Latent layout is the FLUX.2 VAE's.
- `dim` 3584, 28 heads, head_dim 128, RMSNorm eps 1e-5 on Q and K per head.
- `layers.0..43` main blocks; `noise_refiner.0..1` global-modulation blocks run before the Perceiver.
- Injection blocks (gated cross-attention) at layers 6, 14, 22, 30, 38 (the header's `layers.{i}.cross_attn_gate` keys).
  ComfyUI (`model_detection`) and `Fibo2Config.FromWeights` both read the layers from the keys.
- Attention: `attention.qkv` is fused `[3·dim, dim]` (q, k, v). Injection and Perceiver `to_kv` is fused
  `[2·dim, cap_feat_dim]` with K first, then V.
- FFN: `feed_forward.w13` is fused `[2·H, dim]`, SwiGLU, where H is the zero-padded hidden width. The checkpoint has H = 9728,
  which is the true 8/3·dim width (9557) rounded up to a multiple of 256; the padded rows are zero. **Trap:** the upstream
  diffusers formula `(dim·8//3)//64·64` gives 9536, which does not match this checkpoint. Read H from `w2`'s shape.
- AdaLN: `adaLN_modulation` is `Linear(256 → 4·dim)`, chunked into scale_msa, gate_msa, scale_mlp, gate_mlp. Output is
  `x·(1+scale)`; gates are `tanh`. Final layer is scale-only (`LayerNorm(no affine)·(1+scale)` then `linear`).
- Timestep: `t` is the sampler's sigma, 1 = pure noise (ComfyUI `ModelSamplingFlux.timestep(sigma)` returns sigma), so the
  sampler passes sigma, not 1 − sigma. `t_embedder` embeds `t·1000` as a 256-wide sinusoid, cos half then sin half, with
  `freq_i = 10000^(−i/128)` for i = 0..127 (base 10000, exponent over half the width, no shift), then Linear 256 → 1024,
  SiLU, Linear 1024 → 256. ComfyUI `timestep_embedding`, `DiTUtils.SinusoidalTimestepEmbedding` and the Python reference
  agree.
- Noisy tokens (image) use the embedding of `t·1000`. Clean tokens (gist, reference images) use the embedding of t = 0.
  Main blocks pick the modulation per token by mask. Refiner blocks use global modulation on the image tokens alone.

## Position ids (RoPE θ 256, axes `[32, 48, 48]`, interleaved pairs)

- Gist token i (1-based): `(i, 0, 0)`, `i` in `1..G`.
- Image: frame index `max_gist + 1` = 513 for every gist count, then `(513, row, col)`.
- Reference image k: `(513 + k + 1, row, col)`.
- Axis lengths `(2048, 512, 512)` are the sizes of diffusers' precomputed RoPE tables (`axes_lens`): bounds on the ids, not
  positions. The frame ids in use are 1..G, 513 and 514 + k. ComfyUI's `EmbedND` and the engine's `ZImageRope` compute the
  angles from the ids and have no length, so `ZImageRope` can be reused.

## Text conditioning

- Qwen3-VL-4B hidden 2560. Taps (HF `hidden_states[k]` = input to block k):
  bundles `(9,20,31)`, `(5,16,27)`, `(7,18,29)`, `(9,20,31)`, `(11,22,33)`, `(13,24,35)`; 15 unique layers. Bundles 0 and 3
  are the same taps on purpose: ComfyUI's `BUNDLE_LAYERS` lists `(9,20,31)` twice.
- Each tap is normalised with weightless RMS (`x·rsqrt(mean(x²)+1e-6)`) per token, then the three taps of a bundle are
  concatenated: 3·2560 = 7680 = `cap_feat_dim`. ComfyUI casts the bundles to bf16 before the DiT; the engine keeps F32.
- Bundle 0 → Perceiver (512 learned queries, 2 layers, cross-attention then self-attention then FFN, RMSNorm output).
  Bundles 1..5 → injection blocks in `injection_layers` order.
- **Gist count** from `n`, the token count of the whole template-wrapped prompt (`context.shape[1]`, unpadded), not the
  caption alone (ComfyUI `gist_count`):
  `ratio = clamp((n − 256) / (2048 − 256), 0, 1)`, then
  `count = clamp(ceil((256 + ratio·(512 − 256)) / 64) · 64, 256, 512)`.
  The gist tokens are the first `count` rows of the 512 `latent_queries`, run through the Perceiver against bundle 0
  (ComfyUI `latent_queries[:num_gist]`, engine `PerceiverForward`). They are placed before the image tokens in the stream.
- Template: `<|im_start|>user\n{caption}<|im_end|>\n<|im_start|>assistant\n`, where caption is JSON. An empty prompt uses a
  fixed null caption (negative / CFG). Do not change the template: the output depends on it.

## Sampling

- Flow-matching Euler; `shift` 1.15 (`supported_models.Fibo2.sampling_settings`). `model_base.Fibo2` is `ModelType.FLUX`,
  so ComfyUI applies it as `sigma(t) = e^1.15 / (e^1.15 + 1/t − 1)` (`flux_time_shift`), which is `s·t / (1 + (s − 1)·t)`
  with `s = e^1.15`, not `s = 1.15`.
- Edit-turbo workflow: `Fibo2Scheduler` steps = 4, Euler sampler, default size 1024×704.
- Dev step count and CFG are not in the public workflows. See open items.

## Quantization

- `fibo2_dev_int8_convrot` holds 215 `int8_tensorwise` linears, every one with ConvRot 256: attention, cross-attention and
  FFN in the main, refiner and Perceiver blocks (inputs 3584, 7680, 9728). `x_embedder` (128 inputs), `t_embedder`, every
  `adaLN_modulation` and `final_layer.linear` stay BF16.
- The format is supported (`docs/Research/QUANTIZATION_COMFY_FORMATS.md`), but not for Fibo yet: `Fibo2Recipe` refuses any
  tensor that is not F32, F16 or BF16.

## Open items

- **Sigma schedule; compare against ComfyUI on the GPU.** The timestep input matches ComfyUI: `Fibo2Pipeline` passes
  `SigmaAt(i)` and the model multiplies by 1000. The recipe defaults now hand `s = e^1.15` to
  `FlowMatchEulerDiscreteScheduler` (it was 1.15 before the review fix). Still to confirm: the saved ComfyUI sigma list
  against ours.
- **Dev step count and CFG.** The public workflows give only the edit-turbo values. The diffusers v1 pipeline example uses
  50 steps and guidance 5; that is a v1 source, not confirmed for 2.0. Close it from the Bria model card or a dev template.
- **JSON captions.** ComfyUI re-serialises a JSON caption before templating (`parse_prompt`: compact separators, empty
  values and the score keys dropped). The engine tokenizes the prompt as given, so the tokens and `n` differ unless the
  caption is already in that form.
- **Glyph text images.** ComfyUI renders each `text_render` entry to an image through the Qwen3-VL vision tower
  (`VISION_MARKER`), on by default. It encodes the caption twice: with the images while sigma ≤ 0.9, and with empty markers
  and no images above that. `TEXT_IMAGES_START` is a `start_percent` that maps to sigma 0.9 under ComfyUI's shift, so a
  port should gate on sigma. It is a separate component and is not in the first engine PR; the engine refuses captions
  with `text_render` entries.
- **Reference-image editing** (`context_latents`) and the edit-turbo conditioning node.
- **Real-weight verification.** Pending. The bf16 DiT is 16.45 GB, so it needs a GPU host with enough VRAM and host RAM for
  the checkpoint.

## Validation plan

1. Tiny-config CPU parity of the C# denoiser against `tests/python-reference/fibo2_reference.py` (random weights, saved
   inputs; `Fibo2ParityTests`). Covers RoPE, fused splits, AdaLN, gates, Perceiver, injection and one gist count. Both
   sides port the same reading of ComfyUI's forward, so it catches porting slips, not a misreading they share.
2. Vulkan parity for the same tiny config. Pending.
3. Real-weight run against ComfyUI itself (bf16 dev; int8 once wired) on a GPU host with enough memory for the checkpoint.
   Pending. No real-weight parity is claimed until it runs.

## References

- ComfyUI `87ed5a066d9c` (2026-10-21): `comfy/ldm/fibo2/model.py`, `comfy/text_encoders/fibo2.py`,
  `comfy/supported_models.py` and `comfy/model_base.py` (`Fibo2`), `comfy/model_sampling.py` (`ModelSamplingFlux`),
  `comfy_extras/nodes_fibo2.py`. This is the authoritative implementation for the checkpoint.
- diffusers `de988b7ee77c` (2026-10-20): `src/diffusers/models/transformers/transformer_bria_fibo2.py`. Same structure,
  but with unfused names and a different FFN width formula; treat it as a structural reference only.
- Checkpoint headers: `Comfy-Org/Fibo-2.0` revision `b2bffaf57d48` (2026-10-21).
- The diffusers pipeline for 2.0 does not exist. `pipeline_bria_fibo.py` is the v1 model.

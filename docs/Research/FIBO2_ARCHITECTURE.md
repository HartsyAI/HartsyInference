# Fibo 2.0 (Bria AI)

Source snapshot: ComfyUI `comfy/ldm/fibo2/model.py`, `comfy/text_encoders/fibo2.py`, `comfy_extras/nodes_fibo2.py`,
`comfy/supported_models.py` (`Fibo2`), read 2026-10-21. Checkpoint headers: `Comfy-Org/Fibo-2.0`. Upstream diffusers
`transformers/transformer_bria_fibo2.py` (main, 2026-10-21) for the module structure. The diffusers pipeline for 2.0 does
not exist; `pipeline_bria_fibo.py` is the v1 model. Licence: weights CC-BY-NC-4.0 (non-commercial; commercial use needs a
Bria agreement).

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

The checkpoint keeps ComfyUI's native names and fusions. The `briaai/Fibo-2.0-*` HF repos are gated for this account, so
the converter's source files cannot be read there.

## Denoiser

- In/out channels 32, patch 2 (`x_embedder` 128 → 3584). Latent layout is the FLUX.2 VAE's.
- `dim` 3584, 28 heads, head_dim 128, RMSNorm eps 1e-5 on Q and K per head.
- `layers.0..43` main blocks; `noise_refiner.0..1` global-modulation blocks run before the Perceiver.
- Injection blocks (gated cross-attention) at 6, 14, 22, 30, 38. The checkpoint header confirms 14, 22, 30; 6 and 38 come from
  ComfyUI's `injection_layers` default.
- Attention: `attention.qkv` is fused `[3·dim, dim]` (q, k, v). Injection and Perceiver `to_kv` is fused
  `[2·dim, cap_feat_dim]` with K first, then V.
- FFN: `feed_forward.w13` is fused `[2·H, dim]`, SwiGLU, where H is the zero-padded hidden width. The checkpoint has H = 9728,
  which is the true 8/3·dim width (9557) rounded up to a multiple of 256; the padded rows are zero. **Trap:** the upstream
  diffusers formula `(dim·8//3)//64·64` gives 9536, which does not match this checkpoint. Read H from `w2`'s shape.
- AdaLN: `adaLN_modulation` is `Linear(256 → 4·dim)`, chunked into scale_msa, gate_msa, scale_mlp, gate_mlp. Output is
  `x·(1+scale)`; gates are `tanh`. Final layer is scale-only (`LayerNorm(no affine)·(1+scale)` then `linear`).
- Timestep: sinusoidal embedding of width 256 → MLP (1024) → 256 via `t_embedder`. Noisy tokens get `t·1000` (t = 1 is pure
  noise). Clean tokens (gist, reference images) get the embedding of t = 0. In the main blocks each token selects its
  modulation by mask.
- Refiner blocks run global modulation on the image tokens alone, before the Perceiver.

## Position ids (RoPE θ 256, axes `[32, 48, 48]`, interleaved pairs)

- Gist token i (1-based): `(i, 0, 0)`, `i` in `1..G`.
- Image: frame index `max_gist + 1` = 513 for every gist count, then `(513, row, col)`.
- Reference image k: `(513 + k + 1, row, col)`.
- Axis lengths `(2048, 512, 512)`. The engine's `ZImageRope` takes arbitrary position ids and can be reused.

## Text conditioning

- Qwen3-VL-4B hidden 2560. Taps (HF `hidden_states[k]` = input to block k):
  bundles `(9,20,31)`, `(5,16,27)`, `(7,18,29)`, `(9,20,31)`, `(11,22,33)`, `(13,24,35)`; 15 unique layers.
- Each tap is normalised with weightless RMS (`x·rsqrt(mean(x²)+1e-6)`) per token, then the three taps of a bundle are
  concatenated: 3·2560 = 7680 = `cap_feat_dim`.
- Bundle 0 → Perceiver (512 learned queries, 2 layers, cross-attention then self-attention then FFN, RMSNorm output).
  Bundles 1..5 → injection blocks 6, 14, 22, 30, 38 in order.
- Gist count from the text length in tokens: linear in `[256, 2048] → [256, 512]`, rounded up to a multiple of 64, clamped.
  The gist tokens are placed before the image tokens in the stream.
- Template: `<|im_start|>user\n{caption}<|im_end|>\n<|im_start|>assistant\n`, where caption is JSON. An empty prompt uses a
  fixed null caption (negative / CFG). Do not change the template: the output depends on it.

### Open (not implemented, deliberately deferred)

- **Glyph text images.** Official ComfyUI renders each `text_render` entry to an image through the Qwen3-VL vision tower
  (`VISION_MARKER`). It is on by default and switched off above sigma 0.9 (`TEXT_IMAGES_START`). This is a separate
  component and the first engine PR does not include it. Captions with `text_render` entries therefore will not match
  ComfyUI until it lands.
- Reference-image editing (`context_latents`) and the edit-turbo conditioning node.

## Sampling

- Flow-matching Euler; `shift` 1.15 (`supported_models.Fibo2.sampling_settings`).
- Edit-turbo workflow: `Fibo2Scheduler` steps = 4, Euler sampler, default size 1024×704.
- Dev step count and CFG are not in the public workflows. The diffusers v1 pipeline example uses 50 steps and guidance 5;
  that is a v1 source, not confirmed for 2.0. Treat dev defaults as **open** until checked.

## Quantization

- `int8_tensorwise` + ConvRot 256 is already supported (`docs/Research/QUANTIZATION_COMFY_FORMATS.md`). Every Fibo linear
  input dimension is a multiple of 256 (3584, 7680, 9728), so the rotation applies to all of them.

## Validation plan

1. Tiny-config CPU parity of the denoiser against a standalone Python reference of the ComfyUI forward (random weights,
   saved inputs). Covers RoPE, fused splits, AdaLN, gates, Perceiver, injection, gist masking.
2. Vulkan parity for the same tiny config.
3. Real-weight run (bf16 dev and int8) on the 4090 box, not this one: the bf16 DiT is 16.45 GB against an 8 GB card and
   about 12 GB of usable host RAM.

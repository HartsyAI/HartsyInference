# MiniMax-H3-Image architecture dossier

Phase 0 of the MiniMax-H3-Image build plan. This note records what was read from the real checkpoint headers and the
ComfyUI reference source, what the plan got wrong, and what is still open. It contains no engine code changes. Status
and parity evidence belong in [MODEL_STATUS_IMAGE.md](../Checklists/MODEL_STATUS_IMAGE.md) and
[PARITY_VERIFICATION.md](../Checklists/PARITY_VERIFICATION.md) once weights have run. Nothing here has run on a GPU or
on real weights.

## Sources and pinned commits

| Source | Revision read |
|---|---|
| HartsyInference | `1b98110f53dd4d4d3962d93c6dfbe48e5be58400` (branch `ccr-e3e9c4fe-urvb3j`) |
| SwarmUI-HartsyInference-Backend | `41745a99fc50c6dc159ed22de5978d6910cfbecd` |
| SwarmUI core | pinned `70b9a9c` is **not reachable** from the public `Stability-AI/StableSwarmUI` repo. Head read: `6071f8f685dfdde8a8cab1ae5feb06e5865d70fc`. See [SwarmUI core](#swarmui-core). |
| ComfyUI | `989d93d` (v0.45.0), files fetched raw from `Comfy-Org/ComfyUI` |
| Checkpoints | `Comfy-Org/MiniMax-H3-Image` at `main` (headers and LFS oids) |
| Original model | `MiniMaxAI/MiniMax-H3-Image`: `vae/config.json`, `transformer/config.json`, `pdd_config.json` |
| Video reference | `Comfy-Org/MiniMax-H3`, `diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors` (header only) |

## Files

Full sha256 values are the LFS oids from `https://huggingface.co/api/models/Comfy-Org/MiniMax-H3-Image/tree/main?recursive=1`.
Sizes are the byte sizes from the same listing.

| File | Bytes | sha256 |
|---|---|---|
| `diffusion_models/minimax_h3_image_pruned_bf16.safetensors` | 40234657162 | `a767f5325db019588b3491a93d80288d336c826a7aeb87bec67675e554367222` |
| `diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors` | 20979314802 | `83eb140f1ff95c068f8a6886e6d3f3d0d4ab2e07bb2fe6d2ef992a51146ee622` |
| `diffusion_models/minimax_h3_image_pruned_w4a8.safetensors` | 12549794530 | `517cc19b85348386f4df7c806dce80d839d0899e7cdaa3e1dbb09de964381dbb` |
| `vae/minimax_h3_image_vae_fp16.safetensors` | 1168505312 | `d80e9c2e5cfc9844869961ff5a58c9a9e52182df6bfb30b26018dd7f348721af` |
| `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 15687142551 | `35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6` |
| `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | 27141342152 | `bc2ced0fbea64757fa9acddccfc0b3f4819d1dcf1da6c124d690d368be283923` |
| `text_encoders/qwen3.5_9b_w4a8_gptq.safetensors` | 7343591088 | `4cc6a1543ad2061806da46f35f90cadb4801dcba246c44782cf42ec0feb6407c` |

All seven hashes match the plan's 12-character prefixes.

## Header facts (read from the files)

The header was fetched by range request (8-byte length, then the JSON). No weight bytes were downloaded.

| Build | Tensors | Dtypes | Notes |
|---|---|---|---|
| bf16 transformer | 578 | F32 6, BF16 471, F16 51, U8 50 | 50 `comfy_attention.config` U8 blobs |
| int8 ConvRot transformer | 978 | F32 206, BF16 271, F16 51, U8 250, I8 200 | 200 `*.comfy_quant` companions |
| w4a8 transformer | 1378 | F32 406, BF16 271, F16 51, U8 250, I8 200, F8_E4M3 200 | 200 `*.comfy_quant` companions |
| VAE fp16 | 416 | F16 416 | no `latents_mean`/`latents_std`, no metadata |

Confirmed shapes, all three transformer builds unless noted:

- 50 blocks (`blocks.0` to `blocks.49`), 2 token-refiner blocks, `hidden 5376`, `condition_proj.weight [5376, 5120]` (BF16).
- `video_patch_proj.weight [5376, 96]` (F32).
- `adaln_t_table [1025, 8]` (F32).
- `final_layer.video_out.weight [576, 5376]` and `bias [576]` (F32). 576 = 6 heads x 96 rows.
- `blocks.N.adaln_proj.linear.weight [96768, 8]` (F16) and `bias [96768]` (BF16). The row count matches video.
- `final_layer.adaln_proj.linear.weight [10752, 8]` (F16) and `bias [10752]` (BF16).
- `rope.inv_freq [16]` (F32). Not in the plan; it matches `rope_inv_freq_len: 16` in `transformer/config.json`.
- `blocks.N.attn.comfy_attention.config` is U8 `[37]` on every block of every build.
- No `audio_patch_proj.*` and no `final_layer.audio_out.*` keys.
- Metadata key `config` = `{"transformer": {"pdd_steps": 6, "pdd_sigmas": [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0]}}`, identical in all three builds.

VAE (fp16, one file):

- `post_quant_conv.weight [24, 24, 1, 1]` (F16, rank 4) and `bias [24]`.
- `decoder.proj_out.weight [768, 1024]`, `bias [768]`. 768 = 3 x 16 x 16.
- `decoder.transformer_blocks.*`: 24 blocks. `decoder.transformer_blocks.0.scale1 [1024]`.
- `decoder.register_tokens`, `decoder.mask_token`.
- `encoder.conv_in.weight [128, 3, 3, 3, 3]`.
- `quant_conv.weight [48, 48, 1, 1, 1]` (rank 5, as the plan says).

### Image build compared with the H3 video build

Video reference: `minimax_h3_fl2va_pruned_bf16.safetensors`, 532 tensors. Image bf16: 578 tensors.

- Removed for image: `audio_patch_proj.weight`, `audio_patch_proj.bias`, `final_layer.audio_out.weight`, `final_layer.audio_out.bias` (4 tensors).
- Added for image: 50 `blocks.N.attn.comfy_attention.config` blobs.
- 532 - 4 + 50 = 578, which accounts for every tensor.
- Shape or dtype differs: `final_layer.video_out.weight` `[576, 5376]` vs `[96, 5376]`; `final_layer.video_out.bias` `[576]` vs `[96]`.
- Dtype only: `blocks.N.adaln_proj.linear.bias` BF16 (image) vs F16 (video) for all 50 blocks; `final_layer.adaln_proj.linear.bias` BF16 (image) vs F16 (video). **The second one is not in the plan.**
- Everything else compares equal in name, shape and dtype.

The plan's "26 key groups identical" was not reproduced. The grouping used here is per-template, so the count depends on the
grouping rule. The tensor-level accounting above is exact.

## Corrections to the plan

| Plan claim | Finding | Action |
|---|---|---|
| `adaln_proj.linear.bias` BF16 in image, F16 in video | Confirmed for all 50 blocks. Also true for `final_layer.adaln_proj.linear.bias`, which the plan missed. | Read these as BF16 storage; never via float pointers. Add the final-layer bias to the trap list. |
| Picture label `Picture %d: ` (ComfyUI image node) vs engine `<Picture %d>: ` | ComfyUI's image node passes `picture_label="Picture %d: "` (`comfy_extras/nodes_minimax_h3.py:739`). ComfyUI's tokenizer default `<Picture %d>: ` is the video `fl2va` label. The engine uses `<Picture {n}>: ` (`src/HartsyInference.Diffusion/Models/TextEncoders/MiniMaxH3TextEncoding.cs:188`). | **The engine's image path must emit `Picture N: ` without angle brackets.** Phase 5b. |
| Text encoder is Qwen3-VL-32B truncated to layer 50 | Confirmed for `MiniMaxH3Image`: `clip_target` uses `MiniMaxH3Tokenizer` and `te()`, which read the last layer. | None. |
| Five-layer text connector `[13, 19, 25, 31, 36]` | Belongs to `p-h3-image` (`PH3ImageTEModel`), a different family with Qwen3-VL-4B. It is not in this checkpoint's path. | Do not port it. |
| Output head arithmetic "mean of the heads it spans" | The exact arithmetic is `block0 + sum over k in [first, stop) of (dt[k] / S) * block[k]`, with `first = max(start, 1)`. Block 0 has weight 1. `S` is the sum over all spanned intervals, including block 0's own interval. Details in [Output heads](#output-heads). | Implement exactly this. Phase 2b. |
| `post_quant_conv` rank 4 `[24, 24, 1, 1]` | Confirmed. ComfyUI loads it as `Conv2d(24, 24, 1)` and squeezes the temporal axis. | Phase 3. |
| "Latent statistics are not in the VAE file" | Confirmed. They are in the original `vae/config.json`. | See [Latent statistics](#latent-statistics). |
| Still image VAE-encoded as one frame, "encode to a latent" | One frame, yes. But ComfyUI **samples** the posterior, `mean + exp(0.5 * logvar) * noise`, with noise from `torch.Generator().manual_seed(42)`. It does not take the mean. | Phase 5b must replicate sampling, or record why it does not. See [Still-image encode](#still-image-encode). |
| "Decoder 16 heads, RoPE ratio 0.5, theta 100" | Confirmed by the original `vit_decoder_kwargs` and by `MiniMaxH3ImageVAE`: 16 heads, dim_head 64, `rope_dim_ratio=0.5`, `rope_dims=2`, `rope_theta=100`, 24 layers. | None. |
| `vae_token_drop: 0`, `vae_clip_length: 17`, tile 256, overlap 64 | Confirmed in the original config. ComfyUI's image VAE uses the same tile defaults (256 and 64). ComfyUI builds its image VAE with the base-class default `token_drop=3`, but the single-frame path does not chunk temporally, so `token_drop` is not reached for stills. | Set the 2D baseline to 0 as the plan says. Verify in phase 3. |
| Swarm core line 223 requires `audio_patch_proj.weight` | Could not check against the pinned commit. The head classifier (216 lines) has no H3, Wan or Flux rules. | Re-check in phase 6. See [SwarmUI core](#swarmui-core). |
| Prompts truncated to 4096 tokens | Not found in the ComfyUI files read. | Unverified. Do not implement a cap on this basis. |
| "The model output is negated" | Confirmed: `return -out` in `MiniMaxH3ImageModel._forward`. | Assert the sign in the phase 2 parity test. |
| ComfyUI MiniMax LoRA key map | Confirmed. See [LoRA](#lora). The plan's question is answered. | Phase 5c. |

## Latent statistics

The original `vae/config.json` has `latents_mean` and `latents_std`, 24 values each. Comparison:

- Original config mean vs engine `DefaultLatentsMean` (`MiniMaxH3VideoVaeConfig.cs:378`): **exactly equal**.
- Original config std vs engine `DefaultLatentsStd`: equal within float32 rounding (max difference 5.6e-17, from the
  float32 literal in the C# source).
- Original config vs ComfyUI `LATENTS_MEAN` and `LATENTS_STD` (`comfy/ldm/minimax/vae.py`): **exactly equal**.

**Answer: the image VAE uses the video defaults.** The engine's default fallback is correct for this checkpoint. ComfyUI
applies the constants only when the file lacks the tensors (`comfy/sd.py:1072-1073`, `setdefault`).

## Output heads

Source: `comfy/ldm/minimax/model.py`, `FinalLayer.forward` and `_pdd_head`. The image model calls `FinalLayer` with
`grids = (pdd_sigmas,)` and `sample_sigmas` from the sampler's transformer options.

Step selection, for a sampler step with `sigma` and the sampler's next sigma `sigma_next`:

```
i        = argmin_j |sample_sigmas[j] - sigma|
sigma_nx = sample_sigmas[min(i + 1, len - 1)]
start    = argmin_k |grid[k] - sigma|           (first minimum, k in 0..N)
stop     = argmin_k |grid[k] - sigma_nx|        (first minimum)
start    = min(start, N - 1)
stop     = max(stop, start + 1)
```

Here `N = pdd_steps = 6` and `grid` is `pdd_sigmas`, descending from 1.0 to 0.0.

Head arithmetic, for rows reshaped to `[N, out, hidden]` and bias to `[N, out]`:

```
dt[k]   = grid[k] - grid[k + 1]          for k in start .. stop - 1
w[k]    = dt[k] / sum(dt)                 (the sum includes dt[start] even when start == 0)
first   = max(start, 1)
W_eff   = rows[0] + sum over n in first .. stop - 1 of w[n] * rows[n]
b_eff   = bias[0] + sum over n in first .. stop - 1 of w[n] * bias[n]
out     = h @ W_eff^T + b_eff
```

Consequences:

- Block 0 is always included once with weight 1. Blocks 1 to 5 are offsets.
- When `start == 0`, the offset weights are `dt[1..]/S` where `S` includes `dt[0]`. They do not sum to 1. The reference does this, so do we.
- The official schedule lands on grid points, so each step normally spans one interval. For `start = n >= 1`, `stop = n + 1`: `W_eff = rows[0] + rows[n]`, since the single weight is 1.
- The first step, `start = 0`, `stop = 1`, has an empty offset range, so `W_eff = rows[0]` (block 0 only).
- The spanning logic must still match the reference exactly, because multi-interval steps can occur with other step counts.

## Timestep, reference pinning and sign

From `MiniMaxH3ImageModel._forward`:

- `sigma = timestep / 1000` (model_base passes sigma x 1000).
- `t = 1 - sigma`. Text and target segments use `t`.
- Reference-image segments use `max(t, VISUAL_COND_TIMESTEP)` with `VISUAL_COND_TIMESTEP = 0.999`.
- Only one modality tag is used for the image model. The adaLN projection still has rows for three modalities; the image
  model never selects the audio row.
- The output is `-out[..., :orig_h, :orig_w]`, unpatchified from one frame with `patch_size (1, 2, 2)` and 24 channels.
- Batch size must be 1. ComfyUI raises otherwise.

## Sampling

`comfy/ldm/minimax/sampling.py` (`MiniMaxH3ImageModelSampling`): the sigma is piecewise linear in the grid index, the flow
shift is unused (`sampling_settings = {}` for `MiniMaxH3Image`). The grid is stored ascending and interpolated. The
official server accepts exactly 6 steps and guidance 1 (per the model card). Those values are the plan's, not read from
ComfyUI.

## Packed layout and RoPE positions

Source: `PackedLayout` in `comfy/ldm/minimax/model.py`.

- Spatial frame grid, `_frame_grid(h, w)`: one token per 2 x 2 latent patch. Coordinates are
  `(arange(dim // 2) * (ratio / n) + (1 - ratio) / 2) * 32` with `ratio = dim / sqrt(h * w)` and `n = dim // 2`, computed per
  axis for that frame's own latent size.
- Text: time `t = 0 .. text_len - 1`, spatial `(0, 0)`.
- Reference image `i` (0-based): time `text_len + i`, spatial from its own latent grid. Each reference advances the
  cursor by exactly 1.0.
- Target image: time `text_len + n_refs`, spatial from the target latent grid. The target is always the last segment.
- Sequence order: `[text | reference 1 | ... | reference n | target]`.

Latent dims are rounded up to even before layout (`extra_conds` uses `(dim + 1) // 2 * 2`). With pixel sides multiples of
32, the latent sides are multiples of 2, so this does not change any valid size.

Token count: one token covers 32 x 32 pixels. A 2048 x 2048 target is 64 x 64 = 4096 tokens. A 4096 x 4096 target is 16384.

## Image VAE

Source: `comfy/ldm/minimax/vae.py` (`H3ImageVAE`, `MiniMaxH3ImageVAE`), selection in `comfy/sd.py` around lines 1060-1080.

- Selection: `decoder.proj_out.weight` with 768 rows selects a single-frame image decoder. `decoder.x_embedder.weight`
  with 1024 columns selects `MiniMaxH3ImageVAE`. The p-h3-image VAE has 2048 (32 heads) and is not this one.
- Encoder: the video encoder (`encoder.conv_in [128, 3, 3, 3, 3]`), `still_frame=0`.
- Decoder: `ViT3DDecoder(patch_size=16, patch_size_t=1, num_layers=24, heads=16, dim_head=64, rope_dim_ratio=0.5,
  rope_dims=2, rope_theta=100)`, 4 register tokens, `post_quant_conv = Conv2d(24, 24, 1)`.
- Decode accepts only one latent frame. Latents are de-normalized with the statistics, then decoded. Pixel
  normalization uses ImageNet mean and std, folded into `proj_out` as in video.
- Tiles: 256 pixels, overlap 64 minimum, as in video.

### Still-image encode

`H3ImageVAE.encode`:

```
moments = _adaptive_encode(normalize_pixels(x))   # x: [B, 3, 1, H, W]
mean, logvar = moments.chunk(2, dim=1)
noise  = torch.randn(mean.shape, generator=torch.Generator().manual_seed(42))
latents = mean + exp(0.5 * clamp(logvar, -30, 20)) * noise
latents = (latents - latents_mean) / latents_std
```

So a still is one frame, not padded to a clip. It is a **posterior sample** with a fixed seed, not the mean. The engine
must match this for reference latents to agree. Matching torch's CPU Mersenne Twister normal sampling bit-for-bit is a
decision for phase 5b. The plan notes that matching seeds does not mean matching RNG implementations, so the parity test
should feed saved reference noise.

## Text encoder and tokens

Source: `comfy/text_encoders/minimax.py`.

- Raw token ids, no chat template, no special tokens.
- Image tokens: `VISION_START` (151652), vision block, `VISION_END` (151653).
- ComfyUI image node sequence: for each reference, `"Picture N: "` text, then the vision block; then the prompt text.
- The prompt text is passed through unchanged. The user writes `<Picture 1>` inside the prompt itself.
- `MiniMaxH3Tokenizer` raises `ValueError` if a text segment does not fit in one token batch. Any cap is therefore an
  upstream tokenizer setting. The 4096 figure is not confirmed (see the table above).

Engine, for comparison: `MiniMaxH3TextEncoder.cs:86` loads 50 layers. `MiniMaxH3TextEncoding.cs:188` writes
`<Picture {n}>: ` before each image. Phase 5b must change this label for the image path only, and keep the video path
unchanged.

## Model and VAE detection

- ComfyUI `comfy/model_detection.py:401-445`: an H3 checkpoint is recognized by `video_patch_proj.weight` and
  `condition_proj.weight`. It is an image build when `audio_patch_proj.weight` is absent.
- `latents_dim = video_out.rows // (4 * pdd_steps)`. For 576 rows and 6 steps this is 24. Without dividing by the head
  count it would be 144.
- The `transformer` dict from metadata is merged into the DiT config (`model_detection.py:434`). That merge is how
  `pdd_steps` and `pdd_sigmas` reach the model. Note that the whole dict is merged, not just the two keys.

Engine, for comparison: `ModelArchitectureDetector.cs:21-24` requires `audio_patch_proj.` for the H3 rule, so the image
build matches nothing. `InferenceEngine.cs:735` maps `MiniMaxH3` to `minimax-h3` only.

## LoRA

Source: `comfy/lora.py`, `model_key_map` branch for `comfy.model_base.MiniMaxH3`. The image model is a subclass, so the
same branch applies.

- Keys: `diffusion_model.<name>.weight` maps to `<name>`. The key map is the same for video and image.
- PDD rule: when `pdd_steps > 1`, a single-head LoRA on `final_layer.video_out.weight` (or `audio_out`) patches
  rows `0 .. out` (head 0 only), the base of every interval head. A LoRA that stacks heads is the video-side feature the
  plan says to refuse.

## SwarmUI core

- Pinned `70b9a9c` is not reachable from `Stability-AI/StableSwarmUI`. A blobless clone at the head (`6071f8f`) has the
  file `src/Text2Image/T2IModelClassSorter.cs` (216 lines) but no H3, Wan or Flux rule.
- The plan's citation (line 223, `audio_patch_proj.weight` requirement) could not be checked against the pinned commit.
- The extension's `41745a9` is the pinned backend commit. It references `T2IModelClassSorter.CompatMiniMaxH3`, so the
  core class exists at the revision the extension was built against.
- Phase 6 must confirm core's current class list before registering anything.

## Engine state, checked by grep

| Gap (plan) | Location read | Status |
|---|---|---|
| 1. Detection needs audio | `ModelArchitectureDetector.cs:21-24` | Confirmed |
| 2. Transformer requires audio | Not re-read in phase 0 | Plan claim, unchecked |
| 3. Output heads | Not re-read in phase 0 | Plan claim, unchecked |
| 4. VAE 3D only | `MiniMaxH3VideoVaeConfig.cs:244` (`PatchDim` check) | Confirmed |
| 5. No image recipe | Not re-read in phase 0 | Plan claim, unchecked |
| 6. Swarm classification | See [SwarmUI core](#swarmui-core) | Could not confirm |
| Picture label | `MiniMaxH3TextEncoding.cs:188` | Confirmed mismatch with image node |
| Layer 50 tap | `MiniMaxH3TextEncoder.cs:86` | Confirmed |
| comfy_attention blob | No engine handling found | Confirmed gap |
| Qwen3.8 runtime | Grep for `qwen3\.8`, `qwen38`, `qwen3_8`: only Qwen3-8B matches | **No** |

## Feature matrix (from the plan, unchanged)

| Feature | Official | ComfyUI | Engine today | Phase | Outcome |
|---|---|---|---|---|---|
| Text-to-image | Yes | Yes | No | 4 | Implement |
| Fixed 6 steps, guidance 1 | Yes | Yes | No | 4 | Enforce; refuse other values |
| Negative prompt | Rejected | Not used | n/a | 4 | Refuse |
| Sampler and scheduler choice | None | Stock schedulers land on the grid | n/a | 4 | Solver-owned; refuse any name |
| Any size, multiples of 32, about 4 to 20 MP | Yes | Yes | No | 4 | Implement; default 2048x2048 |
| Minimum-area upscale then Lanczos | Yes (server) | No | No | 5a | Implement |
| Edit with one reference | Yes | Yes | No | 5b | Implement |
| Edit with up to 16 references | Yes | Yes | No | 5b | Implement |
| bf16 transformer | Yes | Yes | Loader exists | 4 | Wire |
| int8 ConvRot transformer | Repack | Yes | Loader exists | 4 | Wire; provisional default |
| w4a8 transformer | Repack | Yes | Codec exists | 4 | Wire |
| nvfp4 and int8 text encoder | Repack | Yes | Yes | 4 | Wire |
| Tiled VAE decode | Yes | Yes | 3D only | 3 | Extend to 2D |
| Latent preview | n/a | Yes | Video only | 4 | Wire |
| Prompt weighting | No | Via tokenizer | Video: CondScale | 4 | Reuse; pin in ledger test |
| LoRA | Not stated | Key map in `comfy/lora.py` | Video merges exist | 5c | Implement; bf16 base first |
| Prompt enhancer, Qwen3.5-9B | No | Recommended | Qwen3.5 runtime exists | 5d | Implement, off by default |
| Prompt enhancer, official (Qwen3.8-27B) | Yes | No | No Qwen3.8 runtime | 5d | Refuse and record unless the runtime loads it |
| Block streaming, component placement, phase unload | n/a | n/a | Video has them | 4 | Declare only what the pipeline wires |
| Multi-GPU transformer sharding | Yes (Ulysses) | n/a | Video has it | 8 | After single-GPU correctness |
| Img2img with denoise strength | No | Not documented for image | n/a | 4 | Refuse |
| Mask inpainting | No | Not documented for image | n/a | 4 | Refuse |
| ControlNet, IP-Adapter, regional prompts, refiner | No | No | n/a | 4 | Refuse |
| Batch size above 1 | No | Raises | n/a | 4 | Loop one image at a time or refuse |

## Open items

Not answered in phase 0:

- Official ComfyUI text-to-image and edit workflow templates (plan step 6). Not fetched. The templates live outside the
  ComfyUI core repo.
- The 4096-token prompt truncation. Not found in the ComfyUI files read.
- The SwarmUI core line-223 classifier at the pinned commit `70b9a9c`. The commit is not reachable.
- Whether the engine can load the official enhancer. Answer: no Qwen3.8 runtime was found.

Questions only a GPU can answer:

- Output parity per quant (bf16, int8 ConvRot, w4a8) against ComfyUI at the same build and precision.
- Speed per step, model load time, and peak VRAM at 2048x2048 and 4096x4096.
- Whether int8 ConvRot quality is acceptable as the default.
- Whether replicating torch's CPU posterior sample (seed 42) is needed for reference images to match, or whether
  reference parity can be checked with saved latents.

## Decisions for the owner

1. **Still-image encode sampling.** The reference samples the posterior with seed 42. Replicating torch's CPU generator
   exactly is the only way to match ComfyUI bit-for-bit. The alternative is to use the mean and document the difference.
2. **Enhancer.** Qwen3.8 is not supported by the engine. The choice is Qwen3.5-9B only, or a Qwen3.8 runtime first.
3. **Picture label.** The image path uses `Picture N: ` (no brackets). Confirm this is wanted, since it differs from the
   video label.

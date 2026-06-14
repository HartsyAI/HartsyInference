# Chroma Radiance & Zeta-Chroma (Pixel-Space) Architecture — Research Notes

> **Status:** Web-verified vs ComfyUI implementation; bracketed/⚠ items are **validation-gated** (no checkpoint diffed locally yet) | **Last Updated:** 2026-06-11 | **Needed Before:** `ChromaRadianceTransformer`, `ChromaRadiancePipeline`, `ZetaChromaTransformer`, `ZetaChromaPipeline`, converter/key-mapper support
>
> **Sources of truth:**
> - HuggingFace: [`lodestones/Chroma1-Radiance`](https://huggingface.co/lodestones/Chroma1-Radiance) (pixel-space Chroma), [`lodestones/Zeta-Chroma`](https://huggingface.co/lodestones/Zeta-Chroma) (pixel-proto checkpoints, ~13 GB, mid-pretraining)
> - ComfyUI PR [#9682](https://github.com/comfyanonymous/ComfyUI/pull/9682) — `comfy/ldm/chroma_radiance/` (NeRF head, conv patchify)
> - ComfyUI PR [#12709](https://github.com/comfyanonymous/ComfyUI/pull/12709) — `NextDiTPixelSpace` (Zeta-Chroma pixel variant of the Z-Image/NextDiT S3-DiT)
> - Existing in-repo references: [FLUX_ARCHITECTURE.md](FLUX_ARCHITECTURE.md) (Chroma backbone is documented in `ChromaTransformer`/`ChromaConfig` doc comments), `docs/Research/Z_IMAGE`-related notes in `ZImageConfig`/`ZImageTransformer` doc comments

## Summary

Two **pixel-space** (VAE-free) flow-matching image models from Lodestone Rock:

1. **Chroma Radiance** — the existing Chroma backbone (Flux-derived, 19 double + 38 single blocks) with the VAE removed. Input is raw RGB in `[-1, 1]` patchified by a `Conv2d(3→3072, kernel 16, stride 16)`; output is produced by a per-patch hypernetwork "NeRF head" that replaces `final_layer`. The model predicts **x0** (the clean image), not velocity.
2. **Zeta-Chroma** — **NOT** a Chroma MMDiT despite the name. It is the **Z-Image S3-DiT** (single-stream NextDiT, 30 layers, hidden 3840) retrained for x0-prediction directly in pixel space: `x_embedder` consumes 32×32 RGB patches (in-dim 3072 = 3·32²) and the final layer projects back to pixels. Mid-pretraining — **everything below is validation-gated** for this model.

Both share the same sampling skeleton: flow-match Euler t: 1→0, model output x0, converted to velocity `v = (x_t − x0) / max(t, ε)` before CFG combine and the Euler step (matches ComfyUI's conversion path).

---

## Chroma Radiance (`lodestones/Chroma1-Radiance`)

### Backbone (identical to Chroma — all keys unchanged)

```
depth                19    double blocks (double_blocks.{i})
depth_single_blocks  38    single blocks (single_blocks.{i})
hidden_size          3072
num_heads            24    (head_dim 128)
mlp_ratio            4     (12288)
context_in_dim       4096  (T5-XXL only — no CLIP)
axes_dim             [16, 56, 56], theta 10000 (FluxPosEmbed)
distilled_guidance_layer   approximator: in_dim 64, depth 5, hidden 5120
mod_index_length     344   (3·38 + 12·19 + 2)
```

`distilled_guidance_layer.*`, `double_blocks.*`, `single_blocks.*`, `txt_in.*` keys are byte-compatible with classic Chroma — `ChromaCheckpointConverter` handles them unchanged.

### Input: conv patchify, no VAE

- `img_in_patch` = `Conv2d(3 → 3072, kernel 16, stride 16)` over raw RGB in `[-1, 1]`. Output flattened to `[B, (H/16)·(W/16), 3072]`.
- Keys: `img_in_patch.weight [3072, 3, 16, 16]`, `img_in_patch.bias [3072]`. There are **no** `img_in.*` keys.
- Resolution must be padded to a multiple of 16 (pipeline pads up, crops the output back).
- Optional `_orig_mod.` key prefix (torch.compile artifact) must be stripped on load.

### Output: "NeRF head" (replaces `final_layer`)

Per 16×16 patch, the transformer's output img token (3072) **generates the weights** of a tiny per-patch GLU MLP that refines the 256 per-pixel embeddings of that patch:

1. **`nerf_image_embedder`** — `Linear(3 + 64 → 64)`, run in **FP32**. Per-pixel input = `[r, g, b]` from the *noisy input image* concatenated with 64 DCT/cosine positional features. `max_freqs = 8` → 8² = 64 separable cosine-basis features per pixel position within the 16×16 patch (⚠ exact basis: `cos(π·u·y)·cos(π·v·x)` over `linspace(0,1,16)` positions — validation-gated). Key: `nerf_image_embedder.embedder.0.{weight [64, 67], bias [64]}`.
2. **`nerf_blocks.{0..3}`** (`nerf_depth = 4`) — each block has:
   - `param_generator` = `Linear(3072 → 49152)`; 49152 = 3 × 64×(64×4). The output is split into **three** weight matrices for a per-patch GLU MLP: gate `W1 [64→256]`, value `W2 [64→256]`, out `W3 [256→64]`.
   - `norm.scale` = RMSNorm(64) scale.
   - Forward (per patch, x = `[256, 64]` pixel embeddings): `xn = RMSNorm(x)`, `x = x + W3( silu(xn·W1ᵀ) ⊙ (xn·W2ᵀ) )`.
   - ⚠ **The exact split order / transposition of the 49152-element chunk is validation-gated** (we assume `[W1, W2, W3]` contiguous, each chunk row-major `[out, in]`).
3. **Final head — two variants, detected by key:**
   - **Variant A (conv)**, present when `nerf_final_layer_conv.norm.scale` exists: RMSNorm(64), fold pixels back to `[B, 64, H, W]`, then `Conv2d(64 → 3, kernel 3, padding 1)`.
   - **Variant B (linear)**: `nerf_final_layer.norm.scale` + `nerf_final_layer.linear.weight [3, 64]` applied per pixel (≡ 1×1 conv).
4. **Forward**: unfold the original noisy RGB into 16×16 patches → `[B, N, 256, 3]`; embed (+pos) → `[B, N, 256, 64]`; for each patch, drive the 4 GLU blocks with its transformer img token (processed in tiles of 32 patches to bound memory); fold; final head → **x0 prediction** `[B, 3, H, W]`.

⚠ Whether the last-two-row modulation-table final norm (`ChromaAdaLayerNormContinuousPruned`) is applied before the NeRF head is validation-gated; we skip it (head replaces `final_layer` entirely).

### Sampling

- **Model predicts x0.** Flow-match Euler t: 1→0, static shift/mu = **1.0**, default **50 steps**, CFG **3.5** two-pass.
- Convert to velocity: `v = (x_t − x0_pred) / max(t, ε)`, then CFG-combine on v, then Euler step (matches ComfyUI; equivalently one could step on x0).
- Pixel range `[-1, 1]`; output straight to RGB bytes via `ImagePostProcessor` (no VAE decode). Latent previews are identity (the latent *is* the image).

### Detection

Chroma family (`distilled_guidance_layer.norms.0.scale`) **plus** `nerf_blocks.0.norm.scale` present → Radiance. `img_in.weight` present instead of `img_in_patch.weight` → classic Chroma.

---

## Zeta-Chroma (`lodestones/Zeta-Chroma`, pixel-proto, ⚠ all validation-gated)

### Backbone = Z-Image S3-DiT (NOT Chroma)

```
layers              30      single-stream (layers.{i})
hidden              3840    (30 heads × 128, full MHA, qk_norm)
ffn                 10240   (SwiGLU w1/w2/w3)
axes_dims           [32, 48, 48], rope_theta 256
t_scale             1000
cap_feat_dim        2560    (Qwen3-4B encoder — same as Z-Image)
n_refiner_layers    2       (noise_refiner + context_refiner)
```

Differences from stock Z-Image:
- **x0-prediction** flow matching (Z-Image predicts a negated velocity).
- **Pixel-space** in/out: `x_embedder` takes raw pixel patches. Patch size is inferred from the `x_embedder.weight` in-dim: `p = round(sqrt(in_dim / 3))`; reported in-dim 3072 → **patch 32**.
- A **DeCo-style pixel decoder head** is reported (`decoder_num_res_blocks = 4`, `decoder_max_freqs = 8`, hidden auto-inferred, DCT-style final layer) — ⚠ **decoder head key layout UNCONFIRMED**. Our `LoadWeights` enumerates `decoder*`/unexpected `final_layer.*` keys and throws `UnsupportedModelException` listing them if the layout differs from the plain Z-Image `final_layer` path.
- Output conversion: `v = (x_t − x0) / t` (ComfyUI `NextDiTPixelSpace`).

Keys follow Z-Image single-file naming: `x_embedder.*`, `layers.{0..29}.*`, `t_embedder.*`, `cap_embedder.*`, `noise_refiner.*`, `context_refiner.*`, `final_layer.*`, `cap_pad_token`, `x_pad_token`.

### Sampling (⚠ mid-pretraining defaults)

- Flow-match Euler, default **50 steps**, CFG **5.0**, 1024×1024, resolution divisible by the patch size (32).
- Timestep conditioning assumed to follow Z-Image's `(1 − sigma)` inversion convention (same lineage) — ⚠ validation-gated.
- CFG combine on converted velocity assumed standard `uncond + cfg·(cond − uncond)` (NOT Z-Image's non-standard cond-baseline formula) — ⚠ validation-gated.

---

## Uncertainty table (validation-gated items)

| # | Item | Assumption in our implementation | How to validate |
|---|---|---|---|
| 1 | Radiance cosine positional basis | Separable `cos(π·u·y)·cos(π·v·x)`, `u,v ∈ [0,8)`, positions `linspace(0,1,16)`, feature index `u·8+v`, concatenated after `[r,g,b]` | Diff `NerfEmbedder.fetch_pos` in ComfyUI `comfy/ldm/chroma_radiance/layers.py` against a dumped activation |
| 2 | 49152 param_generator split | `[W1 gate, W2 value, W3 out]` contiguous; each row-major `[out, in]` (W1/W2 `[256, 64]`, W3 `[64, 256]`) | Activation diff on one patch vs ComfyUI |
| 3 | Final-norm before NeRF head | Skipped (head replaces `final_layer` and the mod-table's last 2 rows go unused) | Compare against ComfyUI `ChromaRadiance.forward_orig` |
| 4 | `nerf_final_layer_conv` sub-key names | `nerf_final_layer_conv.{norm.scale, conv.weight [3,64,3,3], conv.bias}` | Key dump of the safetensors |
| 5 | Radiance scheduler shift | Static 1.0 | ComfyUI model_sampling for `chroma_radiance` |
| 6 | Zeta decoder head | Not implemented; plain Z-Image `final_layer` path only, hard error on unknown `decoder*` keys | Key dump of a Zeta pixel-proto checkpoint |
| 7 | Zeta timestep convention | `(1 − sigma)` like Z-Image, ×1000 internally | ComfyUI `NextDiTPixelSpace` |
| 8 | Zeta CFG formula | Standard CFG on velocity | ComfyUI sampling path |
| 9 | Zeta scheduler shift | Static 3.0 (Z-Image Turbo default) | ComfyUI model config |
| 10 | Zeta pad tokens / SeqMultiOf in pixel mode | Same as Z-Image (32, learned pad tokens if present) | Checkpoint key dump |

## Implementation Notes (HartsyInference)

- `ChromaTransformer` gained an internal `ForwardCore` (blocks only, no img embed / final norm / proj_out) so `ChromaRadianceTransformer` reuses the backbone without duplication. Classic `Forward` is unchanged.
- `ChromaRadianceImagePatchifier` (conv patchify) and `ChromaRadianceNerfHead` live under `Models/Denoisers/DiTBlocks/`.
- `ZetaChromaTransformer` wraps an unmodified `ZImageTransformer` configured with `InChannels = 3`, `PatchSize = inferred` — the Z-Image patchify/unpatchify path *is* the pixel head.
- Both pipelines share `X0Prediction.ToVelocity` (`Utilities/`), and both register identity latent-preview factors (pixel-space latents preview as-is).
- Converters: `ChromaCheckpointConverter` passes `img_in_patch.*` / `nerf_*` keys through verbatim (no diffusers rename exists for them) and strips `_orig_mod.`; `ZetaChromaCheckpointConverter` reuses the Z-Image partitioner and additionally buckets `decoder*` keys so the transformer can report them.

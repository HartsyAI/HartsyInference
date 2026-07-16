# Parameter Parity Audit — Image & Video Models

Audit scope: for every image / video model + variant, compare against the official reference
(HF `config.json` files + the diffusers/upstream pipeline `__call__`) and list parameters we are
**MISSING**, have **WRONG**, or **HARDCODE**. Covers BOTH (A) architecture/config fields needed to
load weights and (B) user-facing generation/sampling params.

Generated 2026-06-28 by a fan-out of per-family research agents. Status legend per row:
MATCH / MISSING / WRONG / HARDCODED / NAMED-DIFFERENTLY / UNKNOWN / NEEDS-VERIFY.

---

## Executive summary — ranked by severity

### Tier 1 — load-breaking / wrong-weights (architecture config is WRONG, real checkpoint won't load or will silently mis-shape)
1. **HunyuanImage 2.1** — `V21`/`V21Distilled` use `hidden_size=3072, heads=24`; real model is **3584 / 28**. Every weight shape mismatches. Also `guidance_embed` is **inverted** (full/distilled swapped) and distilled `use_meanflow` is unmodeled.
2. **AuraFlow** — `PosEmbedMaxSize` hardcoded **1024**; real v0.3=**9216**, v0.2=**4096**. Learned pos_embed weight `[1, max, dim]` truncates on load at every resolution.
3. **Qwen-Image** — `ContextDim=4096` but real `joint_attention_dim=**3584**` (Qwen2.5-VL hidden). Text in-proj mis-shaped. `pooled_projection_dim` 2048 vs 768.
4. **F-Lite-7B** — `V1_7B` preset guesses `2560/10/32`; real 7B is **3072/12/28** (doc admits placeholder). Won't load real 7B.
5. **SD3.5** — `dual_attention_layers` wrong for BOTH (Medium off-by-one missing layer 12; Large should be **empty**, we set 13). `pos_embed_max_size` Medium should be **384**, silently gets 192. (Real-weight autodetect mitigates; hardcoded presets are the liability.)
6. **LTX-Video** — no 13B preset; only 2B hardcoded. 13B needs head_dim 128 / 48 layers / cross_attn 4096.
7. **HunyuanVideo** — no plain T2V preset at all (only GameCraft 33-ch / 19+38); plain is 16-ch / 20+40.

### Tier 2 — numerically wrong even when weights load (sampling/guidance/scheduler divergence)
8. **Flux Fill** — `in_channels` should be **384** not 128; Fill checkpoints won't be detected as Tools (gated on `==128`). Tools `AdditionalInChannels` also in wrong units.
9. **OmniGen2** — dual guidance (`text_guidance_scale`=4.0 + `image_guidance_scale`=1.0, triple-pass CFG) entirely missing; only single CFG implemented. `cfg_range` missing.
10. **Lumina-2** — `cfg_normalization` defaults **True** upstream and is unimplemented (diverges out-of-box). Scheduler uses static shift 6.0 vs reference dynamic shift.
11. **HiDream** — MoE expert routing (`num_activated_experts=2` top-2-of-4) NOT implemented (single-expert fallback). Wrong scheduler class (Euler vs UniPC for Full / LCM for Dev).
12. **Wan (all)** — scheduler is Euler; reference is **UniPCMultistepScheduler**. Plus the "request defaults never reach config defaults" `<=0`-fallback dead-code bug.
13. **HunyuanImage** — sampling shift hardcoded 3.0 (ref 5/4); embedded-guidance constant (~6016) replaced by cfgScale.
14. **Anima/Cosmos-Predict2** — RoPE extrapolation axes wrong (we put 2.0 on temporal; Cosmos is 4.0 on H/W). EDM scheduler replaced by flow-match (plausibly intentional for Anima — verify).
15. **Krea 2 / Chroma** — guidance convention `cond + s*(cond-uncond)` vs our `uncond + s*(...)` (Krea 4.5≡our 5.5); Chroma per-variant default steps/cfg are dead (pipeline reads generic request).
16. **LTX-Video 0.9.1+/13B** — `decode_timestep`/`decode_noise_scale` (0.05/0.025) not passed to the timestep-conditioned VAE. STG/skip-layer/per-step guidance schedule missing.
17. **Lance** — `VideoTimestepShift=4.0` likely should be **3.5** (paper/README); `ImageTimestepShift=3.5` likely should be **4.0** (paper Table 2). Both suspect.
18. **ERNIE-Image** — scheduler shift default 1.0; real config ships **4.0**.

### Tier 3 — the systemic "wrong defaults / missing user knobs" problem (affects ~all models)
- The shared `TextToImageRequest` defaults (Steps=20, CfgScale=7.5, 512x512) match almost no model. Most need 28-50 steps, guidance 3-7, 1024px (video: non-square + num_frames/fps which have no home in the request type at all).
- The `<=0`-fallback pattern in video pipelines makes model-correct config defaults **dead code**.
- Pervasively MISSING user knobs: `true_cfg_scale` (Flux/Qwen negative-prompt path — `NegativePrompt` field exists but is silently ignored by Flux/Flux2), `guidance_rescale`, custom `sigmas`/`timesteps`, `num_images_per_prompt` (batch always 1), SD3.5 Skip-Layer-Guidance, SDXL micro-conditioning (`original_size`/`target_size`/`crops_coords_top_left` hardcoded), per-encoder `prompt_2`/`prompt_3`.

### Best-in-class (verified parity, minimal/no gaps)
Ideogram 4, Krea 2, ERNIE-Image (arch), OmniGen2 (arch), F-Lite-10B, HiDream (arch fields), Kandinsky-5 (arch), Matrix-Game 2.0 (arch), Oasis-500m, LTX-2 (arch) — all bit-exact on architecture; gaps are sampling-knob or default-value only.

### Not built
**Cosmos-Predict1 Video2World** (5B/13B) — zero code; full build-spec captured at the end of this doc.

---

> Cross-cutting default note (applies to nearly every image model): the shared
> `Requests/TextToImageRequest.cs` defaults — `Steps=20`, `CfgScale=7.5`, `Width/Height=512` — are
> SD-1.5-era and do **not** match almost any modern model's reference defaults (most are 28-50 steps,
> guidance 3.5-7.0, 1024x1024). Functional when the caller overrides them, but the out-of-box defaults
> are wrong for most models below. This is listed once here rather than repeated per model.

---

# IMAGE MODELS

## Stable Diffusion 1.5 / SDXL (base / refiner / inpaint)

### Reference sources
- SD1.5 unet/vae/scheduler config.json @ stable-diffusion-v1-5/stable-diffusion-v1-5
- SDXL base @ stabilityai/stable-diffusion-xl-base-1.0; refiner @ ...-refiner-1.0
- diffusers `StableDiffusionPipeline` / `StableDiffusionXLPipeline` / `StableDiffusionXLImg2ImgPipeline` `__call__`

### Architecture config (UNet / VAE / scheduler) — core fields MATCH
SD1.5, SDXL base, SDXL refiner UNet+VAE+scheduler core fields (channels, block layout, attention head
dims, cross-attn dims, transformer-layers-per-block, ADM/`projection_class_embeddings_input_dim`,
`addition_time_embed_dim`, VAE scaling/shift, betas, prediction_type) all MATCH. Hardcoded-but-correct:
`act_fn=silu`, `flip_sin_to_cos`, `freq_shift=0`, conv kernels, `resnet_time_scale_shift=default`.

### Findings (prioritized)
1. **SDXL inpaint pipeline is a stub** — `SdxlInpaintPipeline.InpaintFromTokens` throws
   `NotImplementedException`; the 9-channel inpaint UNet (`in_channels=9`) is not modeled in `UNetConfig`
   (`InChannels` hardwired to 4). The only working inpaint path is the soft/legacy blend in `SdxlPipeline`
   via `ImageToImageRequest.Mask` (different algorithm, ignores dedicated inpaint weights). Dedicated
   SDXL-inpaint = MISSING entirely.
2. **SDXL micro-conditioning hardcoded, not user-exposed** — `original_size`, `target_size`,
   `crops_coords_top_left` are fixed to `(Height,Width)` / `(0,0)` in `SdxlPipeline.cs` (~L109-146); the
   negative-side variants (`negative_original_size`, `negative_crops_coords_top_left`,
   `negative_target_size`) don't exist. SDXL's signature crop-conditioning knobs are unreachable.
3. **`guidance_rescale` missing** across all SD/SDXL pipelines (Lin et al. zero-SNR/CFG-rescale fix). No
   field in any request or CFG helper.
4. **Scheduler `steps_offset=1` missing** — both SD1.5 (PNDM) and SDXL ship `steps_offset:1`;
   `SchedulerConfig` has no such field. Verify `TimestepSpacing.Leading` applies the +1, else every run is
   off by one timestep vs reference.
5. **Default-value drift** — `Steps=20` (ref 50), `CfgScale=7.5` applied to SDXL too (SDXL ref 5.0),
   `Strength=0.75` (ref img2img/refiner 0.3), SDXL `Width/Height=512` (SDXL native 1024).
6. **SDXL `prompt_2`/`negative_prompt_2` not separable** at the request level (separate CLIP-L vs CLIP-G
   prompts). Plumbing exists at `GenerateFromTokens` (per-encoder token arrays) but `TextToImageRequest`
   has a single `Prompt`/`NegativePrompt`.
7. **Refiner `denoising_start` handoff missing** — diffusers' ensemble-of-expert-denoisers uses
   `denoising_end` (base) + `denoising_start` (refiner, a fraction). We use an img2img `Strength` re-noise
   model (`SdxlRefinerRequest.Strength` / `RefinerSwapConfig`). NAMED-DIFFERENTLY; strict latent handoff absent.
8. **`num_images_per_prompt`, `eta`, custom `timesteps`/`sigmas` not exposed** (batch always 1, eta=0). Low priority.

---

## Stable Diffusion 3.5 (Medium / Large / Large Turbo)

### Reference sources
- transformer/vae/scheduler config.json @ stabilityai/stable-diffusion-3.5-medium & -large (via ungated mirrors)
- diffusers `StableDiffusion3Pipeline.__call__`

### Architecture config
Core fields MATCH (num_layers 24/38, head_dim 64, heads 24/38, hidden 1536/2432, joint_attention_dim 4096,
pooled 2048, in_channels 16, patch_size 2, qk_norm rms). VAE scaling 1.5305 / shift 0.0609 / 16ch and
scheduler shift 3.0 confirmed.

### Findings (prioritized)
1. **`dual_attention_layers` WRONG for BOTH variants** (`Sd3Config.cs:60,70`):
   - Medium reference = `[0..12]` (13 layers); our `Medium35` = `[0..11]` (12) — off by one, missing layer 12.
   - Large reference = `[]` (NO dual attention); our `Large35` = `[0..12]` (13) — Large should have none.
   - (`AutoDetect` from `attn2.*` weights likely loads real weights correctly; the hardcoded presets are the liability.)
2. **`pos_embed_max_size` WRONG for Medium** — reference Medium=**384**, Large=192. Our default is 192 and
   neither preset sets it, so Medium silently gets 192. Breaks pos-embed slicing at larger latents (`Sd3Config.cs:33,54,64`).
3. **Skip-Layer Guidance (SLG) entirely missing** — `skip_guidance_layers`, `skip_layer_guidance_scale`
   (2.8), `_start` (0.01), `_stop` (0.2). StabilityAI explicitly recommends SLG for SD3.5 Medium.
4. **Dynamic shift (`mu`) not used** — `Sd3Pipeline` always uses fixed shift 3.0; scheduler has
   `CreateWithDynamicShift` but SD3 path never calls it. Fine for stock SD3.5; flagged as missing optional.
5. **Per-encoder prompts missing** — `prompt_2/prompt_3`, `negative_prompt_2/3` not exposed (single prompt
   fanned to all three encoders).
6. **`out_channels`, `sample_size`, `max_sequence_length`(256), `sigmas`, `num_images_per_prompt`, IP-Adapter** not exposed/read from config (mostly hardcoded-but-correct).
7. Default drift: Steps 20 (ref 28), Cfg 7.5 (ref 7.0), W/H 512 (ref 1024).
8. **Large Turbo** has no distinct preset; treated as Large (38L/2432) + distilled weights, recommended steps≈4, guidance≈1.0. Distinct config.json existence UNKNOWN (gated).

---

## FLUX.1 family (dev / schnell / Krea / Tools: Canny / Depth / Fill)

### Reference sources
- transformer/vae config @ FLUX.1-dev (ungated mirrors), FLUX.1-Fill-dev-diffusers, FLUX.1-Canny-dev-nf4
- diffusers `FluxPipeline` / `FluxControlPipeline` / `FluxFillPipeline` `__call__`
- (BFL canonical repos gated 401; values from diffusers-format mirrors.)

### Architecture config
Core MATCH: heads 24, head_dim 128, hidden 3072, num_layers 19, single_layers 38, joint_attention_dim 4096,
pooled 768, axes_dims_rope [16,56,56], theta 10000, guidance_embeds per-variant. out_channels 64.

### Findings (prioritized)
1. **Fill `in_channels` is 384, not 128 (WRONG)** — `FluxConfig.Flux1Tools` hardcodes `InChannels=128`, and
   the pipeline gates Tools mode on `XEmbedInputDim==128` only. FLUX.1-**Fill**-dev = 384 (64 noise + 320
   masked-image+mask). A Fill checkpoint won't be detected as Tools and the concat width is wrong. Fix:
   detect Tools when `XEmbedInputDim>64`; build conditioning per-variant (Canny/Depth +64, Fill +320).
2. **`FluxToolsConfig.AdditionalInChannels` in wrong units** — Canny/Depth set 1 and Fill sets 17, but real
   deltas over the 64-ch base are +64 (Canny/Depth) and +320 (Fill). These presets are also currently dead
   (pipeline reads `XEmbedInputDim`). Inconsistent/dead config.
3. **No `true_cfg_scale` / real negative-prompt path** — `NegativePrompt` is a field but never encoded or
   used anywhere in `FluxPipeline`. diffusers supports `true_cfg_scale>1` (real uncond pass). We silently ignore it.
4. **Schnell step default** — global `Steps=20`; no schnell-aware default (ref 4) nor dev/Tools default (28).
5. **`CfgScale=7.5` field is ignored by Flux** (pipeline uses `guidanceScale` arg default 3.5). Misleading SD-era leftover. Krea should default 4.5.
6. **`max_sequence_length`(512) not exposed**; **custom `sigmas` not supported**.
7. **NEEDS-VERIFY** VAE scaling 0.3611 / shift 0.1159; scheduler base_shift 0.5 / max_shift 1.15 /
   base_seq 256 / max_seq 4096 — not visible in audited files (live in VAE + `CreateWithDynamicShift`).
8. **No dedicated Krea preset** (loads via `FromWeights` as dev; only the 4.5 guidance default is unsurfaced). MINOR: `patch_size=1` hardcoded.

---

## FLUX.2 (dev 32B / Klein 4B / Klein 9B)

### Reference sources
- BFL official `flux2` repo source (`model.py` Flux2Params/Klein4BParams/Klein9BParams, `util.py`,
  `autoencoder.py`, `text_encoder.py`), diffusers `transformer_flux2.py`, diffusers `flux2` pipeline docs.
- (HF repos gated 401; reconciled from BFL source — authoritative for same checkpoints.)

### Architecture config
All numeric arch fields MATCH (in/out 128, hidden 6144/3072, heads 48/24, head_dim 128, depth 8/5,
single 48/20, context 15360/7680, mlp_ratio 3.0, axes [32,32,32,32], theta 2000, guidance per-variant,
eps 1e-6, timestep channels 256, VAE z=32 ch_mult [1,2,4,4], raw latent no scale/shift). Klein 9B preset
(context 12288, hidden 4096, heads 32, depth 8, single 24) matches `Klein9BParams` exactly.

### Findings (prioritized)
1. **dev `text_encoder_out_layers` WRONG** — reference dev (Mistral) taps `[10,20,30]`; our
   `Flux2Pipeline._hiddenLayers` defaults to `[9,18,27]` for both variants. `[9,18,27]` is correct only for
   Klein (Qwen3). Fix: default by encoder type — Mistral→[10,20,30], Qwen→[9,18,27].
2. **dev text encoder is a VLM (Mistral-Small-3.2-24B), not a plain LLM** — supports image input + caption
   upsampling (temp 0.15). Our pipeline routes dev through `LlamaStyleEncoder` and always logs "Qwen3".
   Verify the Mistral-Small backbone (hidden 5120 → 3×5120=15360=ContextInDim) is what loads. Caption upsampling MISSING.
3. **`max_sequence_length=512` not enforced** (reference clamps prompt tokens to 512).
4. **`sigmas` custom schedule not exposed.**
5. **Klein negative-prompt / CFG not wired** — `Flux2KleinPipeline` does CFG with `""` negative; we ignore
   `NegativePrompt`. (dev is guidance-distilled → our embedded-guidance path is correct for dev.)
6. **Reference-image conditioning missing** — both dev (edit) and Klein KV support an `image` input
   (token-level), beyond our init-noise img2img. Larger feature gap.
7. Default drift: steps 20 (ref 50; KV 4), guidance 3.5 (ref 4.0), W/H 512 (ref 1024).
8. Cosmetic: doc comments say Klein 9B uses "Qwen3-4B" — actually Qwen3-8B-FP8; log string hardcodes
   "Qwen3" on the Mistral path. Note `PatchSize=2` (ours, pipeline-side) vs transformer config `patch_size=1` — equivalent, don't conflate.

---

## Z-Image (Turbo / Base)

### Reference sources
- transformer/scheduler config.json @ Tongyi-MAI/Z-Image-Turbo; diffusers `pipeline_z_image.py` (+img2img)

### Architecture config
All transformer-config fields MATCH (dim 3840, n_heads 30, n_layers 30, n_refiner_layers 2, in 16,
patch [2]/f_patch [1], cap_feat_dim 2560, axes_dims [32,48,48], axes_lens [1536,512,512], norm_eps 1e-5,
rope_theta 256.0, t_scale 1000.0). FfnDim 10240 / HeadDim 128 derived from weights. Scheduler shift 3.0,
use_dynamic_shifting false (static) — MATCH for Turbo.

### Findings (prioritized)
1. **CFG truncation (`cfg_truncation`=1.0) MISSING** and **CFG normalization (`cfg_normalization`=False)
   MISSING** — real Z-Image guidance behaviors on the Base/CFG path. `ApplyZImageCfg` does the combine but
   never disables guidance past the truncation timestep nor renormalizes. (Moot for Turbo cfg=1; gap for Base.)
2. **Chat-template parity unverified (NEEDS-VERIFY)** — reference `encode_prompt` applies the Qwen3 chat
   template with `enable_thinking=True` + `add_generation_prompt=True`. Our pipeline defers this upstream;
   thinking mode materially changes the token stream — confirm the upstream encoder matches exactly.
3. **`n_kv_heads`(=30) not a field** — harmless (full MHA), but a future GQA Z-Image variant would silently break.
4. **Z-Image-Base scheduler shift UNKNOWN** — we set Base `SchedulerShift=6.0` but only Turbo (3.0) is verified.
5. Default drift: steps 20 (ref t2i 50 / Turbo 8), guidance method-default 1.0 (ref 5.0; Turbo wants 1.0), W/H 512 (native 1024).
6. LOW: `qk_norm:true` hardcoded (no toggle); `AdaLNEmbedDim=256`, `SeqMultiOf=32`, VAE scale/shift hardcoded (absent from config.json). `num_images_per_prompt`, custom `sigmas` unsupported.

---

## Lumina-Image-2.0

### Reference sources
- transformer/config.json @ Alpha-VLLM/Lumina-Image-2.0; diffusers `pipeline_lumina2.py` `__call__`

### Architecture config
MATCH: hidden 2304, heads 24, kv_heads 8, layers 26, refiner 2, in 16, out null(=in), patch 2,
cap_feat_dim 2304, axes_dim_rope [32,32,32], axes_lens [300,512,512], norm_eps 1e-5, ffn_dim_multiplier
null. FfnDim 6144 / HeadDim 96 derived.

### Findings (prioritized)
1. **`cfg_normalization` defaults to TRUE and is NOT implemented (WRONG at default)** — reference
   renormalizes the guided velocity to the conditional's norm by default; we do plain `CfgHelper.ApplyCfg`
   with no normalization. Single most impactful gap (diverges out-of-box).
2. **Scheduler shift WRONG** — Lumina2 uses dynamic shifting (base 0.5 / max 1.15, mu from image_seq_len);
   we hardcode static `SchedulerShift=6.0` (the config comment even acknowledges this approximation).
3. **`cfg_trunc_ratio`(=1.0) MISSING** — no-op at default but the reference selectively disables CFG past a timestep ratio.
4. **System-prompt parity unverified** — reference prepends a specific default system prompt to the Gemma-2 input; confirm upstream encoder applies exactly that string.
5. MED/LOW: `rope_theta=10000`, `AdaLNEmbedDim=1024` hardcoded (absent from config.json); `multiple_of=256`, `sample_size=128` not stored; steps default 20 (ref 30); guidance default 4.0 MATCHES; W/H 512 (native 1024); `num_images_per_prompt`, `sigmas` unsupported.

---

## Qwen-Image (+ Qwen-Image-Edit, shares config)

### Reference sources
- transformer/vae/scheduler config.json @ Qwen/Qwen-Image (+ Qwen-Image-Edit); diffusers `pipeline_qwenimage.py` `__call__`

### Architecture config
MATCH: num_layers 60, heads 24, head_dim 128, hidden 3072, patch_size 2. VAE z_dim 16, latents_mean/std
(16-vectors verbatim) MATCH.

### Findings (prioritized)
1. **`joint_attention_dim` WRONG** — reference 3584 (Qwen2.5-VL hidden), we hardcode `ContextDim=4096`. If
   the transformer text in-proj is built from `ContextDim` this is a shape bug; the `LlamaStyleEncoder`
   hidden dim must be 3584. Highest priority.
2. **`in_channels` mislabeled** — reference transformer `in_channels=64` (=16 latent x 2x2 patch),
   `out_channels=16`. Our `InChannels=16` is really `out_channels`; numerically `PatchSize^2 x 16 = 64`
   lands right, but the field is semantically wrong — rename/document.
3. **`pooled_projection_dim` WRONG** — reference 768, we set 2048 (likely dead; Qwen-Image has no pooled-CLIP path).
4. **`axes_dims_rope=[16,56,56]` not represented** — only scalar `RopeTheta`; verify `QwenImageRope` hardcodes [16,56,56] (sum 128).
5. **Scheduler `shift_terminal=0.02`** — verify `CreateWithDynamicShift` applies the terminal stretch (base_shift 0.5 / max_shift 0.9 / base_seq 256 / max_seq 8192), else late sigmas off.
6. Default drift: true_cfg 4.0 (we 7.5), steps 50 (we 20), W/H 1024 (we 512). `max_sequence_length=512`, batch>1, `sigmas` unsupported.
7. Note: `V2_14B`/`V2_20B` presets are invented placeholders for an unreleased model — not verified.

---

## Ideogram 4

### Reference sources
- Open-weights `ideogram-ai/ideogram-4-fp8` (upstream `modeling_ideogram4.py`, `pipeline_ideogram4.py`,
  `latent_norm.py`, `sampler_configs.py`); official `developer.ideogram.ai` generate-v4 API.

### Architecture config — EXCELLENT parity
Every numeric arch field MATCHES: 34 layers, emb 4608, 18 heads, head_dim 256, FFN 12288, adaLN 512,
in/out 128, LLM-feat 53248, theta 5e6, MRoPE (24,20,20), Flux.2 KL VAE (scale 8), patch 2, Qwen3-VL-8B
13-layer tap, latent-norm Scale/Shift verbatim. No WRONG arch values.

### Findings (prioritized)
1. **HARDCODED to verify: `MaxTextTokens=2048`** — not confirmed against an upstream config field.
2. **Verify exact sampler constants vs `sampler_configs.py`** — polish-step counts (Turbo 1 / Default 2 /
   Quality 3), main/polish guidance weights (~7.0/3.0), per-preset logit-normal `Mu`/`Std`. Structure
   matches the 3 speed tiers (rendering_speed FLASH/TURBO/DEFAULT/QUALITY → presets); the exact constants are the one place a silent mismatch could hide.
3. Resolution model differs by design (API uses a fixed 2K enum; we accept any W/H mult-16, 256-2048) — more permissive, not wrong.
4. Server-side features correctly omitted (no weights): `magic_prompt`, `style_type`/`style_codes`/
   `color_palette`/`style_reference_images`, `enable_copyright_detection`. v4 `negative_prompt`/`seed`/`num_images` status UNKNOWN (v4 uses asymmetric uncond-CFG, not a negative prompt — matches our pipeline).
5. **MISSING capability (not parity bug): `json_prompt`** — v4's native structured input (bounding-box layout on 0-1000 grid + <=16-color hex palette). We take a flat prompt only.
6. `num_images` hardcoded to 1.

---

## AuraFlow (v0.3 / v0.2)

### Reference sources
- transformer/scheduler/vae config.json @ fal/AuraFlow-v0.3 & v0.2; diffusers `AuraFlowPipeline.__call__`

### Architecture config
MATCH: head_dim 256, heads 12, mmdit 4, single 32, joint_attention_dim 2048, caption_projection 3072,
in/out 4, patch 2, register_tokens 8, inner_dim 3072. VAE SDXL (scaling 0.13025). Scheduler shift 1.73.

### Findings (prioritized)
1. **`pos_embed_max_size` WRONG/HARDCODED (CRITICAL)** — we hardcode `PosEmbedMaxSize=1024`; real checkpoints
   are **v0.3=9216** (96x96), **v0.2=4096** (64x64). The learned `pos_embed.pos_embed` weight is
   `[1, pos_embed_max_size, dim]` — loading into a 1024-row buffer mismatches/truncates and
   `pe_selection_index` indexes the wrong base grid. Breaks real-weight loading at every resolution. The
   XML doc comments (L13, L49-50) claiming "1024 = 32x32" are also wrong.
2. **No v0.2 preset** — v0.2 differs only by `pos_embed_max_size` (4096 vs 9216); add a `V02` preset.
3. Default drift: steps 50 (we 20), guidance 3.5 (we 7.5 — well above tuned range), W/H 1024 (we 512).
4. `sample_size`(64), `max_sequence_length`(256 hardcoded in T5 encoder), `num_images_per_prompt`, `sigmas`, `attention_kwargs` not modeled/exposed.

---

## Chroma / Chroma Radiance / ZetaChroma

### Reference sources
- diffusers `pipeline_chroma.py` + `transformer_chroma.py`; config.json @ lodestones/Chroma1-HD &
  Chroma1-Radiance (`config_radiance.json`). Zeta-Chroma has NO public config (mid-pretraining).

### Architecture config
Chroma/Radiance core MATCH: depth 19, single 38, heads 24, head_dim 128, hidden 3072, approximator
(in 64 / hidden 5120 / 5 layers / out 3072), mod_index_length 344; Radiance nerf (hidden 64 / depth 4 /
ratio 4 / max_freqs 8); guidance_embeds false + pooled_projection dropped (correct — approximator replaces).

### Findings (prioritized)
1. **Effective default steps/CFG WRONG (all variants)** — each config has correct per-variant defaults
   (`ChromaConfig.DefaultCfgScale`=5.0/`DefaultSteps`=35; Radiance 3.5/50; Zeta 5.0/50) but all three
   pipelines read `request.Steps`/`request.CfgScale` (generic 20 / 7.5). The config defaults are **dead**
   unless the call site wires them. Fix: fall back to config defaults when the request is left at generic defaults.
2. **`latents`/`InitialNoise` parity gap (Chroma classic only)** — Radiance & Zeta honor
   `request.InitialNoise`; `ChromaPipeline` always calls `SeedGenerator.CreateNoise` and ignores it. Blocks PyTorch noise-injection parity for main Chroma.
3. `joint_attention_dim=4096`, `axes_dims_rope=[16,56,56]`, theta 10000, mlp_ratio 4.0, qkv_bias true,
   `in_channels=64` are correct in value but HARDCODED in encoder/transformer (not surfaced on `ChromaConfig`).
4. MISSING: `sigmas`, `num_images_per_prompt`, IP-Adapter inputs (diffusers Chroma supports IP-Adapter via `joint_attention_kwargs`).
5. **ZetaChroma entirely UNKNOWN vs reference** — no public config; PatchSize=32, DecoderHidden=3840,
   SchedulerShift=3.0, x0+inverted-timestep sampling all validation-gated guesses. Released non-proto Zeta uses the **Flux 2 VAE** (our modeled proto is VAE-free) — record for the eventual full pipeline.
6. Radiance `SchedulerShift=1.0` / Zeta `3.0` not in any reference config (ComfyUI-derived) — UNKNOWN.

---

## HunyuanImage 2.1

### Reference sources
- Tencent-Hunyuan/HunyuanImage-2.1 GitHub configs (`hunyuanimage_config.py`, `hunyuanimage_dit.py`) +
  pipeline; HF tencent/HunyuanImage-2.1.

### Findings (prioritized) — SEVERE
1. **CRITICAL: wrong backbone dims** — real v2.1 = `hidden_size=3584, heads_num=28`; our `V21` preset uses
   `3072/24`. Every linear/attention weight shape mismatches the real checkpoint. (head_dim 128 matches only
   by coincidence since 3072/24 also = 128.) Fix `V21` and `V21Distilled` to 3584/28.
2. **CRITICAL: `guidance_embed` inverted** — reference: full=**False**, distilled=**True**. Ours: full=true,
   distilled=false. The pipeline's CFG-vs-distilled-guidance branch takes the wrong path for BOTH variants. Swap.
3. **CRITICAL: distilled `use_meanflow=True` not modeled** — no field/flag; distilled + refiner use meanflow sampling. Needs `UseMeanflow` + sampling support.
4. **HIGH: sampling `shift` hardcoded to 3.0** — reference shift = 5 (full) / 4 (distilled); scheduler built
   with literal `3.0f`. Real model also uses a custom sigma scheduler (`get_timesteps_sigmas`), not plain FlowMatchEuler.
5. **HIGH: embedded-guidance constant** — distilled passes a fixed guidance scalar (~6016.0) when guidance is None; our pipeline feeds `cfgScale` into the guidance embed. Verify the real scalar.
6. **MEDIUM: ByT5 glyph branch missing** — `glyph_byT5_v2=True` in both configs (ByT5Mapper in_dim 1472); pipeline always passes `encoderHidden2=null`. Glyph/text-render prompts wrong until built.
7. **LOW: no refiner preset** (`hidden 3328 / heads 26 / in 128 / out 64 / rope [16,56,56] / meanflow`). The base+refiner two-stage pipeline isn't represented.
8. Architecture fields that DO match: double 20 / single 40, in 64, patch 1, rope [64,64], theta 256, text_states_dim 3584, text_states_dim_2 1472, mlp_ratio 4, qk_norm. Config doc comment is also backwards (labels 17B full as guidance-embed true).

---

## ERNIE-Image (+ Turbo)

### Reference sources
- transformer/scheduler/model_index config.json @ baidu/ERNIE-Image; diffusers `ErnieImagePipeline` docs.

### Architecture config — FULL MATCH
hidden 4096, heads 32, head_dim 128, layers 36, ffn 12288, in/out 128, patch 1, text_in_dim 3072,
rope_theta 256, rope_axes [32,48,48], eps 1e-6, qk_layernorm true — all MATCH. VAE = AutoencoderKLFlux2
(128-ch) confirmed. Text encoder = Mistral3 (Ministral-3B), uses `hidden_states[-2]`.

### Findings (prioritized)
1. **HIGH: scheduler shift default WRONG** — real `scheduler_config.json` ships `shift=4.0`; our pipeline
   defaults `schedulerShift=1.0`. Unless every caller passes 4.0 the noise schedule is off. Default to 4.0.
2. **MEDIUM: confirm `[-2]` hidden-state tap** is honored by our encoder (pipeline doc says `[-2]`; verify).
3. **LOW: optional Prompt-Enhancer LLM (`Ministral3ForCausalLM`)** not modeled (acceptable, optional).
4. Default drift: steps 20 (ref 50 / Turbo 8), cfg 7.5 (ref 4.0 / Turbo 1.0). Turbo preset = base backbone (correct; only runtime params differ).

---

## HiDream-I1 (Full / Dev / Fast)

### Reference sources
- transformer/scheduler config.json @ HiDream-ai/HiDream-I1-Full & -Dev; diffusers `HiDreamImagePipeline.__call__`; HiDream `inference.py`.

### Architecture config — FULL MATCH (config values)
patch 2, in/out 16, layers 16, single 32, heads 20, head_dim 128, text_emb_dim 2048, axes_dims_rope
[64,32,32], max_resolution [128,128], caption_channels [4096,4096], llama_layers (48-entry),
num_routed_experts 4, num_activated_experts 2 — all MATCH.

### Findings (prioritized)
1. **MoE expert routing NOT implemented (headline)** — config carries `num_routed_experts=4` /
   `num_activated_experts=2` and `HiDreamConfig` plumbs them into `HiDreamBlock`, but `HiDreamTransformer.cs`
   docstring says "currently single-expert fallback — // TODO: full MoE routing". Top-2-of-4 gated routing
   is NOT in the FFN. This is a numerical-parity defect (not a config defect). Verify/implement.
2. **Wrong scheduler class for Full** — reference Full uses `UniPCMultistepScheduler` (flow_shift 3.0,
   flow_prediction, 2nd-order); Dev uses `FlowMatchLCMScheduler` (shift 6.0). We hardcode
   `FlowMatchEulerDiscreteScheduler` for all — matches none exactly.
3. **Per-variant presets missing** — Full/Dev identical, no Fast. Differences are all in gen params
   (steps 50/28/16, cfg 5.0/0.0/0.0, shift 3.0/6.0/3.0, scheduler class) and none are encoded. Engine
   defaults (20/7.5/3.0/Euler) match no reference preset.
4. **SchedulerShift WRONG for Dev** (we use fixed 3.0, Dev=6.0). guidance default 7.5 vs Full 5.0 / distilled 0.0.
5. `max_sequence_length`(128), `num_images_per_prompt`(1), `sigmas`, default-resolution 512 vs 1024 — minor.

---

## Kandinsky 5.0 (image, T2I-Lite)

### Reference sources
- transformer/config.json @ kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers; diffusers `Kandinsky5T2IPipeline.__call__`.

### Architecture config — FULL MATCH
in/out_visual 16, time_dim 512, patch [1,2,2], model_dim 2560, ff_dim 10240, num_text_blocks 2,
num_visual_blocks 50, axes_dims [32,48,48], in_text_dim 3584 (Qwen2.5-VL), in_text_dim2 768 (CLIP-L),
visual_cond false, derived head_dim 128 / heads 20 — all MATCH.

### Findings (prioritized)
1. **`attention_type` not modeled** — reference has "regular" (T2I-Lite) vs "nabla" (sparse, video family).
   T2I-Lite is "regular" so no gap today, but a "nabla" image checkpoint would be silently treated as regular.
2. Default drift: steps 50 (we 20), cfg 3.5 (we 7.5), 1024 (we 512). No preset injects reference defaults.
3. `max_sequence_length=512` (Qwen truncation) not represented (embeddings-in design; caller supplies cu_seqlens). Text encoders (Qwen2.5-VL + CLIP-L) not in-engine (wiring gap, not param gap).
4. Scheduler shift 5.0 matches our config note but `scheduler_config.json` value not re-confirmed (UNKNOWN-confirm). `num_images_per_prompt` hardcoded 1.

---

## Anima (Cosmos-Predict2 Text2Image)

### Reference sources
- nvidia/Cosmos-Predict2-2B-Text2Image card + nvidia-cosmos/cosmos-predict2 `config_text2image.py`; diffusers
  `pipeline_cosmos2_text2image.py`. NOTE our code targets **Anima** (CircleStone/Comfy retrain that swaps
  T5-XXL for Qwen-3 0.6B + in-checkpoint `llm_adapter`); geometry measured from the Anima checkpoint.

### Architecture config — geometry MATCH
hidden 2048, blocks 28, heads 16, head_dim 128, in/out 16, patch (1,2,2), concat_padding_mask true,
mlp_ratio 4, crossattn 1024, AdaLN-LoRA 256, condition_dim 6144 — all MATCH.

### Findings (prioritized)
1. **RoPE extrapolation axes WRONG** — upstream Cosmos rope3d: H/W extrapolation **4.0x**, temporal 1.0x.
   Our `RopeScale=(2.0,1.0,1.0)` puts 2.0 on temporal and 1.0 on H/W — both axis assignment and magnitude
   differ. If Anima inherited Cosmos's rope this is a real bug; verify against the Anima checkpoint.
2. **Scheduler is EDM upstream, we hardcode flow-match Euler shift 3.0** — Cosmos EDM params
   (`sigma_data=1.0`, `sigma_max=80`, `sigma_min=0.002`, order-7 solver, 35 steps) are all MISSING as knobs.
   Plausibly correct *for Anima* (a flow-match retrain) but a hard divergence from the named Cosmos reference;
   file contains dead `EdmDenoise`/`CosmosCfg` helpers from an abandoned EDM attempt.
3. **CFG form** — pipeline uses `CfgHelper.ApplyCfg` (SD3 form `uncond + s*(cond-uncond)`) while an unused
   Cosmos-form `CosmosCfg` (`cond + s*(cond-uncond)`) sits in the same file. Pin down intent.
4. Cosmos default steps 35 vs generic 20. Text-encoder swap (T5-XXL -> Qwen-3 0.6B) is intentional/measured; leftover T5 `CodebookVocab=32128` embed loaded-but-unused (init artifact, flagged in code).

---

## OmniGen2

### Reference sources
- transformer/config.json @ OmniGen2/OmniGen2; GitHub VectorSpaceLab/OmniGen2 `pipeline_omnigen2.py`.

### Architecture config — FULL MATCH (all 16 fields)
hidden 2520, heads 21, kv_heads 7, layers 32, refiner 2, patch 2, in 16, out null, multiple_of 256,
ffn_dim_multiplier null, norm_eps 1e-5, axes_dim_rope [40,40,40], axes_lens [1024,1664,1664],
timestep_scale 1000, text_feat_dim 2048, theta 10000 — all MATCH.

### Findings (prioritized)
1. **DUAL GUIDANCE MISSING (highest)** — OmniGen2's defining feature is TWO guidance scales:
   `text_guidance_scale` (default **4.0**) and `image_guidance_scale` (default **1.0**), combined via a
   triple-pass CFG (uncond, text-only, text+image). We implement only single classic CFG and have NO
   `image_guidance_scale`. Even pure t2i should use text_guidance 4.0, not our generic 7.5.
2. **`cfg_range`(start,end)=(0.0,1.0) MISSING** — reference can disable CFG outside a timestep window; we CFG every step.
3. **Image-input params MISSING (scoped out)** — `input_images`, `max_pixels`, `max_input_image_side_length`,
   `align_res`, `image_index_embedding` table (our `MaxRefImages=5` passthrough). t2i-only by design; without them `image_guidance_scale` can never be exercised.
4. WRONG defaults: steps 28, text_guidance 4.0. Scheduler `FlowMatchEulerDiscreteScheduler(3.0)` hardcoded; batch 1.

---

## F-Lite (10B / 7B)

### Reference sources
- dit_model/config.json @ Freepik/F-Lite & F-Lite-7B; vae/config.json; GitHub fal-ai/f-lite `model.py`/`pipeline.py`.

### Architecture config (10B) — FULL MATCH
hidden 3072, depth 40, heads 12, in 16, patch 2, mlp_ratio 4, cross_attn_input 4096, rope_base 10000,
residual_v true, train_bias_and_rms false, use_rope true, register_tokens 16, rope_max_grid 512,
T5 tap index 17 (+final_norm). VAE scaling 0.3611 / shift 0.1159 — all MATCH.

### Findings (prioritized)
1. **`FLiteConfig.V1_7B` geometry WRONG (HIGH)** — real F-Lite-7B = `hidden 3072 / heads 12 / depth 28`
   (same width as 10B, depth cut 40->28). Our preset guesses `2560/10/32` (doc admits placeholders). Will
   load wrong shapes against the real 7B checkpoint. Fix to 3072/12/28.
2. LOW: APG (Adaptive Projected Guidance, `apg_config`, threshold 0.03) not implemented (default-off feature gap).
3. INFO: `num_register_tokens=16` correct but is a code constant (not config.json). `dynamic_softmax_temperature`/`gradient_checkpoint` no-ops. Per-variant preset defaults (steps 30, cfg 6.0) correct.

---

## Krea 2 (Base / Turbo) — EXCELLENT parity

### Reference sources
- transformer/config.json @ krea/Krea-2-Turbo & community Krea-2-Base-Diffusers; diffusers krea2 docs + `pipeline_krea2.py`.

### Architecture config — FULL MATCH (verified bit-for-bit)
layers 28, head_dim 128, heads 48, kv_heads 12, hidden 6144, in 64, VAE 16ch (Qwen-Image), patch 2,
intermediate 16384, timestep_embed 256, text_hidden 2560 (Qwen3-VL-4B), text_layers 12, text_heads 20,
text_kv 20, text_intermediate 6912, layerwise_text 2, refiner_text 2, axes [32,48,48], theta 1000,
norm_eps 1e-5, is_distilled (Base false/Turbo true), text_encoder_select_layers (2,5,...,35) — all MATCH.
Scheduler base_shift 0.5 / max_shift 1.15 / base_seq 256 / max_seq 6400, Turbo fixed mu 1.15, promptDropIndex 34 — MATCH.

### Findings (prioritized)
1. **Guidance-scale convention mismatch (HIGH)** — Krea 2 reference velocity = `cond + gs*(cond-uncond)` (so
   Base `guidance_scale=4.5` is effective CFG 5.5). Our pipeline uses standard `CfgHelper.ApplyCfg` =
   `uncond + scale*(cond-uncond)`. Passing 4.5 gives under-guidance. Map `CfgScale -> scale+1` for Krea2 or document passing 5.5.
2. MEDIUM: `promptDropIndex=34` only correct if the caller applies Krea 2's exact chat template (system
   "Describe the image by detailing..." + im_start/user wrapping). External to pipeline; verify tokenizer wiring.
3. LOW: `max_sequence_length=512` (pad/truncate) not enforced; batch 1.

---

## Boogu-Image 0.1 (Base / Turbo / Edit)

### Reference sources
- transformer/scheduler config.json @ Boogu/Boogu-Image-0.1-Base; ComfyUI PR #14523. (Lumina-2/OmniGen2 lineage.)

### Architecture config — core MATCH
hidden 3360, layers 40, double 8, refiner 2, heads 28, kv 7, head_dim 120, patch 2, in 16, out null,
multiple_of 256, ffn_dim_multiplier null, norm_eps 1e-5, axes_dim_rope [40,40,40], axes_lens
[2048,1664,1664], timestep_scale 1000, instruction_feat_dim 4096 — all MATCH. VAE = FLUX.1.

### Findings (prioritized)
1. **Scheduler `seq_len` mismatch (HIGH)** — released `scheduler_config.json` pins `seq_len=4096` (FIXED
   static shift), but `BooguImagePipeline` recomputes seqLen per-image from H/W. At 1024x1024 it coincidentally
   = 4096; at any other resolution the mu/timestep shift diverges. Pass constant 4096 (the scheduler doc-comment even notes this).
2. MEDIUM: Edit `imageGuidanceScale=1.0` default disables double-guidance; model card recommends ~5.0 for edits.
3. **MEDIUM: several fields UNVERIFIED** — `RopeTheta=10000`, `QkNormEps=1e-5`, `OutNormEps=1e-6`,
   `MaxRefImages=5`, `ConditioningDim=1024` are Lumina-2/OmniGen2 assumptions absent from the released config; confirm against Boogu source. "Qwen3-VL-8B tap" is UNKNOWN (card only cites Qwen3-VL-32B for external prompt enhancement; `instruction_feat_dim=4096` confirmed).
4. LOW: config's `instruction_feature_configs`(num_layers 1 / reduce_type mean) and disabled `prompt_tuning_configs` not modeled (no-ops at release values); `base_shift`/`max_shift` 0.5/1.15 inherited, not in scheduler_config.

---

## Microsoft Lens (Lens / Turbo / Base)

### Reference sources
- diffusers config @ YuCollection/Lens-Turbo-Diffusers (faithful conversion; microsoft/Lens* gated 401).
  Official GitHub `LensPipeline.__call__` not publicly readable.

### Architecture config — verifiable fields bit-exact
inner_dim 1536, layers 48, heads 24, head_dim 64, patch 2, in 128, out 32, enc_hidden_dim 2880,
selected_layer_index [5,11,17,23], axes_dims_rope [8,28,28] (sum 64) — all MATCH.

### Findings (prioritized)
1. **`ComputeEmpiricalMu` likely diverges from diffusers dynamic shifting (HIGH)** — real
   `scheduler_config.json` = standard `use_dynamic_shifting` (base_shift 0.5 / max_shift 1.15 / base_seq 256
   / max_seq 4096, shift 3.0, exponential). Our pipeline hardcodes a bespoke piecewise-linear
   `compute_empirical_mu(seq_len, num_steps)` (a1/b1/a2/b2 constants) that also makes mu depend on step count
   (standard dynamic shifting does not). Unless the official Lens pipeline truly overrides diffusers, this mu is WRONG; at minimum HARDCODED + unverified.
2. **MEDIUM: VAE `bnEps` mismatch** — `vae/config.json` `batch_norm_eps=1e-4`; our pipeline/factory default `bnEps=1e-5f`. Default to 1e-4 to match Flux.2 VAE.
3. **MEDIUM: Turbo step count** — reference docs ~8; we default 4. Verify against gated Turbo card.
4. LOW: `RopeTheta`(10000), norm-eps values UNKNOWN (absent from upstream config). `gate_mlp`/`multi_layer_encoder_feature` flags (true) hardcoded not represented. No max-resolution/aspect guard (upstream up to 1440x1440, aspect 1:2..2:1).

---

## Lance (ByteDance) image

### Reference sources
- Qwen/Qwen2.5-VL-3B-Instruct config.json (Lance backbone init); Lance paper arXiv:2605.18678. Upstream
  `Lance_3B/llm_config.json` not locatable.

### Architecture config — backbone bit-exact vs Qwen2.5-VL-3B
hidden 2048, layers 36, heads 16, head_dim 128, kv_heads 2, intermediate 11008, rms_norm_eps 1e-6,
rope_theta 1e6, mrope_section (16,24,24), vocab 151936, qk_norm false, all 7 special token ids — all MATCH.
VAE = Wan2.2 (16x spatial / 4x temporal).

### Findings (prioritized)
1. **`ImageTimestepShift=3.5` likely WRONG (HIGH)** — paper Table 2 lists shift=1.0 (pretraining) and **4.0**
   for CT/SFT/RL (released stages). 3.5 appears invented. `VideoTimestepShift=4.0` is correct; image inference almost certainly also 4.0. Change to 4.0 unless upstream image config overrides.
2. **MEDIUM: sampling/handoff fields UNVERIFIED placeholders** — `LatentPatchSize`(1,2,2), `MaxLatentSize`(32),
   `ConnectorActivation`, `TimestepFrequencyDim`(256), `NumTimesteps`(30), `VaeZChannels`(48) have no reference backing (code flags them validation-gated). Internally consistent but confirm against real safetensors.
3. MEDIUM: MaPE anchor `MapeGenTemporalBase=1000` matches paper "re-anchored to t=1000"; exact per-role offset application validation-gated. 3-way vision CFG (edit) not implemented (t2i-only, acceptable). CfgTextScale 4.0 MATCHES paper.

---

# VIDEO MODELS

> Cross-cutting video note: ALL video pipelines reuse the image-shaped `TextToImageRequest` (Steps=20,
> CfgScale=7.5, 512x512). These defaults are wrong for every video model (most want 40-50 steps,
> guidance 3-6, non-square video resolutions). Worse, several pipelines only fall back to their
> model-correct config default when the request value is `<=0` — which never happens — so the
> model-correct defaults are dead code. `num_frames` and `fps` have NO home in the request type and are
> passed as loose method args. A dedicated video request type (carrying num_frames, fps, and per-model
> step/guidance/size/flow_shift defaults, plus embedded-guidance) would close most user-facing gaps at once.

## Wan 2.2 TI2V-5B (T2V/I2V) + WanAnimate / WanS2V / WanVace

### Reference sources
- transformer/vae/scheduler config.json @ Wan-AI/Wan2.2-TI2V-5B-Diffusers, Wan2.2-Animate-14B, Wan2.2-S2V-14B,
  Wan2.1-VACE-14B; diffusers `pipeline_wan.py` / `pipeline_wan_i2v.py`.

### Architecture config — TI2V-5B + VACE MATCH
TI2V-5B transformer: patch [1,2,2], heads 24, head_dim 128, in/out 48, text_dim 4096, freq_dim 256,
ffn 14336, layers 30, eps 1e-6, rope_max_seq 1024, theta 10000 — all MATCH. VAE z_dim 48 / spatial 16 /
temporal 4 — MATCH. VACE-14B vace_layers / vace_in_channels 96 / layers 40 — MATCH. `BoundaryRatio` default
0 correctly makes TI2V-5B single-expert (A14B presets encode the dual-expert boundary 0.875/0.9).

### Findings (prioritized)
1. **Scheduler is Euler, reference is UniPCMultistepScheduler (WRONG, all variants)** — every Wan pipeline uses
   `EulerCfgStep`; TI2V-5B reference ships `UniPCMultistepScheduler` (solver_order 2, bh2, predict_x0,
   use_flow_sigmas, exponential time-shift). Pipeline doc-comment flags this as validation-gated. Biggest numerics risk.
2. **Request defaults never reach Wan config defaults (WRONG default-path)** — pipelines fall back to
   `_config.NumInferenceSteps`(50)/`GuidanceScale`(5.0) only when the request value is `<=0`, which never
   happens. Effective defaults = 20 steps / cfg 7.5 / 512x512 vs Wan 50 / 5.0 / 480x832.
3. **No Animate / S2V presets; variant-specific arch params entirely absent** — Animate needs
   `motion_encoder_dim=512`, `motion_dim=20`, `motion_encoder_size=512`, `face_encoder_hidden_dim=1024`,
   `face_encoder_num_heads=4`, `inject_face_latents_blocks=5`, `image_dim=1280`, `added_kv_proj_dim=5120`.
   S2V needs `audio_dim=1024`, `num_audio_token=4`, `audio_inject_layers=[0,4,8,...,39]`, `motion_token_num=1024`,
   `cond_dim=16`, `adain_mode="attn_norm"`. None are config fields. S2V_14B preset marked provisional/reconstructed.
4. **WanAnimate `added_kv_proj_dim` semantics WRONG** — our doc says "= ImageDim when set" (1280); reference
   `added_kv_proj_dim=5120` = inner dim (40x128), NOT image_dim.
5. **flow_shift HARDCODED per-config, not resolution-tied** — reference uses 5.0 (720p) / 3.0 (480p) chosen at runtime; no flow_shift request field.
6. MISSING: `text_len`(512) field; VAE `latents_mean`/`latents_std` (48-vec) not parameterized (baked in
   `Wan22VaeLatentNorm`); `fps` exposed nowhere (TI2V-5B is 24fps); `num_videos_per_prompt`(1); `request.Scheduler` ignored. TI2V last_image (FLF2V) only on concat path, not `GenerateFromEmbeddings`.

## LTX-Video (2B / 13B)

### Reference sources
- transformer/vae config.json @ Lightricks/LTX-Video & LTX-Video-0.9.7-distilled; diffusers `pipeline_ltx.py`;
  GitHub Lightricks/LTX-Video sampler YAMLs.

### Architecture config — 2B MATCH, 13B preset MISSING
2B (0.9.0): heads 32, head_dim 64, layers 28, in/out 128, cross_attn_dim 2048, caption 4096, norm_eps 1e-6,
VAE 32x/8x — MATCH. Only `V09` preset exists (hardcoded 2B).

### Findings (prioritized)
1. **No 13B/0.9.7 preset (WRONG/unsupported)** — 13B needs `HeadDim=128` (inner 4096), `NumLayers=48`,
   `CrossAttentionDim=4096`; our single `V09` is hardcoded 2B (64/28/2048). Add `V097` (0.9.8 shares arch).
2. **`decode_timestep`/`decode_noise_scale` MISSING** — 0.9.1+ and all 13B use a timestep-conditioned VAE that
   must decode at `decode_timestep=0.05`, `decode_noise_scale=0.025`. Our base VAE is built `timestepCond:false`
   and the pipeline passes nothing. Output on 0.9.1+ checkpoints wrong (the resnet block already supports it; wiring is the gap).
3. **STG (Spatiotemporal Skip Guidance) / `skip_block_list` / per-step guidance schedule MISSING** — official
   13B configs drive per-step `guidance_scale` arrays, `stg_scale` arrays, `rescaling_scale`, `guidance_timesteps`, `skip_block_list`. We only do constant 2-way CFG. Biggest 13B user-facing gap.
4. WRONG defaults via shared request (20/7.5 override LTX 50/3.0). `guidance_rescale` MISSING. `image_cond_noise_scale`(i2v) MISSING. RoPE base frames/H/W (20/2048/2048) hardcoded-but-correct.

## LTX-2 (22B audio+video)

### Reference sources
- transformer/config.json @ Lightricks/LTX-2; diffusers `LTX2Pipeline.__call__`.

### Architecture config — FULL MATCH (essentially exact)
Video stream (heads 32, head_dim 128, layers 48, in/out 128, cross_attn 4096, caption 3840 Gemma) AND
audio stream (head_dim 64, heads 32, in/out 128, cross_attn 2048, hop 160, sr 16000, scale_factor 4,
pos_embed_max 20) all MATCH. RoPE bases, causal_offset 1, both timestep-scale multipliers, vae_scale_factors
[8,32,32] — MATCH. Only `rope_double_precision=true` not explicitly modeled (precision detail, likely fine).

### Findings (prioritized)
1. **WRONG default steps** — our `NumInferenceSteps=50`; reference default = **40** (and request default 20 overrides via the `<=0` fallback bug).
2. **WRONG default guidance** — our fallback `GuidanceScale=3.0`; reference = **4.0** (request 7.5 overrides).
3. **MISSING: STG (`stg_scale`/`audio_stg_scale`), `modality_scale`/`audio_modality_scale`, `guidance_rescale`** —
   defaults (0 / 1 / 0) make the default path correct, but no way to enable them. STG + modality_scale are the headline LTX-2 quality knobs.
4. MISSING: `decode_timestep`/`decode_noise_scale`/`noise_scale` (default 0/None, ok at default). `max_sequence_length`(1024) — we pad to mult of 128 with no upper clamp.

## HunyuanVideo (T2V)

### Reference sources
- transformer/vae config.json @ hunyuanvideo-community/HunyuanVideo; diffusers `HunyuanVideoPipeline.__call__`.
  NOTE: our `HunyuanVideoConfig` ships ONLY GameCraft presets (33-ch); there is no plain-HunyuanVideo T2V preset or pipeline.

### Architecture config — VAE MATCH; transformer is GameCraft-shaped
VAE: latent 16, block_out [128,256,512,512], layers_per_block 2, groups 32, scaling 0.476986, spatial 8 /
temporal 4, mid_block_add_attention true — all MATCH. RoPE axes [16,56,56], theta 256, text_embed 4096,
pooled 768 — MATCH.

### Findings (prioritized)
1. **Embedded guidance entirely absent (CRITICAL for plain HunyuanVideo)** — reference `guidance_embeds=true` +
   `embedded_guidance_scale=6.0` feed guidance through a `guidance_in` MLP into modulation (distilled CFG).
   `HunyuanVideoConfig` has no guidance flag and `HunyuanVideoDit.Forward` has no guidance-embed path. Fine for
   GameCraft (real CFG 2.0) but plain HunyuanVideo cannot be correct without it.
2. **No plain HunyuanVideo T2V preset or pipeline (HIGH)** — only GameCraft presets (19 double / 38 single / 33-ch)
   exist; reference plain is 20 double / 40 single / 16-ch. Needs a new preset + a T2V pipeline (num_frames 129, embedded guidance 6.0, 720x1280).
3. MEDIUM: `qk_norm="rms_norm"` and `num_refiner_layers=2` not in config — DiT concedes the 2-layer txt refiner is stubbed to a plain projection (parity gap). Embedded-guidance scale 6.0, num_frames 129, true_cfg_scale all have no home.

## Kandinsky-5 video (T2V-Lite / Pro)

### Reference sources
- transformer/config.json @ kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers; diffusers `Kandinsky5T2VPipeline.__call__`.

### Architecture config — T2V-Lite-5s FULL MATCH
in/out_visual 16, time_dim 512, patch [1,2,2], model_dim 1792, ff_dim 7168, text_blocks 2, visual_blocks 32,
axes_dims [16,24,24] (head_dim 64, 28 heads), in_text 3584, in_text2 768, visual_cond true — all MATCH.

### Findings (prioritized)
1. **`attention_type` not modeled (MEDIUM)** — 5s config declares `attention_type:"regular"` (dense), so dense
   is correct for 5s; our code warns ">121 frames = NABLA-trained" keyed off frame count, which is misleading.
   The 10s checkpoints need Flex/sparse. Add an `attention_type` field and key the NABLA warning off it.
2. Default drift: steps 50, guidance 5.0, 512x768, 121 frames @ 24fps — all wrong via shared request.
3. LOW: VideoPro19B preset (model_dim 4096 / ff 16384 / 4 text + 60 visual / time_dim 1024) UNVERIFIED. Distilled (16 steps / cfg 1.0) has no preset. Scheduler shift 5.0 matches (single-file release reportedly 10.0).

## Lance video

### Reference sources
- bytedance/Lance GitHub README (T2V env vars/defaults). Backbone internals undisclosed (self-cited from upstream `Lance_3B/llm_config.json`, not fetchable).

### Findings (prioritized)
1. **`VideoTimestepShift=4.0` contradicts reference `VALIDATION_TIMESTEP_SHIFT=3.5` (HIGH)** — README documents
   3.5 for both image AND video; our config splits them (image 3.5 / video 4.0). The 4.0 appears invented; `RunDenoise` reads it directly, skewing the flow-match schedule. Change to 3.5 or document the source.
2. MEDIUM: no max-frame (121) enforcement; default num_frames (50) and 480p (480x848) resolution not represented; trained fps is 12 (not stored). Backbone numbers (hidden 2048, 36 layers, etc.) UNKNOWN/self-cited.
3. MATCH: `CfgTextScale=4.0`, `NumTimesteps=30`, VAE 48-ch / 16x / 4x.

---

# WORLD / INTERACTIVE (video-frame generation, included for completeness)

## Hunyuan-GameCraft

### Reference sources
- HF tencent/Hunyuan-GameCraft-1.0 README (CLI args) + paper arXiv 2506.17201. Reuses HunyuanVideo 13B backbone.

### Findings (prioritized)
1. **Reference defaults not encoded** — infer-steps 50 (standard) / 8 (distilled) and cfg-scale 2.0 / 1.0 are
   taken as bare args with no defaults; caller can silently diverge. Add standard/distilled presets.
2. **Action-speed range [0,3] documented but never enforced** (`speed` used raw); speed scaling + axis convention validation-gated.
3. **Chunk size 33 @ 25fps not pinned** — frame count derived from caller's latent T; no constant ties it to 33/25fps.
   Action space is WASD-only in the byte payload (reference unifies arrows/Space into camera space — architecturally consistent). CameraNet zero-init + temporal schedule validation-gated. Plucker channels 6, VAE 8/4 MATCH.

## Matrix-Game 2.0 / 3.0

### Reference sources
- Skywork/Matrix-Game-2.0 GitHub (`model.py`, `action_module.py`, `conditions.py`, `inference_universal.yaml`).
  Matrix-Game-3.0 configs gated/404 — most MG3 arch UNKNOWN (resolved at load via `InferShape`).

### Findings (prioritized)
1. **MG2 architecture FULL MATCH** — dim 1536, 30 layers, ffn 8960, heads 12, in 36 / out 16, patch [1,2,2],
   freq 256, CLIP ViT-H 1280/257 tokens, action module (hidden 128, stream 1024, heads 16, rope theta 256,
   window 3, mouse 2), VAE 8/4. Keyboard widths universal/gta/templerun = 4/2/7 confirmed exactly, CAM_VALUE 0.1, denoise list [1000,666,333], timestep_shift 5.0.
2. **MG2 `LocalAttnSize` (6 / gta 4) UNVERIFIED** — not exposed in fetched yaml. **TempleRun 4-step denoise list
   `[1000,750,500,250]` is OUR ASSUMPTION** (only the 3-step universal/gta list confirmed). Verify both.
3. MG2 `num_output_frames` default 150 not encoded; gta mouse is yaw-only in reference (pitch forced 0), ours feeds full 2-dim.
4. **MG3 almost entirely UNKNOWN/validation-gated** — DiT shape, Plucker dim (6144), action-block placement,
   steps (base 50 / distilled 3), guidance 5.0, shift 5.0, memory slots 5 / past 4 / first 15 / seg 10, move 0.05 / rotate 90 are all placeholders resolved at load or pending a checkpoint key-dump. Only Wan2.2-inherited values (z=48, umT5 4096, VAE 16/4) are solid.

## Oasis-500m — CLEAN MATCH

### Reference sources
- GitHub open-oasis `dit.py` (`DiT_S_2`), `generate.py`, `utils.py`.

### Findings
Near-total MATCH: hidden 1024, depth 16, heads 16, patch 2, in 16, mlp_ratio 4, external_cond_dim 25,
max_frames 32, input 18x32, spatial RoPE dim 32 / "pixel" / max_freq 256. Gen knobs: scaling 0.07843 (20/255),
10 DDIM steps, noise_abs_max 20, 25-dim VPT action (23 keys + 2 camera at slots 15/16), v-pred DDIM, zero-action
prepend — all MATCH. Only nit: `OasisPipeline` XML comment says stabilization level "14" but code uses 15 (code is correct per `generate.py`).

---

# NOT STARTED

## Cosmos-Predict1 Video2World (5B / 13B) — NOT BUILT

### Status
Zero V2W code in `src/` (the "Cosmos" hits are Anima = Predict2 image, plus generic `.pt`/FSQ-audio infra).
Research is COMPLETE: `docs/Research/COSMOS_PREDICT1_VIDEO2WORLD_ARCHITECTURE.md` (738 lines, full config + tensor
keys + algorithm). Every param below is MISSING (this is a build-spec, not a parity diff).

### Reference (resolved inference-time values; 5B / 13B differ only in body)
AR backbone (Llama3-shaped): n_layers 16/40, dim 4096/5120, n_heads 32, head_dim 128, n_kv_heads 8 (GQA-4),
ffn 14336 SwiGLU, rmsnorm eps 1e-5, vocab 64000, qk_norm true, 3D RoPE (theta ~500000), additive 3D abs-pos-emb,
cross_attn context_dim 1024 every layer (full, no causal, no RoPE), add_special_tokens false, training_type text_to_video.
DV tokenizer (`Cosmos-Tokenize1-DV8x16x16-720p`): FSQ levels [8,8,8,5,5,5] -> 64000 codebook, compression [8,16,16],
2-level Haar wavelet, ships ONLY as TorchScript `ema.jit` (no safetensors). Latent: pixel_chunk 33, latent_chunk 5,
video_latent_shape [5,40,64], max_seq_len 12800, valid input prefix [1,9] frames. Text encoder = T5-11B (1024-dim,
NOT T5-XXL 4096). Weights = `.pt` BF16 pickle. Optional 7B diffusion decoder (guidance 1.8, steps 15, sigma 8->0.02, overlap 2).
Gen knobs: temperature 0.6-1.0, top_p 0.8-0.9, top_k None, num_input_frames 1 or 9.

### Build blockers (prioritized)
1. **3D RoPE + additive 3D abs-pos-emb** — our `RopeFrequencyBuilder`/`RopeScaling` only do 1D/2D.
2. **DV tokenizer** — FSQ + Haar wavelet + causal 3D conv; TorchScript-only weights (needs JIT introspection or re-impl). Reusable for Phase-10 world models.
3. **T5-11B encoder (1024-dim)** — our T5-XXL plumbing is wrong-shaped (4096).
4. **`.pt` pickle -> safetensors converter** (config.json is a tiny dataclass dump, not AutoConfig — hardcode by size).
5. **AR-token KV-cache + packed causal attention** over 12,800 tokens (current `DenoiseKvCache` is diffusion-prefix-oriented).
5B<->13B differ only in body (16Lx4096 vs 40Lx5120); cross-attn adapter identical. Resolution/frames/fps hardcoded in reference (1024x640, 33 frames, 25fps).

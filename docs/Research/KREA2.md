# Krea 2 — Research & Implementation Plan

> Status: **research complete, implementation not started.** This document is the architecture +
> build plan for Krea 2. Verified against the released checkpoints (`krea/Krea-2-Turbo`,
> `CalamitousFelicitousness/Krea-2-Base-Diffusers`) and the canonical diffusers source
> (`Krea2Transformer2DModel` / `Krea2Pipeline` in `huggingface/diffusers`, plus the Krea 2 technical report).

## 1. Overview

Krea 2 ("K2", released open-weights under the Krea 2 Community License) is a **12.9B (officially "12B")
single-stream MMDiT flow-matching text-to-image** model. Two checkpoints share one architecture:

| Variant | `is_distilled` | Steps | Guidance |
|---|---|---|---|
| **Base** (midtrain) | false | 28 | CFG 4.5 (standard dual-pass) |
| **Turbo / TDM** (distilled) | true | 8 (min 4) | guidance off (CFG 0–1, single pass) |

A Turbo **LoRA** is also published (converts Base → few-step). Supported resolution 128–4096.

Components (`model_index.json`):
* transformer — `Krea2Transformer2DModel` (diffusers)
* text encoder — **Qwen3-VL-4B** (`Qwen3VLModel`, hidden 2560), tokenizer `Qwen2Tokenizer`
* VAE — **Qwen-Image VAE** (`AutoencoderKLQwenImage`, 16 latent channels, f8, per-channel `latents_mean/std`)
* scheduler — `FlowMatchEulerDiscreteScheduler` with resolution-dependent exponential shift

SwarmUI compat note: SwarmUI auto-downloads Qwen3-VL-4B + the Qwen-Image VAE; the user picks the diffusion
model (fp8 / nv4 / bf16). VAE family = `VaeQwenImage`. Built-in soft content filtering (prompt-level).

## 2. Transformer (`Krea2Transformer2DModel`)

Diffusers config (defaults match the released Base checkpoint; the raw checkpoint config uses the short
names in parentheses):

```
in_channels            64        (= vae_channels 16 · patch² 4)   img_in: Linear(64 → 6144)
num_layers             28        (layers)
attention_head_dim     128
num_attention_heads    48        (heads)         hidden_size = 128·48 = 6144 (features)
num_key_value_heads    12        (kvheads)       GQA group 4
intermediate_size      16384     (≈ multiplier 4 · 6144 · 2/3, SwiGLU)
timestep_embed_dim     256       (tdim)
text_hidden_dim        2560      (txtdim, Qwen3-VL-4B hidden)
num_text_layers        12        (txtlayers)     tapped encoder hidden states per token
text_num_attention_heads      20 (txtheads)
text_num_key_value_heads      20 (txtkvheads)    no GQA in text fusion
text_intermediate_size 6912      (≈ 4 · 2560 · 2/3)
num_layerwise_text_blocks 2
num_refiner_text_blocks   2
axes_dims_rope         (32, 48, 48)   sum = head_dim 128, axes (t, h, w)
rope_theta             1000.0    (theta)         FluxPosEmbed convention (use_real, repeat_interleave_real)
norm_eps               1e-5
bias                   false (all attn/ffn linears bias-free; img_in/time/txt projections have bias)
```

### 2.1 Distinctive components

* **Zero-centered RMSNorm** (`Krea2RMSNorm`): effective scale is `weight + 1` (`F.rms_norm(x, weight=weight+1)`),
  upcast to f32. → maps to the engine's existing **`RmsNormScalePlusOne`** path (Gemma-style +1). Used for all
  block norms and per-head Q/K norm.
* **Sigmoid output-gate attention** (`Krea2AttnProcessor`): standard GQA + per-head RMSNorm Q/K + RoPE, then a
  learned `to_gate = Linear(hidden, hidden)` applied as `attn_out = sdpa(...) * sigmoid(to_gate(x))` before `to_out`.
  (New op — a per-token elementwise sigmoid gate on the attention output.)
* **6-way scale_shift modulation** (`Krea2TransformerBlock`): one shared `temb_mod = time_mod_proj(gelu(temb))`
  of width `6·hidden`, plus a per-block learned `scale_shift_table[6, hidden]`:
  ```
  prescale, preshift, pregate, postscale, postshift, postgate = (temb_mod + table).unbind
  h = h + pregate · attn((1+prescale)·norm1(h) + preshift)
  h = h + postgate · ff((1+postscale)·norm2(h) + postshift)
  ```
  (Gates are NOT tanh'd; raw multiplicative gates. No adaLN MLP per block — just the additive table.)
* **SwiGLU** (`Krea2SwiGLU`): `down(silu(gate(x)) · up(x))`, bias-free, inner 16384.

### 2.2 Text-fusion stage (`Krea2TextFusion`)

Input: `[B, txt_seq, 12, 2560]` (12 tapped encoder hidden states stacked per token). Output `[B, txt_seq, 2560]`:
1. **2 layerwise blocks** — reshape to `[B·txt_seq, 12, 2560]`, pre-norm `Krea2TextFusionBlock`s (RMSNorm → GQA self-attn
   (no RoPE, no time mod) → RMSNorm → SwiGLU) attend **across the 12-layer axis** per token.
2. **projector** — `Linear(12 → 1, bias=false)` after `permute` to `[B, txt_seq, 2560, 12]`, collapsing the layer axis.
3. **2 refiner blocks** — same block type, attend **across the token axis** (with the text key-padding mask).

Then `txt_in` = `Krea2TextProjection`: `RMSNorm(2560) → Linear(2560→6144) → gelu_tanh → Linear(6144→6144)`.

### 2.3 Forward

```
temb     = time_embed(timestep)                 # sinusoidal(t·1000, cos-first, dim 256) → Linear(256→6144) → gelu_tanh → Linear(6144→6144); shape [B,1,6144]
temb_mod = time_mod_proj(gelu_tanh(temb))        # Linear(6144 → 6·6144); shared block modulation
txt = txt_in(text_fusion(encoder_hidden_states)) # [B, txt_seq, 6144]
img = img_in(patchify(latent))                   # [B, img_seq, 6144]
x   = concat([txt, img], dim=1)                  # [B, txt_seq+img_seq, 6144]
rope = rotary(position_ids)                       # [seq,3]: text rows (0,0,0); image rows (0,row,col)
for block in 28: x = block(x, temb_mod, rope, mask)
img = x[:, txt_seq:]                              # strip text prefix
out = final_layer(img, temb)                      # (1+scale)·RMSNorm + shift, scale/shift from temb + table[2,6144]; Linear(6144→64)
velocity = unpatchify(out)                        # [B, 16, H, W]
```

`final_layer` uses `temb` (not `temb_mod`). Flow-matching **v-prediction**, t∈[0,1] (1 = noise, 0 = data).

## 3. Text encoder (Qwen3-VL-4B)

Qwen3-VL-4B **language tower** (hidden 2560, 36 layers; vision unused — Krea 2 is T2I). Tokenizer `Qwen2Tokenizer`.

Prompt template (Qwen-Image style), padded **in the middle** `[prefix | prompt | PAD | suffix]`:
```
prefix  = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
suffix  = "<|im_end|>\n<|im_start|>assistant\n"
drop the first 34 tokens (the system prefix) after encoding; 5 suffix tokens.
```
Encoder runs with `output_hidden_states=True`; **stack** the 12 selected layers
`(2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)` (HF `hidden_states` indices, 0 = embeddings) along a new layer
axis → `[B, txt_seq, 12, 2560]`. Encoder M-RoPE positions = cumulative valid-token count (T/H/W equal for text).

## 4. VAE — Qwen-Image (`AutoencoderKLQwenImage`)

16-channel f8 autoencoder. Decode un-normalizes per channel: `latent = latent / latents_std + latents_mean`
(`z_dim = 16`, per-channel `latents_mean`/`latents_std` from the VAE config). Identical to the VAE Qwen-Image
and Anima already use.

## 5. Scheduler

`FlowMatchEulerDiscreteScheduler`, **resolution-aware exponential shift** (Flux `calculate_shift`):
`mu = m·image_seq_len + b` over the line `(base_image_seq_len 256 → base_shift 0.5) … (max_image_seq_len 6400 → max_shift 1.15)`,
`shift = exp(mu)`. Turbo (distilled) pins `mu = 1.15`. Sigmas standard Flux/SD3 (descending t → 0).
→ the engine's existing `FlowMatchEulerDiscreteScheduler.CreateWithDynamicShift(imageSeqLen, baseSeqLen:256,
maxSeqLen:6400, baseShift:0.5f, maxShift:1.15f)`.

## 6. Guidance

* **Base**: standard CFG, dual pass (cond / uncond from negative prompt), `guidance_scale 4.5`,
  `pred = uncond + 4.5·(cond − uncond)`.
* **Turbo/TDM**: single pass, no CFG (`guidance_scale 0`), 8 steps.

## 7. Reuse map (HartsyInference)

| Component | Plan |
|---|---|
| Qwen3-VL-4B language tower | **reuse** `LlamaStyleEncoder`; **add** a `Qwen3_VL_4B` preset (= `Qwen3_4B` with `RopeTheta 5e6`) |
| 12-layer hidden-state tap | **reuse** `EncodeMultiLayer((2,5,…,35))`, then view `[B,S,12·2560] → [B,S,12,2560]` (tap-major) |
| Qwen-Image VAE (decode + encode) | **reuse** `QwenImageVaeDecoder` / `QwenImageVaeEncoder` |
| Flow-match Euler + resolution shift | **reuse** `FlowMatchEulerDiscreteScheduler.CreateWithDynamicShift` (256/6400/0.5/1.15) |
| 3-axis RoPE (Flux `use_real` convention) | **reuse** `FluxRope` with axes `[32,48,48]`, theta 1000 |
| SwiGLU FFN, GQA repeat-interleave, QK-norm | **reuse** `DiTUtils` / `QkNorm` patterns |
| Zero-centered RMSNorm (`weight+1`) | **reuse** `RmsNormScalePlusOne` fold (add-one-at-load) |
| Qwen tokenizer + Qwen-Image chat template | **reuse** `Qwen3Tokenizer` (the QwenImage/Hunyuan template + 34-token prefix drop) |
| **Krea2 transformer** (img_in, time_embed + time_mod_proj, 6-way modulation, sigmoid-gate attention, text-fusion, txt_in, final_layer) | **new** |
| Sigmoid output-gate attention | **new** small op: `attn_out · sigmoid(to_gate(x))` |
| Text-fusion stage (layerwise + projector + refiner) | **new** small transformer (`Krea2TextFusion`) |
| Config, converter, pipeline, tests | **new** Krea2-specific |

## 8. Build plan

Mirror the Ideogram4 / Z-Image build pattern. Files under `src/HartsyInference.Diffusion/`:

1. **`Models/Denoisers/Krea2Config.cs`** — record + `Base`/`Turbo` presets (the §2 numbers; `IsDistilled` flag only
   changes the scheduler-shift mode, not geometry).
2. **`Models/Denoisers/DiTBlocks/Krea2Block.cs`** — `Krea2TransformerBlock`: 6-way `scale_shift_table` modulation,
   GQA attention with per-head zero-center RMSNorm + RoPE + **sigmoid output gate**, SwiGLU. (Hoist the sigmoid-gate
   attention so the text-fusion block can share it.)
3. **`Models/Denoisers/DiTBlocks/Krea2TextFusion.cs`** — `Krea2TextFusionBlock` (pre-norm, no RoPE, no modulation) +
   `Krea2TextFusion` (2 layerwise → `Linear(12→1)` projector → 2 refiner).
4. **`Models/Denoisers/Krea2Transformer.cs`** — orchestrates: `img_in`, `time_embed` (+ `time_mod_proj`),
   `text_fusion` + `txt_in`, RoPE position ids (text=0, image grid), 28 blocks, strip-text, `final_layer`,
   patchify/unpatchify (reuse `DiTUtils.PatchifyNCHW`/`UnpatchifyToNCHW`).
5. **`Schedulers`** — none new (reuse `CreateWithDynamicShift`); add a `mu=1.15` fixed-shift ctor path if not present
   (Turbo).
6. **`Models/TextEncoders/LlamaStyleEncoderConfig.cs`** — add `Qwen3_VL_4B` preset.
7. **`Pipelines/Krea2Pipeline.cs`** — `GenerateFromTokens` (encode 12-layer tap → text fusion is *inside* the
   transformer, so the pipeline passes the `[B,S,12,2560]` stack; build image+text position ids; CFG dual-pass for
   Base, single-pass for Turbo; Qwen-Image VAE decode).
8. **`CheckpointConverters/Krea2CheckpointConverter.cs`** — diffusers keys (`transformer_blocks.{i}.*`,
   `text_fusion.*`, `img_in`, `time_embed`, `time_mod_proj`, `txt_in`, `final_layer`); fp8 scaled-dequant; the raw
   (non-diffusers) checkpoint key remap if the released single-file uses the short-name keys.
9. **Tests** — `tests/python-reference` parity dumps (RoPE, block, text-fusion, full forward). Per the
   2026-08-06 cleanup rule, do not add per-model forward or generation tests; a broken model is visible in
   production. Krea2's sharding/placement coverage lives in the multi-GPU campaign.
10. **SwarmUI extension** — `Krea2Loader` + compat `"krea-2"` (core `T2IModelClassSorter`: detect by the
    `text_fusion.projector.weight` + `transformer_blocks.0.attn.to_gate.weight` keys; VaeFamily `VaeQwenImage`); side
    models Qwen3-VL-4B + Qwen-Image VAE; cache + dispatch + validation; expose a Base/Turbo step→preset mapping.

Estimated net-new surface: ~1 transformer + 2 block files + 1 pipeline + 1 converter + 1 config + tests, with the
encoder / VAE / scheduler / RoPE / SwiGLU all reused. Comparable in size to the Z-Image build.

## 9. Open items / to confirm at implementation time

* Exact released single-file key naming (Comfy fp8/nv4) vs the diffusers folder keys — the converter must handle both
  (`features`/`heads`/… short config vs diffusers `Krea2Transformer2DModel`).
* `time_mod_proj` is applied to `gelu_tanh(temb)` then added to each block's table — confirm the gelu is on the
  *modulation* projection input (it is, per `temb_mod = time_mod_proj(F.gelu(temb))`).
* Whether the released Turbo checkpoint sets `is_distilled=true` and a baked `mu=1.15`, vs. relying on the pipeline.
* Numeric parity (the usual validation-pending gate) once weights are downloaded.

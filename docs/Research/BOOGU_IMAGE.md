# Boogu-Image-0.1 — Research & Implementation Notes

> Status: research complete; implementation in progress. This document is the single source of
> truth for the Boogu-Image port. All numbers were verified against the released checkpoints
> (`Boogu/Boogu-Image-0.1-Base`, `-Edit`, `-Turbo`) and the canonical Python source at
> `github.com/boogu-project/Boogu-Image` (`boogu/models/transformers/transformer_boogu.py`,
> `block_lumina2.py`, `rope.py`, `attention_processor.py`, the flow-match scheduler, and
> `pipeline_boogu.py`).

## 1. Overview

Boogu-Image-0.1 (released 2026-06-16, Apache-2.0) is a **10B unified image generation + editing**
model family:

| Variant | Task | Steps | Guidance |
|---|---|---|---|
| `Base` | text-to-image (T2I) | 25–50 | `text_guidance_scale` 2.0–5.0 (single CFG) |
| `Edit` | text+image-to-image (TI2I) | 25–50 | double guidance: `text_guidance_scale` + `image_guidance_scale` |
| `Turbo` | distilled T2I | 4 | `text_guidance_scale = 1.0` (no CFG) |
| `*-fp8` | quantized weights | — | same; fp8 scaled dequant |

It is an **OmniGen2 / Lumina-Image-2.0 lineage** diffusion transformer (the upstream `rope.py`
credits the OmniGen2 team) with three things layered on top of the OmniGen2 architecture we already
have in the engine:

1. an **8-block double-stream (dual-stream) stage** before the single-stream stack,
2. a **reference-image edit path** (`ref_image_patch_embedder` + `ref_image_refiner` +
   `image_index_embedding`), and
3. a **Qwen3-VL-8B multimodal** text encoder (vision tower *is* used for editing).

Text encoder: **full Qwen3-VL-8B** (not just the language tower). VAE: the **FLUX.1-dev VAE**
(16 latent channels, `scaling_factor = 0.3611`, `shift_factor = 0.1159`, no quant convs) —
identical to what Flux/OmniGen2 already use.

## 2. Transformer architecture (`BooguImageTransformer2DModel`)

`transformer/config.json` (identical across Base and Edit):

```
hidden_size               3360
num_layers                40           # total
num_double_stream_layers  8            # → 32 single-stream layers
num_refiner_layers        2            # each of context / noise / ref-image refiner
num_attention_heads       28
num_kv_heads              7            # GQA, group size 4
head_dim                  120          # 3360/28 == 40+40+40
patch_size                2
in_channels               16           # FLUX VAE latent
axes_dim_rope             [40, 40, 40] # (time, height, width)
axes_lens                 [2048, 1664, 1664]
theta                     10000
multiple_of               256          # SwiGLU FFN rounding
norm_eps                  1e-5
timestep_scale            1000.0
instruction_feature_configs: { instruction_feat_dim 4096, num_instruction_feature_layers 1, reduce_type "mean" }
prompt_tuning_configs: { use_prompt_tuning false, num_layers 0 }   # disabled in the release
```

`hidden_size // num_attention_heads == sum(axes_dim_rope)` (120) is asserted upstream.
Conditioning dim for all modulation linears is `min(hidden, 1024) = 1024`.

### 2.1 Forward chain (`forward`)

```
instruction_hidden_states := mean over the tapped MLLM layers   # 1 layer → identity, [B, T, 4096]
temb, caption := time_caption_embed(timestep, instruction_hidden_states)
        # temb  : [B, 1024]  (sinusoidal(t*1000) → SiLU MLP)
        # caption: [B, T, hidden]  (RMSNorm(4096) → Linear(4096→hidden))
(padded latents, padded ref-img, masks, l_eff_ref, l_eff_img, sizes) := flat_and_pad_to_seq(...)
(context_rope, ref_img_rope, noise_rope, joint_rope, cap_lens, seq_lens,
 combined_img_rope, combined_img_lens) := rope_embedder(freqs_cis, instr_mask, ...)

# context refinement (text): 2 × non-modulated block, Text RoPE
for layer in context_refiner: caption = layer(caption, instr_mask, context_rope)

# image patch embed + refinement:
img  = x_embedder(patchify(latent))                         # Linear(p²·16 → hidden)
ref  = ref_image_patch_embedder(patchify(ref_latents))     # Linear(p²·16 → hidden)  (edit only)
ref[j-th image] += image_index_embedding[j]                # distinguishes up to 5 ref images
for layer in noise_refiner:    img = layer(img, img_mask, noise_rope, temb)        # modulated, Image RoPE
for layer in ref_image_refiner: ref = layer(ref, ref_mask, ref_img_rope, temb)     # modulated, Image RoPE
combined_img = concat([ref, img], seq)                      # ref first, then noise image

# dual-stream stage (8 blocks): instruction stream + image stream, joint attention
instruct = caption ; image = combined_img
for layer in double_stream_layers:
    image, instruct = layer(image, instruct, img_mask, joint_mask,
                            combined_img_rope, joint_rope, temb, cap_lens, seq_lens)

# fuse → single-stream stack (32 blocks):
joint = concat([instruct, image], seq)                      # text first, then image
for layer in single_stream_layers: joint = layer(joint, joint_mask, joint_rope, temb)

# output:
joint = norm_out(joint, temb)                               # LuminaLayerNormContinuous → p²·out_channels
velocity = unpatchify(image-tail of joint)                  # [B, 16, H, W]
```

Note ordering differences inside the joint sequence: the **double-stream** stage concatenates
`[ref_img, noise_img]` for the image stream and conceptually `[instruct | image]` for the joint
attention; the **single-stream** stage uses `[instruct, ref_img, noise_img]`. The output keeps only
the trailing `noise_img` tokens.

### 2.2 Blocks

All blocks are **Lumina/OmniGen2 sandwich-norm** blocks. Per-stream the structure is:

```
(norm_h, gate_msa, scale_mlp, gate_mlp) = norm1(h, temb)            # LuminaRMSNormZero (shift=None)
h = h + tanh(gate_msa) · norm2(attn(norm_h))                        # norm2 = RMSNorm on attn output
h = h + tanh(gate_mlp) · ffn_norm2(swiglu(ffn_norm1(h)·(1+scale_mlp)))
```

* **single_stream_layers** (32), **noise_refiner** (2), **ref_image_refiner** (2): modulated, the
  block above. → **identical to our existing `OmniGen2Block(modulation: true)`.**
* **context_refiner** (2): same without gates/scale (`modulation: false`). → **`OmniGen2Block(modulation: false)`.**
* **double_stream_layers** (8): the only genuinely new block — see §2.3.

Attention everywhere: GQA (28 Q heads / 7 KV heads, K/V repeat-interleaved to 28 before SDPA),
per-head RMSNorm on Q and K (`qk_norm="rms_norm"`, eps 1e-5), RoPE applied to Q/K
(`use_real=False` Lumina convention: adjacent feature pairs are `(real, imag)`, multiplied by
`cos + i·sin`). SwiGLU FFN = three bias-free linears, inner dim `round_up(4·hidden, 256)`.

### 2.3 Double-stream block (`BooguImageDoubleStreamTransformerBlock`)

Two parallel streams (image, instruction) with **three sub-layers** on the image stream and two on
the instruction stream. Three separate modulations are produced for the image stream
(`img_norm1/2/3`) and two for the instruction stream (`instruct_norm1/2`):

```
img_n1, img_gate_msa, img_scale_mlp, img_gate_mlp = img_norm1(image, temb)
img_n2, img_shift_mlp, _, _                       = img_norm2(image, temb)
img_n3, img_gate_self, _, _                       = img_norm3(image, temb)
ins_n1, ins_gate_msa, ins_scale_mlp, ins_gate_mlp = instruct_norm1(instruct, temb)
ins_n2, ins_shift_mlp, _, _                       = instruct_norm2(instruct, temb)

# (1) joint cross-attention over [instruct || image] with its OWN q/k/v projections:
joint = img_instruct_attn.processor(img=img_n1, instruct=ins_n1, joint_mask, joint_rope, cap_lens, seq_lens)
ins_attn, img_attn = split(joint)                  # instruct part / image part (separate out projections)

# (2) image self-attention (standard block attn) over image tokens only:
img_self = img_self_attn(img_n3, img_mask, combined_img_rope)

# (3) residual updates:
image    = image    + tanh(img_gate_msa)  · img_attn_norm(img_attn)
image    = image    + tanh(img_gate_self) · img_self_attn_norm(img_self)
image    = image    + tanh(img_gate_mlp)  · img_ffn_norm2( img_ff( img_ffn_norm1( (1+img_scale_mlp)·img_n2 + img_shift_mlp ) ) )
instruct = instruct + tanh(ins_gate_msa)  · instruct_attn_norm(ins_attn)
instruct = instruct + tanh(ins_gate_mlp)  · instruct_ffn_norm2( instruct_ff( instruct_ffn_norm1( (1+ins_scale_mlp)·ins_n2 + ins_shift_mlp ) ) )
```

The **joint cross-attention** (`img_instruct_attn`) is a dedicated processor: it has its own
`img_to_{q,k,v}`, `instruct_to_{q,k,v}`, `img_out`, `instruct_out` linears (the parent `Attention`'s
`to_q/k/v` are deleted; only `to_out.0` is kept). Q/K/V are computed separately per stream, then
concatenated as `[instruct, image]` per sample (respecting `cap_lens` / `seq_lens`), GQA + qk-norm +
joint RoPE applied, one SDPA over the full joint sequence, then split back and projected by
per-stream `img_out` / `instruct_out`, then the parent `to_out.0`.

The image self-attention (`img_self_attn`) is an ordinary block attention identical to `OmniGen2Block`'s.

**Weight keys** for `double_stream_layers.{i}`:
`img_self_attn.{to_q,to_k,to_v,to_out.0,norm_q,norm_k}.weight`,
`img_instruct_attn.{to_out.0,norm_q,norm_k}.weight` and
`img_instruct_attn.processor.{img_to_q,img_to_k,img_to_v,instruct_to_q,instruct_to_k,instruct_to_v,img_out,instruct_out}.weight`,
`img_feed_forward.linear_{1,2,3}.weight`, `instruct_feed_forward.linear_{1,2,3}.weight`,
`img_norm1/2/3.{linear,norm}.weight(+linear.bias)`, `instruct_norm1/2.{linear,norm}.weight(+linear.bias)`,
`img_ffn_norm1/2.weight`, `img_attn_norm.weight`, `img_self_attn_norm.weight`,
`instruct_ffn_norm1/2.weight`, `instruct_attn_norm.weight`.

### 2.4 Time / caption embedding (`Lumina2CombinedTimestepCaptionEmbedding`)

```
time_proj          = Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
timestep_embedder  = Linear(256→1024) → SiLU → Linear(1024→1024)      # → temb [B,1024]
caption_embedder   = RMSNorm(4096) → Linear(4096→hidden)              # → caption [B,T,hidden]
```

Keys: `time_caption_embed.timestep_embedder.linear_{1,2}.{weight,bias}`,
`time_caption_embed.caption_embedder.0.weight` (RMSNorm), `time_caption_embed.caption_embedder.1.{weight,bias}`.
(Differs from OmniGen2, which uses `time_proj.{0,2}` and a bare `caption_embedder` Linear.)

### 2.5 Output norm (`LuminaLayerNormContinuous`)

```
scale  = Linear_1(SiLU(temb))                # 1024 → hidden
x      = LayerNorm(x, elementwise_affine=False, eps=1e-6) · (1 + scale)
out    = Linear_2(x)                          # hidden → p²·out_channels (= 4·16 = 64)
```

Keys: `norm_out.linear_1.{weight,bias}`, `norm_out.linear_2.{weight,bias}`.
(Differs from OmniGen2's RMSNorm-zero `norm_out.norm.weight` + `norm_out.linear`.)

### 2.6 RoPE position assignment (`rope.py`)

Frequencies are built exactly like OmniGen2 (`get_1d_rotary_pos_embed(d, e, theta)` per axis,
concatenated; adjacent-pair complex rotation). Position ids per token:

* **text token i**: `(i, i, i)`.
* **ref image** (edit): for each ref image, `axis0 = pe_shift` (a running offset, starts at
  `cap_len`, advances by `max(ref_H_tokens, ref_W_tokens)` after each ref image), `axis1 = row`,
  `axis2 = col`.
* **noise image**: `axis0 = pe_shift` (after all ref images), `axis1 = row`, `axis2 = col`.

For pure **T2I (no ref images)** this collapses to: text `(i,i,i)`, image `(cap_len, row, col)` —
**identical to our `OmniGen2Rope`** image mode (`timeOffset = cap_len`) and joint mode.

## 3. Scheduler (`FlowMatchEulerDiscreteScheduler`, time-shifting)

`scheduler/scheduler_config.json`: `do_shift true, dynamic_time_shift false, time_shift_version
"v1", seq_len 4096, num_train_timesteps 1000`.

Static **v1** shift (the released setting):

```
t_arr = linspace(0, 1, num_steps+1)[:-1]                       # descending sigmas
mu    = lin(seq_len)         where lin maps (256 → 0.5) … (4096 → 1.15)   # mu(4096) ≈ 1.15
# logistic transform per element:
t1  = clip(1 - t, eps, 1-eps);  y = exp(mu) / (exp(mu) + (1/t1 - 1)^sigma);  t' = 1 - y   (sigma=1)
timesteps = t';  _timesteps = cat(t', [1.0])
```

Euler step: `prev = sample + (t_next - t) * model_output` (upcast to f32). Velocity prediction,
flow-matching (no inversion/negation needed; the model predicts the field directly here, unlike
Lumina-2.0). Our `FlowMatchEulerDiscreteScheduler` already has Euler + an exponential (`exp(mu)`)
dynamic shift; we add the **v1 logistic static** path (`mu = lin(seq_len)` with the (256,0.5)→(4096,1.15)
line, then the `1 - exp(mu)/(exp(mu)+(1/(1-t)-1))` map).

The Turbo variant uses 4 steps and the same scheduler family with `text_guidance_scale = 1.0`.

## 4. Guidance

* **T2I (Base/Turbo)** — single CFG. `model_pred(cond)` vs `model_pred(neg)`:
  `pred = pred_cond + (text_guidance_scale − 1)·(pred_cond − pred_neg)`.
  (Equivalent to `neg + tg·(cond − neg)`.) Turbo: `tg = 1` → just `pred_cond`.
* **Edit (TI2I)** — double guidance, three model passes per step:
  * `cond`      = text + ref image,
  * `drop_text` = negative text + ref image,
  * `drop_all`  = negative text + **no** ref image,
  * `delta_text  = cond − drop_text`, `delta_image = drop_text − drop_all`,
  * `pred = cond + (text_guidance_scale − 1)·delta_text + (image_guidance_scale − 1)·delta_image`.
  Defaults: `text_guidance_scale 4.0`, `image_guidance_scale 1.0` (set `image_guidance_scale` > 1
  to strengthen reference adherence). When `image_guidance_scale == 1` the `drop_all` pass is skipped
  (text-only guidance with the reference kept). `empty_instruction_guidance_scale` and boosted
  orthogonal guidance are optional extras (deferred).

## 5. Text encoder — Qwen3-VL-8B (multimodal)

The MLLM encodes a chat-templated instruction (plus reference images for edit) and the transformer
consumes its hidden states:

```
prompt   = chat_template(system_prompt[task], <image> tokens…, instruction)
vlm_in   = processor(prompt, images)            # input_ids, attention_mask, pixel_values, image_grid_thw
hidden   = mllm(vlm_in, output_hidden_states=True).hidden_states
instr    = mean(last num_instruction_feature_layers hidden states)   # = final hidden state (1 layer)
```

`instr` (shape `[B, L, 4096]`) and `attention_mask` feed the transformer as
`instruction_hidden_states` / `instruction_attention_mask`. `num_instruction_feature_layers = 1` and
`reduce_type = "mean"` ⇒ the conditioning is simply the **final decoder hidden state** over the full
templated sequence (text + any image tokens).

System prompts (verbatim from `pipeline_boogu.py`):
* T2I: `"You are a helpful assistant that generates high-quality images based on user instructions. The instructions are as follows."`
* TI2I: `"Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."`

For **editing** the reference image is fed *both* to the Qwen3-VL **vision tower** (so the
conditioning text "sees" the image) and to the DiT via VAE latents (`ref_image_patch_embedder`).
This requires a Qwen3-VL vision tower, which the engine did not previously have (see §7).

VAE latent scaling (FLUX): encode `z = (vae.encode(x).mean − 0.1159) · 0.3611`; decode
`x = vae.decode(z / 0.3611 + 0.1159)`.

## 6. Reuse map (what we build vs. what already exists)

| Component | Plan |
|---|---|
| FLUX VAE encode/decode | **reuse** `VaeEncoder` / `VaeDecoder` + `VaeConfig.Flux` |
| Qwen3-VL **language tower** | **reuse** `LlamaStyleEncoder` (`Qwen3_VL_8B` config), tap = final hidden state |
| Single-stream / noise / ref / context blocks | **reuse** `OmniGen2Block` (modulation true/false) |
| 3-axis RoPE (T2I) | **reuse** `OmniGen2Rope`; add a position-id table builder for the edit ref-image offsets |
| GQA + qk-norm + SDPA, SwiGLU, patchify, timestep sinusoid | **reuse** `DiTUtils`, `QkNorm`, `IBackend` ops |
| Flow-match Euler scheduler | **reuse** `FlowMatchEulerDiscreteScheduler`; **add** v1 logistic static shift |
| Pipeline base, CFG helper, dtype casts | **reuse** `DiffusionPipelineBase`, `CfgHelper`, `DtypeCastHelper` |
| **Double-stream block** | **new** `BooguImageDoubleBlock` |
| Time+caption embed, output norm (Lumina2 variants) | **new** small methods on `BooguImageTransformer` (different keys/shapes than OmniGen2) |
| Ref-image edit path (patch embed, ref refiner, index embed, concat order) | **new** in `BooguImageTransformer` |
| **Qwen3-VL vision tower** | **new** reusable `HartsyInference.Vision`/text-encoder subsystem (also unblocks Lance/OmniGen2 edit) |
| Config, converter, pipelines (T2I + Edit + Turbo), tests | **new** Boogu-specific |

## 7. Qwen3-VL vision tower (new subsystem)

No Qwen-VL vision tower existed in the engine (an acknowledged "deferred" gap referenced by Lance and
OmniGen2). Building it here is reusable across those models. Required pieces:

* image preprocessing: smart-resize to multiples of (patch·merge) with a pixel budget, RGB
  normalize (OpenAI CLIP mean/std), temporal patch (single frame ⇒ duplicated to temporal patch
  size), produce `pixel_values` + `image_grid_thw`;
* patch embed (Conv3d/linear over `temporal·patch·patch·3`), the vision blocks (full attention with
  2D vision RoPE over the grid, SwiGLU/GELU MLP, RMSNorm), window/full attention as per Qwen3-VL,
  and the **patch merger** (spatial-merge of 2×2 patches → LM hidden dim);
* merge: replace the `<image>` placeholder tokens in the LM input-embeds with the merged vision
  tokens, then run the language tower (`LlamaStyleEncoder`) over the combined inputs-embeds with
  multimodal M-RoPE position ids, and tap the final hidden state.

Config (verified from `Boogu/Boogu-Image-0.1-Base/mllm/config.json`):

```
model_type             qwen3_vl                 image_token_id   151655
vision_config:
  depth                27                       hidden_size      1152
  num_heads            16  (head_dim 72)        intermediate     4304   (hidden_act gelu_pytorch_tanh)
  patch_size           16  spatial_merge_size 2  temporal_patch_size 2   in_channels 3
  num_position_embeddings 2304  (learned, interpolated to the grid)
  out_hidden_size      4096                     deepstack_visual_indexes [8, 16, 24]
text_config:
  hidden 4096  layers 36  heads 32  kv_heads 8  head_dim 128  rms_eps 1e-6
  rope_theta 5e6  rope_scaling.mrope_section [24,20,20]  mrope_interleaved true
  vision_start/end tokens 151652 / 151653
```

Implementation notes (the parity-sensitive parts):
* **patch embed**: Conv3d over `temporal_patch_size·patch·patch·3 → 1152`; a single still image is repeated to fill the temporal patch (t=1 grid).
* **2D windowing + RoPE**: Qwen3-VL vision uses full attention with 2D positional RoPE over the
  `(h, w)` grid; merge size 2 groups patches for the merger.
* **deepstack**: features from vision layers `[8, 16, 24]` are *also* injected into the language model
  (added to the corresponding image-token positions) — the LM forward must accept these extra inputs,
  so this is not a plain ViT→merger→prepend.
* **merger**: LayerNorm + MLP projecting `spatial_merge²·1152 → 4096`.
* **LM integration**: replace `image_token_id` placeholders in the LM input-embeds with the merged
  vision tokens, run the language tower over *embeds* (not ids) with **interleaved M-RoPE** position
  ids `[24,20,20]`, inject deepstack features, and tap the final hidden state. This requires an
  embeds-input + M-RoPE + deepstack path on `LlamaStyleEncoder` (a shared, parity-validated addition).

This is a sizeable standalone subsystem; it is specced here in full and is the remaining work for the
multimodal **edit** conditioning. The DiT-side edit path (reference image via the VAE latent stream:
`ref_image_patch_embedder` → `ref_image_refiner` → concat) is already implemented and tested, so edits
driven by self-descriptive instructions work today; the vision tower adds image-aware conditioning.

## 8. Pipeline surface

`BooguImagePipeline : DiffusionPipelineBase` exposing synchronous methods returning
`(byte[] rgbData, int width, int height, int seed)` with `Action<GenerationProgress>?` callbacks
(matching the engine convention — no `IAsyncEnumerable`):

* `GenerateFromTokens(...)` — T2I (Base/Turbo), single CFG, text-only MLLM encode.
* `EditFromTokens(...)` / TI2I — encodes reference image(s) through the VAE (DiT latent path) and the
  Qwen3-VL vision tower (conditioning path), runs the double-guidance loop.

Resolution: output H/W aligned to multiples of 16; native generation resolution bounded by
`max_input_image_pixels` / `max_input_image_side_length` (pretrain max 2K), upsampled to the final
size at the end. `seq_len` for the scheduler shift is the packed image token count
(`(H/8/2)·(W/8/2)`); the released config pins it at 4096 (≈1024²).

## 9. Open items / deferred

* Prompt rewriting / "instruction reasoner" (the released pipeline can rewrite prompts via the same
  Qwen3-VL or DashScope) — deferred; we encode the user instruction directly.
* TeaCache / TaylorSeer step-skipping caches — deferred (inference-speed optional).
* Prompt-tuning (`PromptEmbedding`) — disabled in the release (`num_layers 0`); not implemented.
* Boosted orthogonal guidance and `empty_instruction_guidance_scale` — deferred.
* fp8 checkpoints — handled by the existing fp8 scaled-dequant path in the converter.

## 10. Validation plan

Per-component CPU-vs-reference parity (tolerance ~1e-4 f32) via `tests/python-reference` dumps,
following the Ideogram4/OmniGen2 pattern:

1. RoPE tables (T2I + ref-image offsets) vs `rope.py`.
2. Single block (modulated + non-modulated) and the double-stream block vs the Python blocks.
3. Time/caption embed + output norm.
4. Scheduler v1 sigma schedule vs the Python scheduler.
5. Full transformer forward on a tiny synthetic config.
6. Qwen3-VL vision tower: `pixel_values`/`image_grid_thw` build + merged image tokens vs HF.
7. End-to-end T2I and Edit smoke generations (skip when weights/VRAM absent).

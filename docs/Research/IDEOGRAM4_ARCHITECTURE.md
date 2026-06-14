# Ideogram 4 Architecture — Research Notes

> **Status:** Complete (read-only from upstream code; no checkpoint inspected on disk yet) | **Last Updated:** 2026-06-07 | **Needed Before:** `Ideogram4Transformer`, `Ideogram4Pipeline`, `Ideogram4CheckpointConverter`, and the structured-prompt builder ([STRUCTURED_PROMPT_BUILDER.md](STRUCTURED_PROMPT_BUILDER.md))
>
> **Sources of truth:**
> - GitHub: [ideogram-oss/ideogram4](https://github.com/ideogram-oss/ideogram4) — `src/ideogram4/modeling_ideogram4.py`, `pipeline_ideogram4.py`, `scheduler.py`, `sampler_configs.py`, `latent_norm.py`, `constants.py`, `autoencoder.py`, `magic_prompt.py`, `quantized_loading.py`
> - GitHub docs: `docs/model_architecture.md`, `docs/pipeline.md`, `docs/inference.md`, `docs/prompting.md`
> - HuggingFace (official, gated): `ideogram-ai/ideogram-4-nf4`, `ideogram-ai/ideogram-4-fp8`
> - HuggingFace (ComfyUI, ungated): [`Comfy-Org/Ideogram-4`](https://huggingface.co/Comfy-Org/Ideogram-4)
> - Text encoder config: [`Qwen/Qwen3-VL-8B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) `config.json` (`text_config`)
> - ComfyUI reference (treat as **example, not gospel** — several known mistakes; see § Differences Between Implementations): [docs.comfy.org Ideogram v4](https://docs.comfy.org/tutorials/image/ideogram/ideogram-v4)
>
> **License:** "Ideogram 4 Non-Commercial" (gated on HuggingFace). The DiT weights and inference code are non-commercial. The ComfyUI repackage (`Comfy-Org/Ideogram-4`) mirrors the same weights. **This is a non-commercial license — flag for the package/legal boundary; HartsyInference itself stays MIT/permissive, the model weights carry their own terms (same handling as the existing GameCraft license-acceptance gate).**

## Summary

Ideogram 4 is a **9.3B-parameter single-stream Diffusion Transformer** (DiT) text-to-image foundation model, open-weight release of the `ideogram-oss` org. Unlike the dual-stream MMDiT designs already in this codebase (Flux, SD3.5, Lens), Ideogram 4 concatenates text tokens and image-latent tokens into **one unified sequence** and runs them through **34 identical single-stream blocks** — there are no separate text/image branches. Conditioning comes from **Qwen3-VL-8B-Instruct**: the pipeline runs the prompt through the VLM's 36-layer language model and **concatenates the hidden states of 13 layers** `(0, 3, 6, …, 33, 35)` channel-wise (4096 × 13 = **53248**), then RMSNorms and projects that down to the model width 4608. The other headline trick is **3D MRoPE** (Qwen-VL-style multimodal rotary embedding with sections `(24, 20, 20)`, θ=5e6) that puts text and image tokens in a **shared positional space**, plus a `segment_ids` block-diagonal attention mask and an `image_indicator` embedding to keep the two modalities from cross-contaminating.

The model is trained on **structured JSON captions** (scene summary + style block + per-object descriptions with bounding boxes and hex color palettes), which is why community users report prompting is hard — you get the most out of it by hand-authoring JSON or running an LLM "magic prompt" expander. That motivates the companion [STRUCTURED_PROMPT_BUILDER.md](STRUCTURED_PROMPT_BUILDER.md) design: a model-agnostic structured-prompt data model with a per-model serializer dialect (Ideogram-4 JSON first, regional-attention prompting for other models later).

VAE is the **Flux.2 semantic VAE** (`flux2-vae.safetensors`, 32-channel, 8× spatial) — **already implemented** in this codebase for Flux.2/Lens — but Ideogram does **NOT** use the Flux.2 BN running-stats un-normalize. Instead it applies its own **fixed 128-value per-channel `LATENT_SHIFT`/`LATENT_SCALE`** constants at the pipeline boundary (`z = z * scale + shift` before decode). Sampling is **flow-matching Euler** with a **logit-normal timestep schedule** whose mean auto-adjusts for resolution, and **asymmetric CFG** (the unconditional branch is the image-only token sequence with zeroed text features — shorter and cheaper than a symmetric pass) with a **two-stage guidance schedule** (gw≈7 for the bulk, gw≈3 for the last few "polish" steps).

For HartsyInference, the genuinely new pieces are: (1) **single-stream unified-sequence DiT** with **scale-only AdaLN** (no shift) + tanh-gated residuals + sandwich norms; (2) **3D MRoPE with non-equal interleaved sections** and a large image-position offset; (3) **multi-layer Qwen3-VL hidden-state capture** (13 layers — the `LlamaStyleEncoder` family already supports Qwen, needs the multi-layer tap that Lens also needs); (4) the **fixed-constant latent normalization** (replaces the BN un-normalize used by Flux.2/Lens); (5) **asymmetric CFG** with a per-step guidance schedule; (6) the **structured-JSON prompt path**. The VAE, flow-match Euler scheduler core, RMSNorm, SwiGLU, and Qwen tokenizer are all reusable.

## Detailed Findings

### Model variants & distribution

| Variant | Repo | Format | Notes |
|---|---|---|---|
| **Ideogram-4 nf4** | `ideogram-ai/ideogram-4-nf4` | NF4 (bitsandbytes) | 9.3B; CUDA-only; needs Diffusers + bnb. Not our load path (managed bnb dependency). |
| **Ideogram-4 fp8** | `ideogram-ai/ideogram-4-fp8` | FP8 | 9.3B; "all hardware". Closest to our existing FP8 load path. |
| **Comfy-Org fp8_scaled** | `Comfy-Org/Ideogram-4` `diffusion_models/ideogram4_fp8_scaled.safetensors` (~13.8 GB) | FP8 `fp8_scaled` | Per-tensor `scale_weight` companions — **we already handle this** (`Tensor.Fp8ScaleFactor`, see Flux Krea / SD3.5). Primary target. |
| **Comfy-Org nvfp4_mixed** | `diffusion_models/ideogram4_unconditional_nvfp4_mixed.safetensors` | NVFP4 mixed | Needs the NVFP4 codec (the Lens TE path added `Nvfp4Codec` — reuse). |

There is a single architecture; only quantization + the sampler-preset defaults differ. One C# `Ideogram4Config` covers all.

**⚠ The "unconditional" model file is a likely ComfyUI artifact, not a second architecture.** The official `pipeline_ideogram4.py` runs **one** transformer twice per step (positive = full text+image tokens; negative = same weights, image-only tokens with **zeroed text features**). The separate `ideogram4_unconditional_*.safetensors` in the Comfy repo is best understood as the *same weights repackaged for the negative pass* (or a Comfy-specific split), **not** a distinct unconditional network. **Verify on download** by diffing tensor hashes against the conditional file. Do not implement two separate transformers unless the hashes genuinely differ. See § Open Questions.

### Transformer config (derived — `docs/model_architecture.md` + `modeling_ideogram4.py`)

```
params              9.3 B
num_layers          34          (single-stream Ideogram4TransformerBlock)
dim (inner)         4608
num_heads           18
head_dim            256         (4608 / 18)
mlp_hidden          12288       (SwiGLU; NOT 8/3 ratio — explicit 12288)
adaln_dim           512
in_channels         128         (= 32 VAE ch × 2×2 patch)
out_channels        128         (proj_out; unpatched to 32 ch in pipeline)
rope_base (θ)       5_000_000
mrope_section       (24, 20, 20)   sums to 64; doubled → 128 freq pairs = head_dim/2
norm                RMSNorm everywhere except final_layer.norm_final (LayerNorm w/ affine)
qk_norm             RMSNorm(256) per head, on Q and K (not V)
```

**Derived:** `head_dim=256` is unusually large (most DiTs use 64/128). `mrope_section (24,20,20)` × 2 = `48+40+40 = 128 = head_dim/2`, the number of rotary frequency pairs.

### Top-level module hierarchy & state-dict keys (`modeling_ideogram4.py`)

These are the exact PyTorch attribute names → safetensors keys (diffusers-style, **no `transformer.` prefix** in the Comfy bf16 file; verify nf4/fp8 official prefix on download):

```
input_proj.{weight,bias}                       Linear(128 → 4608)      image-latent token embed
llm_cond_norm.weight                           RMSNorm(53248)          over the 13-layer Qwen concat
llm_cond_proj.{weight,bias}                    Linear(53248 → 4608)    text-feature projection
t_embedding.mlp_in.{weight,bias}               Linear(4608 → 4608)     (after sinusoidal embed)
t_embedding.mlp_out.{weight,bias}              Linear(4608 → 4608)
adaln_proj.{weight,bias}                        Linear(4608 → 512)      shared timestep→AdaLN cond (c)
embed_image_indicator.weight                   Embedding(2, 4608)      text-vs-image token tag
rotary_emb.inv_freq                            buffer (not trained)    MRoPE freqs

layers.{i}.attention.qkv.weight                Linear(4608 → 13824)    fused QKV (3 × 4608)
layers.{i}.attention.norm_q.weight             RMSNorm(256)            per-head QK norm
layers.{i}.attention.norm_k.weight             RMSNorm(256)
layers.{i}.attention.o.weight                  Linear(4608 → 4608)     out proj (no bias)
layers.{i}.feed_forward.w1.weight              Linear(4608 → 12288)    SwiGLU gate
layers.{i}.feed_forward.w3.weight              Linear(4608 → 12288)    SwiGLU up
layers.{i}.feed_forward.w2.weight              Linear(12288 → 4608)    SwiGLU down
layers.{i}.attention_norm1.weight              RMSNorm(4608)           pre-attn
layers.{i}.attention_norm2.weight              RMSNorm(4608)           post-attn (sandwich)
layers.{i}.ffn_norm1.weight                    RMSNorm(4608)           pre-mlp
layers.{i}.ffn_norm2.weight                    RMSNorm(4608)           post-mlp (sandwich)
layers.{i}.adaln_modulation.{weight,bias}      Linear(512 → 18432)     per-block, 4×4608 chunks

final_layer.norm_final.{weight,bias}           LayerNorm(4608)         affine (the ONLY LayerNorm)
final_layer.adaln_modulation.{weight,bias}     Linear(512 → 4608)      scale-only (no shift)
final_layer.linear.{weight,bias}               Linear(4608 → 128)      proj_out
```

### Block forward pass (`Ideogram4TransformerBlock.forward`)

The distinctive parts vs Flux/SD3/Lens: **scale-only modulation (no shift terms)**, **tanh-gated residuals**, and **sandwich norms** (a norm both before AND after each sublayer).

```python
# Per-block AdaLN from the shared 512-d conditioning vector c
mod = adaln_modulation(c)                       # (B, 1, 18432)
scale_msa, gate_msa, scale_mlp, gate_mlp = mod.chunk(4, dim=-1)   # 4 × (B,1,4608)
gate_msa = tanh(gate_msa);  gate_mlp = tanh(gate_mlp)
scale_msa = 1.0 + scale_msa;  scale_mlp = 1.0 + scale_mlp

# Attention sublayer (NOTE: scale only, no shift; norm2 wraps the attn output)
h = attention_norm1(x) * scale_msa
qkv = qkv(h).view(B, L, 3, 18, 256)             # → q, k, v
q = norm_q(q);  k = norm_k(k)                    # per-head RMSNorm
q, k = apply_rotary_pos_emb(q, k, cos, sin)      # 3D MRoPE
attn = scaled_dot_product_attention(q, k, v, attn_mask=block_diag(segment_ids))
attn = o(attn)
x = x + gate_msa * attention_norm2(attn)

# MLP sublayer (SwiGLU, sandwich norm, scale-only)
h = ffn_norm1(x) * scale_mlp
mlp = w2(silu(w1(h)) * w3(h))
x = x + gate_mlp * ffn_norm2(mlp)
return x
```

**Watch in the C# port:**
1. **No shift in any modulation** — `_modulate` is `x * (1 + scale)`, not `x * (1 + scale) + shift`. Every other DiT in this codebase has a shift term; do not copy that pattern.
2. **Chunk order is `(scale_msa, gate_msa, scale_mlp, gate_mlp)`** — only 4 chunks, not 6.
3. **Sandwich norm:** `attention_norm2` / `ffn_norm2` normalize the *sublayer output* before gating, in addition to `attention_norm1` / `ffn_norm1` on the input. This is the Gemma/OLMo-style sandwich pattern; the existing `LlamaStyleEncoderConfig.HasFfnSandwichNorms` flag is the conceptual cousin but the DiT block is bespoke.
4. **`tanh` on both gates** — gates are squashed to (−1, 1) before scaling the residual.
5. **Fused QKV** `Linear(4608 → 13824)` splits 3-ways into `(q, k, v)`, each `(B, L, 18, 256)`. Reuse the existing fused-QKV split helper.

### Timestep embedding (`Ideogram4EmbedScalar`)

```python
scaled = 1e4 * t                          # t ∈ [0, 1]
emb = sinusoidal_embedding(scaled, 4608)  # sin/cos pairs
emb = silu(mlp_in(emb))
temb = mlp_out(emb)                        # (B, 1, 4608)
c = adaln_proj(temb)                       # (B, 1, 512) — the shared AdaLN conditioning
```

`c` (512-d) feeds **every** block's `adaln_modulation` and the `final_layer.adaln_modulation`. This is the "AdaLN dimension = 512" in the arch doc — a width-512 bottleneck shared across all 34 blocks (parameter-efficient vs Flux's full-width per-block modulation).

### Final layer (`Ideogram4FinalLayer`)

```python
scale = 1.0 + adaln_modulation(c)        # (B, 1, 4608) — scale only, no shift
out = linear(norm_final(x) * scale)       # LayerNorm(4608, affine) → Linear(4608 → 128)
```

`norm_final` is the **only LayerNorm** in the model (it has both weight and bias); everything else is RMSNorm.

### Text encoding & multi-layer capture (`pipeline_ideogram4.py`)

```python
# 1. chat template (Qwen3 tokenizer)
messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

# 2. forward through Qwen3-VL-8B-Instruct language model (36 layers), tap 13
QWEN3_VL_ACTIVATION_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)
captured = {}
for layer_idx, decoder_layer in enumerate(language_model.layers):
    hidden_states = decoder_layer(...)
    if layer_idx in tap_set:
        captured[layer_idx] = hidden_states     # each (B, L, 4096)

# 3. stack + channel-concat
stacked = torch.stack(selected, dim=0)          # (13, B, L, 4096)
stacked = stacked.permute(1, 2, 3, 0).reshape(B, L, 4096 * 13)   # (B, L, 53248)
# non-LLM (padding / image) positions are zeroed

# 4. inside the DiT:  llm_features = llm_cond_proj(llm_cond_norm(stacked))   # (B, L, 4608)
```

**Qwen3-VL-8B-Instruct language config (verified from `config.json` `text_config`):**

| Field | Value |
|---|---|
| hidden_size | 4096 |
| num_hidden_layers | 36 (indices 0..35; layer 35 = last) |
| num_attention_heads | 32 |
| head_dim | 128 |
| intermediate_size | 12288 |
| rms_norm_eps | 1e-6 |
| vocab_size | 151936 |
| rope_theta | 5_000_000 |

Only the **language tower** is needed (text-only prompts → no vision encoder). The 13-layer concat is `4096 × 13 = 53248` exactly, matching `llm_cond_proj` input. Max text tokens = **2048**.

### Unified-sequence layout, MRoPE & masking

The sequence is `[text tokens (≤2048)] + [image latent tokens (grid_h × grid_w)]` concatenated and run through the same 34 blocks. Three constructs keep modalities separated:

1. **`embed_image_indicator: Embedding(2, 4608)`** — image tokens get index-1 embedding added, text tokens index-0 (tag values come from `OUTPUT_IMAGE_INDICATOR=2` / `LLM_TOKEN_INDICATOR=3` mapped to 0/1; verify the exact mapping on first run).
2. **`segment_ids` block-diagonal attention mask** — built per sample so padded text positions (`SEQUENCE_PADDING_INDICATOR = -1`) are masked out; image tokens attend to valid text + all image, text attends within itself.
3. **3D MRoPE** — `position_ids` shape `(B, L, 3)` = `(temporal, height, width)`. Image grid positions are offset by `IMAGE_POSITION_OFFSET = 65536` so they never collide with text token indices in the shared positional space.

**MRoPE construction (`Ideogram4MRoPE`):**

```python
head_dim = 256;  base = 5_000_000;  mrope_section = (24, 20, 20)
inv_freq = 1.0 / (base ** (arange(0, 256, 2) / 256))   # 128 values
pos = position_ids.permute(2, 0, 1).float()             # (3, B, L)  [t, h, w]
freqs = einsum(inv_freq, pos)                           # (3, B, L, 128)

# interleave the three axes' freqs into a single 128-vector per token:
freqs_t[..., 0::3] = freqs[0][..., 0::3]                # temporal at idx%3==0
freqs_t[..., 1::3] = freqs[1][..., 1:72:3]              # height   (24×3=72)
freqs_t[..., 2::3] = freqs[2][..., 2:62:3]              # width    (20×3=60)
emb = cat([freqs_t, freqs_t], dim=-1)                   # (B, L, 256)
return cos(emb), sin(emb)
```

This is **non-interleaved-pair, sectioned MRoPE** — different from both Flux's pair-rotation and Lens/Qwen-Image complex-polar RoPE. The `apply_rotary_pos_emb(q, k, cos, sin)` is the standard `q*cos + rotate_half(q)*sin` form. **The exact interleave slicing (`1:72:3`, `2:62:3`) is fiddly — port it verbatim and validate against a Python dump.**

### Latent normalization (`latent_norm.py`) — **differs from Flux.2/Lens**

```python
LATENT_SHIFT = (... 128 floats, range ≈ −0.35 .. +0.38 ...)
LATENT_SCALE = (... 128 floats, range ≈ +1.53 .. +1.94 ...)
assert shift.shape == (128,) and scale.shape == (128,)
# applied at decode:  z = z * latent_scale + latent_shift   (then unpatchify → VAE)
```

These are **fixed hard-coded per-channel constants**, NOT the Flux.2 VAE `bn.running_mean/var`. The Flux.2/Lens decode path in this codebase applies a BatchNorm un-normalize; **Ideogram replaces that with these 128 constants applied to the 128-channel packed latent before the 2×2 unpatchify**. The 128 values must be copied verbatim from `latent_norm.py` into a C# constant table (or shipped as a small resource). See § Open Questions on whether the Flux.2 VAE's own BN is bypassed entirely.

### Scheduler (`scheduler.py`) — logit-normal flow matching

```python
class LogitNormalSchedule:
    def __call__(self, t):                       # t ∈ [0,1] uniform grid
        z   = ndtri(t)                           # inverse normal CDF
        y   = mean + std * z
        t_  = 1 - expit(y)                       # 1 − sigmoid(y)
        return t_.clamp(t_min, t_max)
    # t_min = 1/(1 + exp(0.5 * logsnr_max)),  logsnr_max = 18.0
    # t_max = 1/(1 + exp(0.5 * logsnr_min)),  logsnr_min = −15.0

def get_schedule_for_resolution((H, W), known_mean=mu, std=std):
    mean = known_mean + 0.5 * log(num_pixels / known_pixels)   # known = (512, 512)
    return LogitNormalSchedule(mean, std, ...)

step_intervals = linspace(0.0, 1.0, num_steps + 1)             # uniform grid → schedule()
```

The denoise loop is a plain Euler integrator over the schedule-warped timesteps:

```python
for i in range(num_steps - 1, -1, -1):
    t = schedule(step_intervals[i + 1]); s = schedule(step_intervals[i])
    pos_v = dit(pos_features, z, t)
    neg_v = dit(neg_features, z, t)                  # asymmetric: zeroed text, image-only seq
    v = gw[i] * pos_v + (1 - gw[i]) * neg_v          # == neg + gw*(pos − neg)
    z = z + v * (s - t)                              # delta = s − t  (negative, t→0)
```

This is **not** the diffusers `FlowMatchEulerDiscreteScheduler.set_timesteps(sigmas, mu)` `exp(mu)/(exp(mu)+(1/σ−1))` time-shift used by Flux/SD3.5/Lens. It is a **logit-normal warp of a uniform grid**. The existing `FlowMatchEulerDiscreteScheduler` cannot be reused directly — Ideogram needs a **new `LogitNormalSchedule` helper** (small, ~30 lines: `ndtri` via `erfinv`, `expit`, the resolution-mean adjust, the clamp). The Euler step itself (`z += v·Δ`) is standard.

### Sampler presets (`sampler_configs.py`)

| Preset | Steps | Guidance (main / polish) | mu (base mean) | std |
|---|---|---|---|---|
| **V4_QUALITY_48** | 48 | gw=7 for first 45, **gw=3 for last 3** | 0.0 | 1.5 |
| **V4_DEFAULT_20** | 20 | gw=7 for first 18, **gw=3 for last 2** | 0.0 | 1.75 |
| **V4_TURBO_12** | 12 | gw=7 for first 11, **gw=3 for last 1** | 0.5 | 1.75 |

`guidance_schedule` is stored **reversed (last-step-to-first)** in the dataclass — index carefully. `mu` is the **base mean before the resolution adjust**; `std` is the logit-normal spread. The "polish" steps drop guidance near t→0 to reduce over-saturation/artifacting (an Ideogram-specific scheme — note it for the C# loop).

### Asymmetric CFG (`pipeline_ideogram4.py`)

```python
# positive: full sequence [text(≤2048) + image] with real llm_features
# negative: image-only sequence with ZEROED text features (no text prefix → shorter, cheaper)
neg_llm_features = torch.zeros(B, num_image_tokens, llm_features.shape[-1], ...)
# neg_position_ids / neg_segment_ids / neg_indicator come from the image-only slice:
#   inputs["segment_ids"][:, max_text_tokens:]
v = gw * pos_v + (1 - gw) * neg_v
```

The two passes have **different sequence lengths** (positive includes the text prefix; negative is image-only). This is the "asymmetric" CFG — it saves compute vs a symmetric uncond pass and is why the unconditional branch needs no negative prompt. **For the C# port, this means two `Forward` calls per step with different seq lengths — do not assume a batch-of-2 of identical shape** (unlike Lens/SD3.5 which duplicate one tensor). Run them as two separate forwards or pad-and-mask.

### VAE (`autoencoder.py`)

`flux2-vae.safetensors` (336 MB, FP16/BF16) — the **Flux.2 semantic VAE already loaded by `Flux2Pipeline.cs`**. 32 latent channels, 8× spatial downsample. The pipeline-side 2×2 patch gives the transformer's 128-channel I/O. **Decode unpatchify:**

```python
z = z * latent_scale + latent_shift             # 128-constant un-normalize (NOT BN)
ae_ch = z.shape[-1] // (patch * patch)          # 128 / 4 = 32
z = z.view(B, grid_h, grid_w, patch, patch, ae_ch)
z = z.permute(0, 5, 1, 3, 2, 4).contiguous()
z = z.view(B, ae_ch, grid_h * patch, grid_w * patch)   # (B, 32, H/8, W/8)
rgb = autoencoder.decoder(z)                    # → [-1, 1]
rgb = ((rgb + 1.0) * 127.5).round().uint8
```

**Reuse `VaeDecoder` (Flux.2 preset) + `Flux2CheckpointConverter`**; only the un-normalize step changes (constants instead of BN).

### Resolution & quantization

- **Resolution:** any dimension from **256 to 2048, multiples of 16**, aspect ratios up to **6:1 / 1:6**. The logit-normal mean auto-adjusts via `0.5·log(num_pixels/512²)`. No fixed bucket table (unlike Lens) — fully flexible.
- **Quantization:** official `nf4` (CUDA + bitsandbytes — **skip**, managed bnb dependency) and `fp8` (all hardware — **our target**). Comfy adds `fp8_scaled` (per-tensor scale companions, **already supported**) and `nvfp4_mixed` (reuse the `Nvfp4Codec` added for the Lens text encoder).

## Key Numbers / Constants

| Constant | Value | Source |
|---|---|---|
| Parameters (DiT) | 9.3 B | README |
| Layers | 34 | `modeling_ideogram4.py` |
| Hidden (inner) dim | 4608 | model_architecture.md |
| Attention heads | 18 | " |
| Head dim | 256 | 4608/18 |
| MLP hidden (SwiGLU) | 12288 | " |
| AdaLN dim | 512 | `adaln_proj` |
| In/out channels (transformer) | 128 / 128 | `input_proj` / `final_layer.linear` |
| VAE latent channels | 32 | Flux.2 VAE |
| Patch (pipeline) | 2×2 | 32 × 4 = 128 |
| VAE spatial downsample | 8× | autoencoder |
| RoPE θ (base) | 5_000_000 | `Ideogram4MRoPE` |
| MRoPE sections | (24, 20, 20) | " |
| Text encoder | Qwen3-VL-8B-Instruct (lang tower) | pipeline |
| TE hidden | 4096 | Qwen config |
| TE layers (total / used) | 36 / 13 | Qwen config / `QWEN3_VL_ACTIVATION_LAYERS` |
| TE tapped layers | (0,3,6,9,12,15,18,21,24,27,30,33,35) | `constants.py` |
| Text feature concat dim | 53248 | 4096 × 13 |
| Max text tokens | 2048 | pipeline |
| Per-block modulation outputs | 4 × 4608 = 18432 | `adaln_modulation` |
| Final modulation | 4608 (scale only) | `final_layer.adaln_modulation` |
| Latent norm channels | 128 | `latent_norm.py` |
| LATENT_SHIFT range | ≈ −0.35 .. +0.38 | `latent_norm.py` (copy verbatim) |
| LATENT_SCALE range | ≈ +1.53 .. +1.94 | `latent_norm.py` (copy verbatim) |
| Scheduler logsnr_max / min | 18.0 / −15.0 | `scheduler.py` |
| Schedule known resolution | (512, 512) | `scheduler.py` |
| Default preset | V4_QUALITY_48 (48 steps) | `inference.md` |
| Guidance main / polish | 7.0 / 3.0 | `sampler_configs.py` |
| SEQUENCE_PADDING_INDICATOR | −1 | `constants.py` |
| OUTPUT_IMAGE_INDICATOR | 2 | `constants.py` |
| LLM_TOKEN_INDICATOR | 3 | `constants.py` |
| IMAGE_POSITION_OFFSET | 65536 | `constants.py` |
| Resolution range | 256–2048, ×16 | README |
| Aspect ratio max | 6:1 | README |

## Data Layouts / Formats

### Tensor shapes through the pipeline (1024×1024 example)

```
prompt → chat template → tokenize → input_ids                 [B, L_text ≤ 2048]
↓ Qwen3-VL-8B language tower, tap 13 layers
13 × [B, L_text, 4096]  → concat                              [B, L_text, 53248]
↓ (inside DiT) llm_cond_norm + llm_cond_proj
text features                                                 [B, L_text, 4608]

noise (pure Gaussian at t=1)                                  [B, grid_h·grid_w, 128]   grid = 64×64
↓ input_proj
image tokens                                                 [B, 4096, 4608]
+ embed_image_indicator                                       (image tag added)

unified sequence  [text tokens | image tokens]               [B, L_text + 4096, 4608]
+ 34× single-stream blocks (3D MRoPE, block-diag mask)
↓ final_layer (slice image tokens only)
velocity                                                      [B, 4096, 128]

z₀ (after Euler loop)                                         [B, 4096, 128]
↓ z·latent_scale + latent_shift  → reshape/unpatchify
                                                             [B, 32, 128, 128]   (H/8 × W/8)
↓ Flux.2 VAE decode
RGB                                                          [B, 3, 1024, 1024]  [-1,1]
```

### ComfyUI checkpoint layout (`Comfy-Org/Ideogram-4`, total ~46.8 GB)

```
diffusion_models/ideogram4_fp8_scaled.safetensors              ~13.8 GB  FP8 scaled (per-tensor scale_weight)
diffusion_models/ideogram4_unconditional_*.safetensors         (verify: same weights as conditional? — see Open Q)
diffusion_models/ideogram4_*_nvfp4_mixed.safetensors           NVFP4 (reuse Nvfp4Codec)
text_encoders/qwen3vl_8b_fp8_scaled.safetensors                ~8 GB    Qwen3-VL-8B language tower
text_encoders/qwen3vl_8b_nvfp4.safetensors                     NVFP4 variant
text_encoders/gemma4_e4b_it_fp8_scaled.safetensors             ~2 GB    ⚠ MAGIC-PROMPT LLM, NOT a conditioning encoder
vae/flux2-vae.safetensors                                      ~336 MB  Flux.2 VAE (reused)
```

## Algorithm Steps (pseudocode for the C# port)

```
Ideogram4Pipeline.GenerateFromTokens(promptIds, height, width, presetName, seed):
    (gridH, gridW) = (height / 8 / 2, width / 8 / 2)        // VAE 8× + 2×2 patch
    numImgTokens   = gridH * gridW

    // 1. Qwen3-VL multi-layer encode (text-only); concat 13 layers; project happens inside DiT
    layers13 = qwen.EncodeTapLayers(promptIds, QWEN3_VL_ACTIVATION_LAYERS)   // List[13] of [B, Lt, 4096]
    txtConcat = ConcatChannel(layers13)                                       // [B, Lt, 53248]

    // 2. Build positions / segment_ids / indicators for the UNIFIED sequence
    posTxt, segTxt, indTxt = BuildTextMeta(Lt)
    posImg, segImg, indImg = BuildImageMeta(gridH, gridW, IMAGE_POSITION_OFFSET)

    // 3. Asymmetric CFG inputs
    posFeatures = txtConcat                                                   // text+image pass
    negFeatures = Zeros(B, numImgTokens, 53248)                              // image-only, zeroed text

    // 4. Noise + schedule
    z = DeterministicRng.Gaussian(seed, [B, numImgTokens, 128])
    preset = SamplerPresets[presetName]                                       // steps, gw[], mu, std
    schedule = LogitNormalSchedule(mu + 0.5*log(H*W / 512²), preset.std)
    grid = LinSpace(0, 1, preset.steps + 1)

    // 5. Euler denoise with two-stage guidance
    for i in (steps-1 .. 0):
        t = schedule(grid[i+1]); s = schedule(grid[i])
        posV = transformer.Forward(z, posFeatures, posMeta, t)               // full seq
        negV = transformer.Forward(z, negFeatures, imgOnlyMeta, t)           // image-only seq
        v = preset.gw[i] * posV + (1 - preset.gw[i]) * negV
        z = z + v * (s - t)

    // 6. Free transformer + encoder before VAE (PHASE_3_DEVIATIONS #18 pattern)
    backend.Sync(); backend.FreeWeights(transformer.EnumerateWeights()); backend.FreeWeights(qwen.EnumerateWeights())

    // 7. Latent un-normalize (CONSTANTS, not BN) + unpatchify + decode
    z = z * LATENT_SCALE + LATENT_SHIFT
    latent2D = Unpatchify(z, gridH, gridW, patch=2, aeChannels=32)            // [B, 32, H/8, W/8]
    rgb = flux2Vae.Decode(latent2D)                                          // [-1,1]
    return ClampToUint8(rgb)
```

## Reference Implementations

- **`ideogram-oss/ideogram4 — modeling_ideogram4.py`** — `Ideogram4Transformer`, `Ideogram4TransformerBlock`, `Ideogram4Attention`, `Ideogram4MLP`, `Ideogram4MRoPE`, `Ideogram4EmbedScalar`, `Ideogram4FinalLayer`. **Primary reference for the C# transformer.**
- **`pipeline_ideogram4.py`** — `Ideogram4Pipeline`: Qwen tap-and-concat, asymmetric CFG, Euler loop, latent norm, unpatchify. **Primary reference for the C# pipeline.**
- **`scheduler.py`** — `LogitNormalSchedule`, `get_schedule_for_resolution`, `make_step_intervals`. **Primary reference for the new scheduler helper.**
- **`sampler_configs.py`** — the three presets + guidance schedules.
- **`latent_norm.py`** — the 128 `LATENT_SHIFT` / `LATENT_SCALE` constants (copy verbatim).
- **`constants.py`** — indicator/offset constants + the 13 tap layers.
- **`autoencoder.py`** + `flux2-vae.safetensors` — Flux.2 VAE; already implemented (`Flux2CheckpointConverter`, `VaeDecoder`).
- **`magic_prompt.py` + `magic_prompt_system_prompts/`** — the LLM JSON-expander system prompts. Relevant to [STRUCTURED_PROMPT_BUILDER.md](STRUCTURED_PROMPT_BUILDER.md), not the inference core.
- **`Qwen/Qwen3-VL-8B-Instruct` `config.json`** — language-tower dims. The `LlamaStyleEncoder` family already supports Qwen3.
- **diffusers `AutoencoderKLFlux2`** — already in tree via Flux.2.

## Differences Between Implementations

The reference is single-source (`ideogram-oss/ideogram4`). The divergences worth pinning are vs **ComfyUI** (which the user flagged as having mistakes) and vs **other DiTs in this codebase**:

1. **Gemma-4 is NOT a conditioning text encoder.** ComfyUI bundles `gemma4_e4b_it_fp8_scaled.safetensors` (~2 GB) as a **local LLM to generate the structured-JSON prompt** (its "magic prompt"). The actual model conditioning comes from **Qwen3-VL only**. Treating Gemma-4 as a second text encoder would be wrong. The official repo uses an API/Claude/Ideogram-hosted LLM for the same job. **In HartsyInference, the JSON builder is a separate utility (the prompt builder), with an optional pluggable LLM expander — not a model component.**
2. **The "unconditional" checkpoint is (almost certainly) not a separate network.** Official CFG runs one transformer twice. See § Open Questions — verify by hashing.
3. **Asymmetric CFG, not symmetric.** Unlike Lens/SD3.5 (duplicate one tensor batch-of-2, identical shapes), Ideogram's negative pass is a *shorter, image-only* sequence with zeroed text. Two forwards of different length.
4. **Scale-only AdaLN (no shift), tanh gates, sandwich norms.** Every other DiT here has shift terms and a single pre-norm. Do not copy `FluxDoubleStreamBlock` / `JointBlock` modulation wholesale.
5. **Logit-normal schedule, not the SD3 time-shift.** The existing `FlowMatchEulerDiscreteScheduler` is the wrong reuse target; write a small `LogitNormalSchedule`.
6. **Fixed-constant latent norm, not BN.** Flux.2/Lens decode applies VAE `bn.running_mean/var`; Ideogram applies its own 128 constants. Confirm the Flux.2 VAE's internal BN is bypassed (see Open Q).
7. **3D MRoPE with non-equal sections `(24,20,20)` and a 65536 image offset** — a third RoPE flavor distinct from Flux pair-rotation and Lens/Qwen complex-polar. The interleave slicing is bespoke.
8. **head_dim = 256** is large; the existing SDPA kernels must handle 256-wide heads (verify the `sdpa_f32.ptx` shared-memory tiling at 256, or fall back to the tiled path).

## Open Questions

- **~~Is `ideogram4_unconditional_*.safetensors` distinct weights?~~ RESOLVED — yes, distinct.** Reading the verbatim upstream `pipeline_ideogram4.py` settled it: the pipeline loads `transformer/` and `unconditional_transformer/` as **separate state dicts** and builds two `Ideogram4Transformer` instances, running both per step. The C# port implements two transformers accordingly. (Confirm tensor hashes differ on download as a sanity check, but the upstream code is unambiguous.)
- **Does the Flux.2 VAE BatchNorm get bypassed entirely?** Ideogram applies its own `LATENT_SHIFT/SCALE`. Need to confirm the `flux2-vae.safetensors` Ideogram ships either has identity BN or that `autoencoder.decoder()` doesn't re-apply BN un-norm. Inspect `autoencoder.py`.
- **Exact `embed_image_indicator` index mapping.** `Embedding(2, 4608)` with constants `OUTPUT_IMAGE_INDICATOR=2` / `LLM_TOKEN_INDICATOR=3` — what maps to embedding row 0 vs 1? Read the indicator-to-index code.
- **QK-norm and stream-norm eps values.** Likely 1e-6 (matches Qwen) but confirm in `Ideogram4RMSNorm` default.
- **`segment_ids` block-diagonal mask exact rule** for image↔text attention (full cross-attend or restricted?). Read the mask builder.
- **`t_embedding` sinusoidal dim & frequency base** — is it the standard `1e4`-base sinusoidal at 4608, or half-dim sin/cos? Verify against the dump.
- **Magic-prompt system prompts** — `magic_prompt_system_prompts/` contents define the JSON the model expects; useful for the prompt-builder defaults but not load-bearing for inference.
- **NF4 path** — we will NOT support bitsandbytes NF4 (managed dep). Confirm the fp8 / fp8_scaled / nvfp4 paths fully cover the weights (they should).

## Implementation Notes (recommendations for HartsyInference)

### What can be reused

- **`VaeDecoder` (Flux.2 preset) + `Flux2CheckpointConverter`** — the VAE is identical; only swap BN un-norm for the constant latent-norm.
- **`LlamaStyleEncoder` (Qwen3 preset)** — already supports Qwen3. Needs the **multi-layer hidden-state tap** (13 layers) — the **same net-new capability Lens needs** (Lens taps GPT-OSS layers [5,11,17,23]). Build one shared `EncodeTapLayers(int[] layerIndices)` capability and both models use it. **Do NOT duplicate.** (See [MICROSOFT_LENS_ARCHITECTURE.md](MICROSOFT_LENS_ARCHITECTURE.md) § multi-layer capture and the AGENTS.md reuse rule.)
- **Fused-QKV split helper** — existing (Flux/SD3 converters split `qkv`).
- **SwiGLU `w1/w2/w3`** — existing `SwiGluFfn` (bias=False). Note Ideogram's `w2(silu(w1)·w3)` matches.
- **RMSNorm (learned scale)** — existing primitive.
- **Euler step** (`z += v·Δ`) — trivial; the scheduler warp is the only new math.
- **FP8 `fp8_scaled` scale-companion folding** — existing (`Tensor.Fp8ScaleFactor`).
- **`Nvfp4Codec`** — added for Lens; reuse for the nvfp4 variants.
- **`DeterministicRng`** — seeded Gaussian noise.
- **`Qwen3Tokenizer` / `Microsoft.ML.Tokenizers.BpeTokenizer`** — Qwen3 chat template already in tree.

### What's net-new

| Component | Effort | Why |
|---|---|---|
| **`Ideogram4Config.cs`** | Low | One preset; all dims above. |
| **`Ideogram4Transformer.cs` + `Ideogram4Block.cs`** | Medium (~4-6 days) | Single-stream unified-sequence; scale-only tanh-gated AdaLN + sandwich norms; new block, not a Flux clone. |
| **`Ideogram4Mrope.cs`** | Medium (~2-3 days) | 3D sectioned MRoPE `(24,20,20)`, θ=5e6, 65536 offset, bespoke interleave. Validate vs Python dump. |
| **Multi-layer Qwen tap** | Medium (~2-3 days) | Shared with Lens. Add `EncodeTapLayers` to the encoder family. |
| **`LogitNormalSchedule.cs`** | Low (~1 day) | `ndtri`/`erfinv` + `expit` + resolution-mean adjust + clamp. New scheduler. |
| **`Ideogram4Pipeline.cs`** | Medium (~3 days) | Asymmetric CFG (two-length forwards), two-stage guidance, constant latent-norm, Flux.2 decode. |
| **`Ideogram4CheckpointConverter.cs`** | Low (~1-2 days) | Diffusers-naming passthrough + fused-QKV split + fp8_scaled / nvfp4 folding; route VAE to Flux.2 converter. |
| **Latent-norm constants** | Low | Copy 128 `LATENT_SHIFT`/`LATENT_SCALE` floats verbatim. |
| **`Ideogram4DebugDump.cs` + `dump_ideogram4_full_forward.py` + `diff_ideogram4_layers.py` + `Ideogram4DiffTests.cs`** | Medium (~2-3 days) | Layer-by-layer diff harness (SD3.5/Lens template). MRoPE + asymmetric CFG are the likely first-run bug hotspots. |
| **`Ideogram4GenerationTests.cs`** | Low (~1 day) | End-to-end scaffold; VRAM probe (~14 GB fp8 DiT + ~8 GB Qwen + ~5 GB VAE — needs eviction discipline on 12 GB). |
| **Structured-prompt builder** | — | Separate workstream, see [STRUCTURED_PROMPT_BUILDER.md](STRUCTURED_PROMPT_BUILDER.md). |

### VRAM budget (12 GB target, RTX 3060)

- DiT fp8 (9.3 B) ≈ **9.5 GB** → tight. fp8_scaled is the realistic path; activations at 1024² (≤2048 text + 4096 image tokens × 4608 dim) are non-trivial.
- Qwen3-VL-8B language tower fp8 ≈ **8 GB** (encode once, then `FreeWeights` before the DiT).
- Flux.2 VAE ≈ **~5 GB** (load after `FreeWeights(transformer)`).
- **Eviction plan (same as Lens/SD3.5/Qwen-Image):** encode → free Qwen → run DiT → free DiT → VAE decode. 12 GB is feasible at fp8 with discipline; 8 GB likely needs nvfp4 + tiled VAE decode.

### Suggested implementation order

1. **Multi-layer Qwen tap** (shared with Lens) — unblocks both.
2. **`Ideogram4Config` + `Ideogram4Mrope` + `Ideogram4Block` + `Ideogram4Transformer`** — port block-by-block; validate MRoPE against a dump early.
3. **`LogitNormalSchedule`** + **latent-norm constants**.
4. **`Ideogram4CheckpointConverter`** (diffusers passthrough + fp8_scaled/nvfp4).
5. **`Ideogram4Pipeline`** (asymmetric CFG, two-stage guidance).
6. **Validation harness** — expect 1-3 first-run bug iterations (MRoPE interleave, asymmetric-CFG seq lengths, latent-norm vs BN are the suspects).
7. **SwarmUI extension wiring** — register the new arch loader + Qwen side-model in the SwarmUI-HartsyInference extension (see Phase 4 checklist + the [[swarmui_extension]] memory).

### What NOT to do

- **Don't treat Gemma-4 as a text encoder.** It's the optional JSON-prompt LLM. Conditioning = Qwen3-VL only.
- **Don't reuse `FlowMatchEulerDiscreteScheduler`** — Ideogram's schedule is logit-normal, not SD3 time-shift.
- **Don't reuse the Flux.2/Lens BN latent un-normalize** — Ideogram uses fixed constants.
- **Don't assume symmetric batch-of-2 CFG** — the negative pass is a shorter image-only sequence.
- **Don't add a shift term to the AdaLN modulation** — Ideogram modulation is scale-only.
- **Don't implement two transformers for the "unconditional" file** until you've confirmed the weights actually differ.
- **Don't pull in bitsandbytes/NF4** — fp8 / fp8_scaled / nvfp4 cover the weights with our existing codecs.
</content>
</invoke>

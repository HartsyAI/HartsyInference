# Z-Image Architecture

> **Status:** Research / spec for implementation. Apache 2.0. By Tongyi Lab (Alibaba).
> **Related:** [FLUX_ARCHITECTURE.md](FLUX_ARCHITECTURE.md), [TEXT_ENCODERS.md](TEXT_ENCODERS.md), [DIFFUSION_SCHEDULERS.md](DIFFUSION_SCHEDULERS.md), [VAE_ARCHITECTURE.md](VAE_ARCHITECTURE.md).

## Important correction up front

Z-Image is **not** a Flux variant. It is a **Lumina2 / NextDiT** architecture (Alibaba's own DiT lineage). The earlier "Flux-lineage S3-DiT" framing in `MODEL_SUPPORT_ROADMAP.md` was wrong. Some sub-components overlap with Flux's single-stream block (RMSNorm + QK-norm MHA + SwiGLU FFN + AdaLN), but the modulation count, RoPE setup, sequence assembly, and refiner blocks are different enough that **`FluxTransformer` cannot be reused as-is**. Plan for a new `ZImageTransformer` class that may share *sub-component* code with Flux (`SwiGluFfn`, `QkNorm`, `AdaLNModulation`) but not the top-level forward.

## Variants

| Variant | Params | Steps | Status |
|---|---|---|---|
| **Z-Image-Turbo** | 6B | 8 NFE, CFG=1.0 | Released. SchedulerShift=3.0. Apache 2.0. |
| **Z-Image-Base** | 6B *(same dim as Turbo)* | 28–50 steps, CFG 3.0–5.0 | **Released 2026-01-28** at `Tongyi-MAI/Z-Image`. SchedulerShift=6.0. Apache 2.0. |
| **Z-Image-Omni-Base** | 6B | — | diffusers code merged 2025-12-24 (PR #12857); HF weights repo not yet visible. |
| **Z-Image-Edit** | — | — | "to be released" on the official GitHub README. |

There is **no 20B Base** — earlier roadmap claim was wrong. Base shares the Turbo dim/depth (3840 hidden, 30 layers); it's the un-distilled foundation, not a larger model. Architecture is **byte-identical to Turbo** in every transformer config field — only the weights themselves and the sampling regime differ.

## Transformer config (Turbo and Base share this)

Source: `Tongyi-MAI/Z-Image-Turbo/transformer/config.json`.

| Field | Value | Notes |
|---|---|---|
| Class | `ZImageTransformer2DModel` | Single-stream (no separate double-stream blocks like Flux) |
| `dim` | 3840 | Hidden size |
| `n_layers` | 30 | Main DiT blocks |
| `n_heads` | 30 | |
| `n_kv_heads` | 30 | No GQA in DiT — full MHA |
| head_dim | 128 | = dim / n_heads |
| `n_refiner_layers` | 2 | Each of `noise_refiner` and `context_refiner` |
| `in_channels` | 16 | Latent channels (Flux VAE) |
| `all_patch_size` | [2] | 2×2 spatial patchify |
| `all_f_patch_size` | [1] | Frame patch size — 1 for image-only |
| `qk_norm` | true | RMSNorm on Q and K |
| `norm_eps` | 1e-5 | |
| `rope_theta` | **256.0** | Much smaller than Flux's 10000 |
| `axes_dims` | **[32, 48, 48]** | RoPE split across (frame_idx, H, W); sums to head_dim=128 |
| `axes_lens` | [1536, 512, 512] | Max axis lengths for precomputed freqs |
| `cap_feat_dim` | 2560 | Matches Qwen3-4B hidden_size — fed in directly |
| `t_scale` | 1000.0 | Sinusoidal timestep scale |
| Guidance embed | **None** | Turbo is fully distilled (`guidance_scale=0.0`); Base uses ordinary CFG, not embedded guidance |
| FFN | **SwiGLU** | `w2(silu(w1(x)) * w3(x))` — Lumina/LLaMA naming |
| Norm | **RMSNorm everywhere** | No LayerNorm; `attention_norm1/2`, `ffn_norm1/2` |
| Modulation | **AdaLN, 4 outputs** | `[scale_msa, gate_msa, scale_mlp, gate_mlp]` per token. Embed dim **256**. Linear maps `min(dim, 256)=256 → 4*dim = 15360`. |
| Sequence layout | "Single-stream" | `[refined_caption_tokens, refined_image_tokens]` concatenated, jointly processed by all 30 main layers. Final layer applies per-token modulation (only image tokens get unpatchified). |
| Padding | `SEQ_MULTI_OF = 32` | Sequence padded to multiple of 32 with learned `cap_pad_token` and `x_pad_token` |

### AdaLN scope

- `noise_refiner` blocks: have `adaLN_modulation` ✓
- `context_refiner` blocks: **no `adaLN_modulation`** ✗ — caption refiner runs without timestep modulation
- Main `layers`: have `adaLN_modulation` ✓
- `all_final_layer`: has `adaLN_modulation` ✓ (different Sequential layout — see weight keys below)

## Text encoder — Qwen3-4B (full causal LM)

Source: `Tongyi-MAI/Z-Image-Turbo/text_encoder/config.json`.

| Field | Value |
|---|---|
| Architecture | `Qwen3ForCausalLM` (the **full LLM**, not encoder-only) |
| hidden_size | **2560** |
| num_hidden_layers | 36 |
| num_attention_heads | 32 |
| num_key_value_heads | 8 (**GQA**) |
| head_dim | 128 |
| intermediate_size | 9728 |
| vocab_size | 151,936 |
| max_position_embeddings | 40,960 |
| rope_theta | 1,000,000 |
| Norm | RMSNorm |
| FFN | SwiGLU |
| Tied embeddings | Yes |
| Distribution | ComfyUI ships as `qwen_3_4b.safetensors` |

The 2560 hidden directly matches the transformer's `cap_feat_dim=2560` → no projection needed. The only adapter is `cap_embedder`: an RMSNorm-style scale + Linear(2560 → 3840), 2 weights total.

**Embedding extraction:** Z-Image takes the **last hidden state** of the full Qwen3 forward (not encoder-only — the "encoder" here is just running the LLM and pulling its hidden states). It applies a **system prompt prefix** before tokenization; see `Comfyui-Z-Image-Utilities` for the canonical text. Parity requires using the same chat template + system prompt.

## VAE — Flux VAE, reused verbatim

Source: `Tongyi-MAI/Z-Image-Turbo/vae/config.json`.

- Class: `AutoencoderKL`, `_name_or_path: "flux-dev"` (literally the Flux VAE checkpoint).
- 16 latent channels, 8× spatial downscale.
- 4 down blocks of 2 layers each, channels `[128, 256, 512, 512]`.
- **scaling_factor: 0.3611, shift_factor: 0.1159** (Flux values).
- `force_upcast: true` (decode in FP32 for stability).
- No `quant_conv` / `post_quant_conv`.

**Implication:** Existing Flux VAE in SharpInference works as-is. Reuse the same scale/shift constants.

## Scheduler

Source: `Tongyi-MAI/Z-Image-Turbo/scheduler/scheduler_config.json` and `Tongyi-MAI/Z-Image/scheduler/scheduler_config.json`.

- `FlowMatchEulerDiscreteScheduler`
- `num_train_timesteps = 1000`
- `use_dynamic_shifting = false` — **simpler than Flux**, which uses dynamic shifting based on resolution.
- **`shift = 3.0` for Turbo, `shift = 6.0` for Base** — the only scheduler-config difference between the two variants.

**Recipes:**
- **Turbo:** `num_inference_steps = 9` (yields 8 NFEs), `cfg_scale = 1.0` (single forward per step, no CFG, no negative prompt).
- **Base:** `num_inference_steps = 28..50`, `cfg_scale = 3.0..5.0` (default 4.0), with a negative prompt; standard CFG (two forwards per step).

## Weight keys (transformer)

Top-level keys from `diffusion_pytorch_model.safetensors.index.json`:

```text
# Embedders
t_embedder.mlp.{0,2}.{weight,bias}            # 2-layer MLP timestep embedder, sinusoidal -> dim
all_x_embedder.2-1.{weight,bias}              # patch embedder (Linear). Name "2-1" comes from patch_size=2, f_patch=1
all_final_layer.2-1.linear.{weight,bias}      # output projection back to patch space
all_final_layer.2-1.adaLN_modulation.1.{weight,bias}  # final layer modulation (".1" -> Sequential index for Linear)
cap_embedder.0.weight                         # RMSNorm-style scale
cap_embedder.1.{weight,bias}                  # Linear 2560 -> 3840
cap_pad_token                                 # learned [1, dim] caption pad embedding
x_pad_token                                   # learned image-patch pad embedding

# Context refiner (no AdaLN)
context_refiner.{0,1}.attention.{to_q,to_k,to_v,to_out.0}.weight
context_refiner.{0,1}.attention.{norm_q,norm_k}.weight    # QK RMSNorm
context_refiner.{0,1}.{attention_norm1,attention_norm2}.weight
context_refiner.{0,1}.feed_forward.{w1,w2,w3}.weight       # SwiGLU
context_refiner.{0,1}.{ffn_norm1,ffn_norm2}.weight

# Noise refiner (with AdaLN)
noise_refiner.{0,1}.adaLN_modulation.0.{weight,bias}       # 256 -> 4*3840 = 15360. Note ".0" — different layout than final_layer
noise_refiner.{0,1}.attention.{to_q,to_k,to_v,to_out.0}.weight
noise_refiner.{0,1}.attention.{norm_q,norm_k}.weight
noise_refiner.{0,1}.{attention_norm1,attention_norm2}.weight
noise_refiner.{0,1}.feed_forward.{w1,w2,w3}.weight
noise_refiner.{0,1}.{ffn_norm1,ffn_norm2}.weight

# Main 30 transformer layers (same shape as noise_refiner)
layers.{0..29}.adaLN_modulation.0.{weight,bias}
layers.{0..29}.attention.{to_q,to_k,to_v,to_out.0}.weight
layers.{0..29}.attention.{norm_q,norm_k}.weight
layers.{0..29}.{attention_norm1,attention_norm2}.weight
layers.{0..29}.feed_forward.{w1,w2,w3}.weight
layers.{0..29}.{ffn_norm1,ffn_norm2}.weight
```

### Naming gotchas

- **`adaLN_modulation` Sequential index differs:** `Sequential(SiLU, Linear)` → for `noise_refiner` and `layers`, the Linear is index `0` (key `.0.weight`). For `all_final_layer.2-1`, the Linear is index `1` (key `.1.weight`). Match these positions exactly.
- **`attention.to_out` is wrapped:** `Sequential(Linear, Dropout)` → key is `to_out.0.weight`, no bias.
- **No biases on linear weights** except `t_embedder.mlp.*`, `adaLN_modulation` Linears, `all_final_layer.2-1.linear`, `all_x_embedder.2-1`.
- **FFN dim:** Lumina convention is `int(8/3 * dim)` rounded to multiple of 256. For dim=3840 that's ≈10240 — **verify with actual tensor shapes from the safetensors header**, don't hardcode.

## SwarmUI single-file FP8Mix checkpoint

`mcmonkey/swarm-models/SwarmUI_Z-Image-Turbo-FP8Mix.safetensors` (~6.57 GB).

- At ~1 byte/param for ~6B params → **FP8-mixed transformer-only** (not bundled with text encoder or VAE).
- "Mix" suffix = sensitive layers (`t_embedder`, `cap_embedder`, `adaLN_modulation`, `all_final_layer`) kept in BF16/FP16; bulk `layers.*.{attention,feed_forward}` linears stored as `torch.float8_e4m3fn`.
- **Same key naming** as the diffusers index above (the format originates in the ComfyUI/Lumina2 lineage).
- Per-tensor handling at load: read dtype from safetensors header. If `F8_E4M3` → route through existing FP8 path (cast to F16 at GEMM time, or native FP8 GEMM on Ada+). Non-FP8 tensors will be in BF16 → existing BF16→F16 path.
- **Text encoder and VAE must be supplied separately** (`qwen_3_4b.safetensors` + Flux VAE `ae.safetensors`), just like ComfyUI does.

## Implementation strategy for SharpInference

1. **`ZImageConfig.cs`** — record with the fields above, `Turbo` static preset. Auto-detect via weight keys (`t_embedder.mlp.0.weight` + `cap_embedder.1.weight` + `all_x_embedder.2-1.weight` + 30 `layers.*` blocks).
2. **`ZImageTransformer.cs`** — new top-level class. Cannot reuse `FluxTransformer` (different modulation count, sequence assembly, RoPE). Sub-components reusable: `SwiGluFfn`, `QkNorm`, `AdaLNModulation` (with 4-output config). New: `ZImageBlock`, `ZImageContextRefinerBlock`, `ZImageNoiseRefinerBlock`, `ZImageRope` (multi-axis [32,48,48], theta=256).
3. **Qwen3-4B text encoder** — needed alongside (or in `SharpInference.LLM` if that's where Qwen lives). 36 layers, GQA 32→8, RoPE θ=1e6, RMSNorm, SwiGLU, tied embeddings. Apply Z-Image chat template + system prompt before tokenization.
4. **VAE** — reuse existing Flux VAE path verbatim.
5. **Scheduler** — extend flow-match Euler with static `shift=3.0`. No dynamic shifting branch.
6. **`ZImageCheckpointConverter.cs`** — passthrough naming for diffusers/single-file format. Detect FP8Mix by dtype-inspecting `layers.0.attention.to_q.weight`.
7. **`ZImagePipeline.cs`** — encode prompt with Qwen3-4B (last hidden state, full sequence) → pad to multiple of 32 with `cap_pad_token` → 2× `context_refiner` → patchify+embed image latents → 2× `noise_refiner` (with t-modulation) → concat `[caption, image]` → 30× main `layers` → final layer → unpatchify → VAE decode.

## Forward pass diagram

```
prompt
  ↓ (Z-Image system prompt + chat template)
Qwen3-4B tokenize + forward → last_hidden_state [B, L, 2560]
  ↓ cap_embedder (RMSNorm scale + Linear 2560→3840)
caption_tokens [B, L, 3840]
  ↓ pad to multiple of 32 with cap_pad_token
  ↓ 2× context_refiner (no AdaLN)
refined_caption [B, L_pad, 3840]

noise latent [B, 16, H, W]
  ↓ all_x_embedder (patchify 2×2 → Linear)
image_tokens [B, H/2 * W/2, 3840]
  ↓ pad to multiple of 32 with x_pad_token
  ↓ 2× noise_refiner (with t-modulation, AdaLN)
refined_image [B, S_pad, 3840]

t (sigma)
  ↓ t_embedder (sinusoidal × t_scale=1000 → MLP)
t_emb [B, 256]                    (note: 256, not 3840 — AdaLN_EMBED_DIM)

concat [refined_caption, refined_image] along sequence axis
  ↓ 30× ZImageBlock (RMSNorm → QK-norm MHA with multi-axis RoPE → AdaLN(t_emb, 4 outputs) → SwiGLU FFN)
  ↓ all_final_layer (AdaLN(t_emb, 2 outputs) → RMSNorm → Linear 3840 → patch_dim)

slice off image-token portion → unpatchify → latent [B, 16, H, W]
  ↓ Flux VAE decode (scale=0.3611, shift=0.1159)
image
```

## Key references

- [Tongyi-MAI/Z-Image-Turbo (HF)](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) — model card, configs.
- [Z-Image-Turbo transformer/config.json](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/main/transformer/config.json)
- [Z-Image-Turbo text_encoder/config.json](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/main/text_encoder/config.json) — Qwen3-4B.
- [Z-Image-Turbo vae/config.json](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/main/vae/config.json) — Flux VAE.
- [Z-Image-Turbo scheduler/scheduler_config.json](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/main/scheduler/scheduler_config.json)
- [Tongyi-MAI/Z-Image GitHub](https://github.com/Tongyi-MAI/Z-Image) — official PyTorch reference (Apache-2.0).
- [diffusers transformer_z_image.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_z_image.py) — canonical Python port. Classes: `ZImageTransformer2DModel`, `ZImageTransformerBlock`, `ZSingleStreamAttnProcessor`, `TimestepEmbedder`, `RopeEmbedder`, `FinalLayer`.
- [diffusers pipelines/z_image/](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/z_image) — `pipeline_z_image.py`, `pipeline_z_image_img2img.py`.
- [diffusers PR #12756](https://github.com/huggingface/diffusers/pull/12756) — adds `from_single_file` for `ZImageTransformer2DModel`. Note the dtype fix `t_freq.to(self.mlp[0].compute_dtype)` — replicate in our timestep embedder for FP8 robustness.
- [ComfyUI workflow template image_z_image_turbo.json](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_z_image_turbo.json) — numerical-parity reference (sampler `euler`, scheduler `simple`/`normal`, steps 8, cfg 1.0, shift 3.0).
- [Comfyui-Z-Image-Utilities](https://github.com/Koko-boya/Comfyui-Z-Image-Utilities) — official Z-Image system prompt verbatim.
- [Qwen3-4B config](https://huggingface.co/Qwen/Qwen3-4B/blob/main/config.json) — text encoder reference.
- [SwarmUI_Z-Image-Turbo-FP8Mix.safetensors](https://huggingface.co/mcmonkey/swarm-models/resolve/main/SwarmUI_Z-Image-Turbo-FP8Mix.safetensors) — single-file FP8Mix distribution we're targeting first.
- [Z-Image base release discussion](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/discussions/88)

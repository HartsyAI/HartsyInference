# Classifier-Free Guidance — Research Notes

## Table of Contents

1. [Summary](#1-summary)
2. [Detailed Findings](#2-detailed-findings)
3. [Key Numbers/Constants](#3-key-numbersconstants)
4. [Data Layouts/Formats](#4-data-layoutsformats)
5. [Algorithm Steps](#5-algorithm-steps)
6. [Reference Implementations](#6-reference-implementations)
7. [Differences Between Implementations](#7-differences-between-implementations)
8. [Open Questions](#8-open-questions)
9. [Implementation Notes](#9-implementation-notes)

---

## 1. Summary

Classifier-Free Guidance (CFG) is the mechanism that makes text prompts actually control image generation in diffusion models. During each denoising step, the model runs **two forward passes** — one conditioned on the text prompt and one conditioned on an empty/negative prompt — and combines the outputs to steer generation toward the prompt. The core formula is:

```
output = uncond + guidance_scale * (cond - uncond)
```

This document covers:
- The original CFG formulation from Ho & Salimans (2022)
- Negative prompt encoding and how it replaces the unconditional pass
- CFG Rescale to prevent overexposure at high guidance scales
- Flux's guidance embedding approach (single forward pass instead of double)
- Perturbed Attention Guidance (PAG) as an alternative/complementary technique

---

## 2. Detailed Findings

### 2.1 Classifier-Free Guidance (Ho & Salimans, 2022)

**Paper:** "Classifier-Free Diffusion Guidance" ([arXiv:2207.12598](https://arxiv.org/abs/2207.12598))

#### Background: Classifier Guidance

Earlier work by Dhariwal & Nichol (2021) used a separately trained image classifier to guide diffusion sampling. The guided score estimate was:

```
score_guided = score_uncond(x_t, t) + s * grad_x log p(y | x_t)
```

This required training an auxiliary classifier on noisy images, which is expensive and limits flexibility.

#### Classifier-Free Approach

Ho & Salimans eliminated the need for a separate classifier by jointly training a conditional and an unconditional diffusion model. During training, the conditioning signal `c` is randomly dropped (replaced with a null token) with probability `p_uncond` (typically 10-20%). This teaches the same network to function as both a conditional and unconditional model.

At inference time, the conditional and unconditional predictions are combined:

```
eps_guided(x_t, t, c) = eps_uncond(x_t, t) + w * (eps_cond(x_t, t, c) - eps_uncond(x_t, t))
```

Or equivalently, using the formulation from the paper:

```
eps_guided(x_t, t, c) = (1 + w) * eps_cond(x_t, t, c) - w * eps_uncond(x_t, t)
```

where `w` is the guidance weight. In practice, implementations use `guidance_scale = 1 + w`, so the formula becomes:

```
eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
```

- `guidance_scale = 1.0`: No guidance (purely conditional prediction)
- `guidance_scale > 1.0`: Amplifies the effect of conditioning
- `guidance_scale < 1.0`: Dampens conditioning (rarely used)

#### Training with Unconditional Dropout

During training, for each sample in the batch, the conditioning `c` is replaced with a null embedding with probability `p_uncond`:

```
c_input = null_embedding   if random() < p_uncond
c_input = encode(prompt)   otherwise
```

Typical `p_uncond` values: 0.1 for SD1.5/SDXL, 0.1-0.2 in general.

**Sources:**
- [Ho & Salimans, 2022 — Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
- [Dhariwal & Nichol, 2021 — Diffusion Models Beat GANs](https://arxiv.org/abs/2105.05233)

### 2.2 Negative Prompt Encoding

In the original CFG formulation, the "unconditional" prediction uses an empty string encoded through the text encoder (CLIP for SD/SDXL). This produces a null embedding that represents "no particular content."

**Negative prompts** replace this null embedding with an encoded negative prompt. The mechanism is straightforward:

1. The positive prompt is tokenized and encoded through CLIP: `cond_embedding = CLIP(positive_prompt)`
2. The negative prompt (defaulting to `""`) is tokenized and encoded through CLIP: `uncond_embedding = CLIP(negative_prompt)`
3. Both embeddings have shape `[batch, 77, 768]` for SD1.5 or `[batch, 77, 2048]` for SDXL (77 is CLIP's max token length)
4. CFG is applied using the negative-prompt embedding as the "unconditional" prediction

The CFG formula with negative prompts:

```
noise_pred = noise_pred_negative + guidance_scale * (noise_pred_positive - noise_pred_negative)
```

This means: steer generation **toward** the positive prompt and **away from** the negative prompt, with `guidance_scale` controlling the strength. A negative prompt like "blurry, low quality" causes the model to actively avoid those concepts.

**Practical batching:** In practice, both the conditional and unconditional inputs are concatenated into a single batch and run through the UNet in one forward pass for efficiency:

```python
# Concatenate for batched inference
latent_model_input = torch.cat([latents] * 2)          # [2*B, C, H, W]
prompt_embeds = torch.cat([negative_embeds, positive_embeds])  # [2*B, 77, D]

# Single forward pass with batch size 2*B
noise_pred = unet(latent_model_input, t, prompt_embeds)

# Split and apply CFG
noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
```

**Sources:**
- [AUTOMATIC1111 Wiki — Negative Prompt](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Negative-prompt)
- [diffusers StableDiffusionPipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py)

### 2.3 CFG Rescale

**Paper:** "Common Diffusion Noise Schedules and Sample Steps are Flawed" — Lin et al., WACV 2024 ([arXiv:2305.08891](https://arxiv.org/abs/2305.08891))

High guidance scales (e.g., > 10) cause overexposure and oversaturation because CFG increases the standard deviation of the noise prediction beyond what the model was trained to produce. Section 3.4 of the paper proposes **CFG Rescale** to fix this.

#### The Problem

When `guidance_scale` is high, the guided noise prediction `noise_cfg` has a much larger standard deviation than the conditioned prediction `noise_pred_text`. This pushes pixel values toward extremes, causing washed-out or oversaturated images.

#### The Formula

The rescale technique normalizes the CFG output to match the standard deviation of the conditioned prediction:

```python
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    # Compute std per sample (across all spatial/channel dims)
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)

    # Rescale to match conditioned prediction's std
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)

    # Blend between rescaled and original
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg
```

- `guidance_rescale = 0.0`: No rescaling (default, backwards compatible)
- `guidance_rescale = 0.7`: Recommended starting point for models trained with v-prediction
- `guidance_rescale = 1.0`: Full rescaling (forces output std to match conditioned prediction exactly)

The standard deviation is computed per sample in the batch, across all channels and spatial dimensions. This preserves relative magnitudes between samples while preventing the overall scale from blowing up.

**Sources:**
- [Lin et al., 2023 — Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/abs/2305.08891)
- [diffusers rescale_noise_cfg implementation](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py)

### 2.4 Flux Guidance Embedding (Single Forward Pass)

Flux (Black Forest Labs) handles guidance fundamentally differently from SD1.5/SDXL. Instead of running two forward passes and combining them, Flux encodes the guidance scale directly as an embedding that conditions the model during a single forward pass.

#### How It Works

In Flux's forward pass, the conditioning vector `vec` is constructed by summing multiple embeddings:

```python
# 1. Timestep embedding (sinusoidal, 256-dim, projected to hidden_size)
vec = self.time_in(timestep_embedding(timesteps, 256))

# 2. Guidance embedding (only for guidance-distilled models like Flux-dev)
if self.params.guidance_embed:
    vec = vec + self.guidance_in(timestep_embedding(guidance, 256))

# 3. Pooled text embedding
vec = vec + self.vector_in(y)
```

The key components:
- `timestep_embedding(x, dim=256)`: Creates 256-dimensional sinusoidal positional encoding from scalar values
- `guidance_in`: An `MLPEmbedder(in_dim=256, hidden_dim=3072)` — a small MLP that projects the 256-dim sinusoidal embedding to the model's hidden dimension (3072)
- The guidance value is treated identically to the timestep: converted to sinusoidal features, then linearly projected

The resulting `vec` modulates the transformer blocks via adaptive layer norm (adaLN), influencing scale and shift parameters throughout all double-stream and single-stream blocks.

#### Guidance Distillation

Flux-dev was trained via **guidance distillation**: the student model learns to replicate the output of a teacher model that uses traditional CFG, but conditioned on the guidance scale value. This means:

- The guidance behavior is "baked into" the model weights
- A single forward pass produces results equivalent to what would require two passes with traditional CFG
- The model learns to internally approximate the `uncond + scale * (cond - uncond)` computation

#### Flux-dev vs Flux-schnell

| Property | Flux-dev | Flux-schnell |
|----------|----------|--------------|
| `guidance_embed` | `true` | `false` |
| Guidance scale | Adjustable (default 3.5) | Fixed/none (baked in during distillation) |
| Forward passes per step | 1 | 1 |
| Steps | 20-50 | 1-4 |

Flux-schnell was distilled with a fixed guidance value (reportedly ~3.5), so it has no guidance embedding input at all — the guidance effect is permanently encoded in its weights.

#### Performance Advantage

The guidance embedding approach provides a ~2x speedup over traditional CFG because:
- Traditional CFG: 2 forward passes per denoising step (or 1 batched pass at 2x batch size = 2x compute + 2x VRAM for activations)
- Flux guidance embedding: 1 forward pass per step at normal batch size

For a 50-step generation with a 12B parameter model like Flux, this saves 50 full forward passes, which is substantial.

**Sources:**
- [Flux GitHub — guidance issue #63](https://github.com/black-forest-labs/flux/issues/63)
- [Flux GitHub — guidance issue #159](https://github.com/black-forest-labs/flux/issues/159)
- [DeepWiki — Flux Model Architecture](https://deepwiki.com/black-forest-labs/flux/4.1-flux-model-architecture)
- [Hugging Face — Flux Pipeline](https://huggingface.co/docs/diffusers/main/api/pipelines/flux)

### 2.5 Perturbed Attention Guidance (PAG)

**Paper:** "Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance" — Ahn et al., ECCV 2024 ([arXiv:2403.17377](https://arxiv.org/abs/2403.17377))

PAG is an alternative guidance technique that does not require an unconditional/negative-prompt forward pass. Instead, it modifies the self-attention maps within the model to create a structurally degraded prediction, then guides away from the degradation.

#### Core Mechanism

In selected layers of the UNet/DiT, the self-attention computation:

```
SA(Q, K, V) = softmax(Q * K^T / sqrt(d)) * V
```

is replaced with an identity operation:

```
PSA(Q, K, V) = I * V = V
```

where `I` is an identity matrix of size `(hw x hw)`. This bypasses the attention mechanism entirely, removing structural/spatial information while preserving per-token features.

#### PAG Formula

The guided output is:

```
eps_pag(x_t) = eps_theta(x_t) + s_pag * (eps_theta(x_t) - eps_perturbed(x_t))
```

where:
- `eps_theta(x_t)` is the normal (unperturbed) model prediction
- `eps_perturbed(x_t)` is the prediction with identity-replaced self-attention in selected layers
- `s_pag` is the PAG guidance scale (default 3.0)

This is structurally similar to CFG but the "unconditional" prediction is replaced by a "structurally degraded" prediction.

#### Combining PAG with CFG

When used together with CFG, both guidance signals are applied:

```
# Step 1: Apply CFG
eps_cfg = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

# Step 2: Apply PAG on top
eps_final = eps_cfg + pag_scale * (eps_cond - eps_perturbed)
```

This means PAG requires an **additional** forward pass (the perturbed pass), so using CFG + PAG together requires 3 forward passes per step total (unconditional, conditional, perturbed).

#### Which Layers to Apply PAG

The choice of layers significantly affects results:

| Model | Recommended `pag_applied_layers` |
|-------|----------------------------------|
| SD1.5 (conditional) | `["input_blocks.14.1"]` (equivalent to `["down.block_2"]`) |
| SD1.5 (unconditional) | `["input_blocks.14.1", "input_blocks.16.1", "input_blocks.17.1", "middle_block.1"]` |
| SDXL | `["mid"]` (default) |
| SD3 | Supported, layer selection varies |
| PixArt-Sigma | Supported |

The mid block and deeper down blocks contain the most structural information, making them the most effective targets for perturbation.

#### Performance Characteristics

- PAG alone (no CFG): 2 forward passes per step (normal + perturbed)
- PAG + CFG: 3 forward passes per step (unconditional + conditional + perturbed)
- PAG alone can achieve good results even with empty prompts (unconditional generation)
- Default `pag_scale = 3.0` works well for most cases; higher values can cause smoothing

**Sources:**
- [Ahn et al., 2024 — Perturbed-Attention Guidance](https://arxiv.org/abs/2403.17377)
- [Hugging Face Diffusers — PAG Guide](https://huggingface.co/docs/diffusers/en/using-diffusers/pag)
- [Hugging Face Diffusers — PAG API](https://huggingface.co/docs/diffusers/main/api/pipelines/pag)

### 2.6 Self-Attention Guidance (SAG) — not PAG, added 2026-08-11

**Paper:** "Self-Attention Guidance: Improving Sample Quality of Diffusion Models Using Self-Attention" — Hong et al., CVPR 2023 ([arXiv:2210.00939](https://arxiv.org/abs/2210.00939))

This section previously didn't exist — this doc's Tier-2.3 framing (in the extension backlog plan) had grouped SAG with PAG as though they were the same mechanism-family with different names. **They aren't.** PAG replaces self-attention with an identity op in selected layers (§2.5); SAG does something structurally different: it uses the self-attention map itself as a saliency signal to **blur the diffusion input**, not the attention computation.

#### Core Mechanism

1. Run a normal forward pass, capturing the self-attention map from one designated layer (usually a mid-resolution layer, analogous to PAG's `["mid"]` default).
2. Average the attention map over heads, threshold it (e.g. mean + a multiplier) to get a soft mask of "attended-to" (high-saliency) regions of the latent.
3. Gaussian-blur the **input latent** `x_t` — not the attention output — restricted to (or blended by) that mask, producing `x_t_blurred`.
4. Run a **second** forward pass on `x_t_blurred` to get `eps_blurred`.
5. Guide away from the blurred prediction: `eps_sag = eps_theta(x_t) + sag_scale * (eps_theta(x_t) - eps_blurred(x_t_blurred))`.

The self-attention map is only ever *read* (to build the blur mask) — SAG never patches or replaces the attention computation the way PAG does. This means SAG's forward-pass count is 2 without CFG (normal + blurred) or 3 with CFG (uncond + cond + blurred-cond), the same count as PAG, but the *mechanism* needing new engine machinery is different: PAG needs an attention-computation hook (`IAttentionHook`-shaped, §9 below); SAG needs (a) a way to read out an intermediate attention map from a specific block without disrupting the forward pass, and (b) a Gaussian-blur-with-mask operator on a 4D latent tensor. Neither of those exists in the engine today (confirmed: no attention-map-readout parameter on any `Sdpa`-family backend call, no blur kernel outside `HartsyInference.Vision`'s unrelated CV pipelines). Do not assume PAG's hook design (below) covers SAG — it only covers half of this cluster.

**Sources:**
- [Hong et al., 2023 — Self-Attention Guidance](https://arxiv.org/abs/2210.00939)
- [Hugging Face Diffusers — SAG Pipeline](https://huggingface.co/docs/diffusers/en/api/pipelines/self_attention_guidance)

---

## 3. Key Numbers/Constants

### Guidance Scale Defaults

| Model | Default guidance_scale | Recommended Range | Notes |
|-------|------------------------|-------------------|-------|
| SD1.5 | 7.5 | 5–15 | Higher = more prompt adherence, more artifacts |
| SDXL | 5.0–7.5 | 3–10 | Generally needs less than SD1.5 |
| SD3 | 4.0–4.5 | 3–7 | Flow-matching model |
| Flux-dev | 3.5 | 1–5 | Guidance embedding, not CFG |
| Flux-schnell | N/A | N/A | Guidance baked in, no parameter |
| LCM | 1.0–2.0 | 1–3 | Distilled models need very low CFG |

### PAG Constants

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `pag_scale` | 3.0 | 0–5 | 0 disables PAG |
| `pag_adaptive_scale` | 0.0 | 0–1 | Enables adaptive scaling |

### CFG Rescale Constants

| Parameter | Default | Recommended | Notes |
|-----------|---------|-------------|-------|
| `guidance_rescale` | 0.0 | 0.0–0.7 | 0.7 for v-prediction models |

### Flux Guidance Embedding Dimensions

| Component | Value |
|-----------|-------|
| Sinusoidal embedding dim | 256 |
| MLP hidden dim (guidance_in) | 3072 |
| MLP architecture | Linear(256, 3072) -> SiLU -> Linear(3072, 3072) |

### Training Unconditional Dropout

| Model | p_uncond |
|-------|----------|
| SD1.5 | 0.1 (10%) |
| SDXL | 0.1 (10%) |
| General recommendation | 0.1–0.2 |

---

## 4. Data Layouts/Formats

### Prompt Embeddings (SD1.5)

```
positive_embeds:  [batch, 77, 768]   # CLIP ViT-L/14
negative_embeds:  [batch, 77, 768]   # Same shape, empty string or negative prompt
```

### Prompt Embeddings (SDXL)

```
positive_embeds:  [batch, 77, 2048]  # Concatenated CLIP ViT-L/14 (768) + OpenCLIP ViT-bigG (1280)
negative_embeds:  [batch, 77, 2048]  # Same shape
positive_pooled:  [batch, 1280]      # OpenCLIP pooled output
negative_pooled:  [batch, 1280]      # Same shape
```

### Batched UNet Input (with CFG)

```
latent_input:     [2*batch, 4, H/8, W/8]    # Duplicated latents
prompt_embeds:    [2*batch, 77, D]           # [negative; positive] concatenated
timesteps:        [2*batch]                  # Same timestep repeated
```

The first half of the batch dimension is unconditional/negative, the second half is conditional/positive.

### Noise Prediction Output

```
noise_pred:       [2*batch, 4, H/8, W/8]    # Split into uncond and cond halves
noise_pred_uncond = noise_pred[:batch]
noise_pred_cond   = noise_pred[batch:]
```

### Flux Guidance Input

```
timesteps:        [batch]                     # Float tensor of timestep values
guidance:         [batch]                     # Float tensor of guidance scale values
y (pooled text):  [batch, hidden_dim]         # Pooled CLIP embedding
```

No batch doubling needed — single forward pass.

---

## 5. Algorithm Steps

### 5.1 Standard CFG Pipeline (SD1.5/SDXL)

```
INPUTS:
  prompt: string
  negative_prompt: string (default "")
  guidance_scale: float (default 7.5)
  num_inference_steps: int (default 50)
  guidance_rescale: float (default 0.0)

ALGORITHM:
  1. Encode positive prompt:     cond_embeds = text_encoder(tokenize(prompt))
  2. Encode negative prompt:     uncond_embeds = text_encoder(tokenize(negative_prompt))
  3. Initialize latents:         latents = randn(batch, 4, H/8, W/8) * init_noise_sigma
  4. Set scheduler timesteps:    scheduler.set_timesteps(num_inference_steps)

  5. FOR EACH timestep t in scheduler.timesteps:
     a. Duplicate latents:       latent_input = concat([latents, latents], dim=0)
     b. Scale model input:       latent_input = scheduler.scale_model_input(latent_input, t)
     c. Concatenate embeddings:  embeds = concat([uncond_embeds, cond_embeds], dim=0)
     d. Run UNet:                noise_pred = unet(latent_input, t, embeds)
     e. Split predictions:       noise_uncond, noise_cond = noise_pred.chunk(2)
     f. Apply CFG:               noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
     g. Apply CFG Rescale (optional):
        IF guidance_rescale > 0:
           std_text = std(noise_cond, per_sample)
           std_cfg  = std(noise_pred, per_sample)
           rescaled = noise_pred * (std_text / std_cfg)
           noise_pred = guidance_rescale * rescaled + (1 - guidance_rescale) * noise_pred
     h. Scheduler step:          latents = scheduler.step(noise_pred, t, latents)

  6. Decode latents:             image = vae.decode(latents / vae_scaling_factor)
```

### 5.2 Flux Guidance Pipeline

```
INPUTS:
  prompt: string
  guidance_scale: float (default 3.5)   # Only for Flux-dev; ignored for schnell
  num_inference_steps: int (default 50)

ALGORITHM:
  1. Encode text:                txt_embeds = t5_encoder(prompt)
  2. Encode pooled text:         pooled = clip_encoder(prompt)  # pooled output
  3. Initialize latents:         latents = randn(batch, C, H/8, W/8)
  4. Set scheduler timesteps:    scheduler.set_timesteps(num_inference_steps)

  5. FOR EACH timestep t in scheduler.timesteps:
     a. Prepare guidance tensor: guidance = tensor([guidance_scale] * batch)
     b. Run DiT (single pass):   velocity = dit(latents, t, txt_embeds, pooled, guidance)
     c. Scheduler step:          latents = scheduler.step(velocity, t, latents)
        # Flow-matching Euler: latents = latents + velocity * dt

  6. Decode latents:             image = vae.decode(latents)
```

No batch doubling. No second forward pass. The guidance scale is embedded via sinusoidal encoding and added to the timestep conditioning vector inside the model.

### 5.3 PAG Pipeline (with CFG)

```
INPUTS:
  prompt: string
  negative_prompt: string (default "")
  guidance_scale: float (default 7.0)
  pag_scale: float (default 3.0)
  pag_applied_layers: list (default ["mid"])

ALGORITHM:
  1. Encode prompts (same as standard CFG)
  2. Initialize latents
  3. Set scheduler timesteps
  4. Install PAG hooks on specified layers (replace self-attention with identity)

  5. FOR EACH timestep t in scheduler.timesteps:
     a. Triplicate latents:      latent_input = concat([latents, latents, latents], dim=0)
     b. Concatenate embeddings:  embeds = concat([uncond_embeds, cond_embeds, cond_embeds], dim=0)
     c. Run UNet with PAG hooks:
        - First 1/3 of batch: unconditional pass (normal attention)
        - Second 1/3 of batch: conditional pass (normal attention)
        - Third 1/3 of batch: conditional pass with identity self-attention in selected layers
        noise_pred = unet(latent_input, t, embeds)
     d. Split predictions:       noise_uncond, noise_cond, noise_perturbed = noise_pred.chunk(3)
     e. Apply CFG:               noise_cfg = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
     f. Apply PAG:               noise_pred = noise_cfg + pag_scale * (noise_cond - noise_perturbed)
     g. Scheduler step:          latents = scheduler.step(noise_pred, t, latents)

  6. Decode latents
```

---

## 6. Reference Implementations

### Primary References

| Implementation | Language | URL |
|---------------|----------|-----|
| diffusers StableDiffusionPipeline | Python | [GitHub](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py) |
| diffusers StableDiffusionXLPipeline | Python | [GitHub](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py) |
| diffusers FluxPipeline | Python | [GitHub](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py) |
| diffusers PAG utilities | Python | [GitHub](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py) |
| Flux model (Black Forest Labs) | Python | [GitHub](https://github.com/black-forest-labs/flux) |
| diffusers ClassifierFreeGuidance guider | Python | [GitHub](https://github.com/huggingface/diffusers/blob/main/src/diffusers/guiders/classifier_free_guidance.py) |
| PAG official implementation | Python | [GitHub](https://github.com/cvlab-kaist/Perturbed-Attention-Guidance) |

### Key Code Snippets

**CFG application (from diffusers StableDiffusionPipeline):**
```python
if self.do_classifier_free_guidance:
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)
```

**CFG Rescale (from diffusers):**
```python
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg
```

**Flux guidance embedding (from Black Forest Labs flux model):**
```python
vec = self.time_in(timestep_embedding(timesteps, 256))
if self.params.guidance_embed:
    vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
vec = vec + self.vector_in(y)
```

### Papers

| Paper | Authors | Year | Venue | arXiv |
|-------|---------|------|-------|-------|
| Classifier-Free Diffusion Guidance | Ho & Salimans | 2022 | NeurIPS Workshop | [2207.12598](https://arxiv.org/abs/2207.12598) |
| Common Diffusion Noise Schedules are Flawed | Lin et al. | 2024 | WACV | [2305.08891](https://arxiv.org/abs/2305.08891) |
| Perturbed-Attention Guidance | Ahn et al. | 2024 | ECCV | [2403.17377](https://arxiv.org/abs/2403.17377) |
| Scaling Rectified Flow Transformers (SD3) | Esser et al. | 2024 | ICML | [2403.03206](https://arxiv.org/abs/2403.03206) |

---

## 7. Differences Between Implementations

### SD1.5 vs SDXL CFG

| Aspect | SD1.5 | SDXL |
|--------|-------|------|
| Text encoder | CLIP ViT-L/14 (768-dim) | Dual CLIP (768+1280 = 2048-dim) |
| Pooled embeddings | Not used in CFG | Used (added to time embedding) |
| Default guidance_scale | 7.5 | 5.0–7.5 |
| Negative pooled embedding | N/A | Zero tensor or encoded negative |

### SD1.5/SDXL vs SD3

| Aspect | SD1.5/SDXL | SD3 |
|--------|------------|-----|
| Architecture | UNet | DiT (MM-DiT) |
| Prediction type | Epsilon (noise) | Velocity (flow) |
| CFG formula | Same formula | Same formula, applied to velocity predictions |
| Default guidance_scale | 7.5 / 5.0 | 4.0–4.5 |
| Text encoders | 1 or 2 CLIP | 2 CLIP + T5 |

### SD/SDXL vs Flux (Fundamental Divergence)

| Aspect | SD/SDXL (Traditional CFG) | Flux (Guidance Embedding) |
|--------|--------------------------|---------------------------|
| Forward passes per step | 2 (or 1 batched at 2x size) | 1 |
| VRAM for guidance | ~2x model activations | ~1x model activations |
| Negative prompts | Supported natively | Not applicable (no unconditional pass) |
| Guidance flexibility | Any scale at inference | Constrained to trained range |
| Compute cost per step | ~2x base | ~1x base |
| Mechanism | Output-space interpolation | Input-space conditioning |
| Training requirement | Unconditional dropout | Guidance distillation from teacher |

### CFG vs PAG

| Aspect | CFG | PAG |
|--------|-----|-----|
| What is modified | Conditioning (text vs null) | Self-attention maps (normal vs identity) |
| Extra forward passes | +1 (unconditional) | +1 (perturbed) |
| Works unconditionally | No (requires conditioning contrast) | Yes (structural guidance) |
| Training required | Yes (unconditional dropout) | No (modifies inference only) |
| What it improves | Prompt adherence | Structural coherence |
| Can combine with other | Standalone | Can combine with CFG |

### True CFG vs Flux Guidance: When to Prefer Each

**True CFG advantages:**
- Negative prompts allow explicit exclusion of unwanted content
- Works with any model trained with unconditional dropout
- Well-understood behavior, extensive community knowledge
- Guidance scale can be any value (not bounded by distillation range)

**Flux guidance embedding advantages:**
- 2x faster inference per step
- 2x less VRAM for the guidance component
- Simpler pipeline code (no batch doubling)
- Quality may be higher at equivalent perceptual guidance strength (Flux-dev guidance_scale=3.5 ~ SD CFG=7)

---

## 8. Open Questions

- [x] ~~Whether PAG can be applied to Flux DiT architecture (no UNet)~~ — **Answered 2026-08-11**, from direct knowledge of this engine's `FluxTransformer`/`Sd3Transformer`/`ChromaTransformer`/`HiDreamTransformer` (gained wiring step-cache into all four, §6 of `ROADMAP.md`): yes, mechanically simpler here than the UNet case. Every block's self-attention is a single `backend.ScaledDotProductAttention(...)` call site inside that block's class (`FluxDoubleStreamBlock`/`FluxSingleStreamBlock`, `ChromaDoubleStreamBlock`/`ChromaSingleStreamBlock`, `HiDreamBlock`, SD3's `JointBlock`) — there is no equivalent of UNet's named `input_blocks.14.1`-style path to look up; the unit of selection is a numeric block index into a homogeneous loop (`config.Depth` for SD3, `NumLayers`/`NumSingleLayers` for HiDream, double+single counts for Flux/Chroma), the same loop shape the step-cache work already threads a `stepCache` parameter through. See the updated §9 hook sketch below for the concrete fork point.
- [ ] Optimal `guidance_rescale` values for different models/schedulers — 0.7 is recommended for v-prediction models, but model-specific tuning may be needed
- [ ] Whether adapter-based guidance distillation (AGD) is practical for arbitrary models — Recent research ([arXiv:2503.07274](https://arxiv.org/abs/2503.07274)) suggests lightweight adapters can simulate CFG in a single pass
- [ ] **New (2026-08-11): SAG has no engine-side prerequisite machinery at all** — unlike PAG (which only needs a per-block attention-computation branch, mechanically cheap given the single-call-site shape above), SAG needs an attention-map readout path (nothing today lets a caller inspect an intermediate `Sdpa` call's attention weights — the backend call returns only the attended output) and a masked Gaussian-blur op on a 4D latent (doesn't exist outside `HartsyInference.Vision`'s unrelated CV code). Scope SAG as its own estimate, not "PAG's sibling, same size."

---

## 9. Implementation Notes

### C# Implementation Strategy

#### CFG Application

The core CFG operation is a simple element-wise computation:

```csharp
// noise_pred, noise_uncond, noise_cond are all Tensor<float> of shape [B, C, H, W]
// This is embarrassingly parallel — ideal for SIMD or GPU kernel
for (int i = 0; i < length; i++)
{
    noisePred[i] = noiseUncond[i] + guidanceScale * (noiseCond[i] - noiseUncond[i]);
}
```

This should be implemented as:
1. A fused SIMD kernel for CPU (using `Vector256<float>`)
2. A simple CUDA kernel for GPU

The CFG computation is negligible compared to the UNet/DiT forward pass itself.

#### Batch Doubling for CFG

The key performance decision is whether to:
1. **Batch both passes together** (concatenate latents + embeddings, run once with 2x batch): Maximizes GPU utilization, uses 2x VRAM for activations
2. **Run two separate passes** (unconditional then conditional): Uses 1x VRAM for activations, but loses parallelism

For GPU inference with sufficient VRAM, option 1 is preferred. For CPU or memory-constrained GPU, option 2 is necessary.

```csharp
public interface IGuidanceStrategy
{
    /// <summary>
    /// Prepares model inputs by duplicating/tripling latents and combining embeddings.
    /// Returns the batched inputs ready for a single forward pass.
    /// </summary>
    (Tensor latents, Tensor embeddings) PrepareInputs(
        Tensor latents, Tensor condEmbeds, Tensor uncondEmbeds);

    /// <summary>
    /// Applies guidance to the model output (splits batch, computes guided prediction).
    /// </summary>
    Tensor ApplyGuidance(Tensor modelOutput, float guidanceScale);
}
```

#### CFG Rescale Implementation

```csharp
static Tensor RescaleNoiseCfg(Tensor noiseCfg, Tensor noiseConditioned, float guidanceRescale)
{
    if (guidanceRescale <= 0f) return noiseCfg;

    // Compute std per sample (reduce over C, H, W dimensions)
    var stdText = noiseConditioned.Std(dims: new[] { 1, 2, 3 }, keepdim: true);
    var stdCfg = noiseCfg.Std(dims: new[] { 1, 2, 3 }, keepdim: true);

    // Rescale and blend
    var rescaled = noiseCfg * (stdText / stdCfg);
    return guidanceRescale * rescaled + (1f - guidanceRescale) * noiseCfg;
}
```

#### Flux Guidance Embedding

For Flux, guidance is handled inside the model forward pass, not in the pipeline:

```csharp
// Inside FluxTransformer.Forward():
var vec = TimeIn(TimestepEmbedding(timesteps, 256));
if (config.GuidanceEmbed)
{
    vec = vec + GuidanceIn(TimestepEmbedding(guidance, 256));
}
vec = vec + VectorIn(pooledTextEmbeds);
```

The `TimestepEmbedding` function creates sinusoidal features:

```csharp
static Tensor TimestepEmbedding(Tensor timesteps, int dim)
{
    // Half the dimensions for sin, half for cos
    int halfDim = dim / 2;
    var freqs = Exp(-Log(10000f) * Arange(0, halfDim) / halfDim);
    var args = timesteps.Unsqueeze(-1) * freqs.Unsqueeze(0);
    return Concat(Cos(args), Sin(args), dim: -1);  // [batch, dim]
}
```

#### PAG Hook System — design pass, 2026-08-11 (still NOT implemented; see Tier 2.3 in the extension backlog plan, which deliberately scopes this as its own follow-up project rather than bundling it into the guidance-math-param execution pass that shipped CFG-Rescale/TCFG)

The original sketch below (a generic `IAttentionHook` interface) undersold the actual shape of the work in this codebase. Concrete findings from this session's step-cache work (which touched every DiT block loop in `Sd3Transformer`/`ChromaTransformer`/`FluxTransformer`/`HiDreamTransformer`):

- **Fork point.** Self-attention in every one of these four architectures is one call site per block class: `backend.ScaledDotProductAttention(output, q, k, v, attnBias, scale, ...)`. PAG's `PSA(Q,K,V) = V` collapses to: for a perturbed block, skip that call and copy `V` into `output` instead (a single `backend.Copy`/`Scale(v, 1.0f)`-shaped op — cheaper than the attention call it replaces, not more expensive). No new backend primitive is needed for the perturbation itself.
- **Selection mechanism.** Not named layers (no `input_blocks.14.1`-style path exists in a DiT). The natural C# shape is a block-index set threaded through `Forward`, the same way `DeviceFeatureCache? stepCache` was threaded through this session — e.g. `IReadOnlySet<int>? perturbBlocks`, checked once per block-loop iteration (`if (perturbBlocks?.Contains(i) == true) { /* identity */ } else { /* normal SDPA */ }`). This composes cleanly with the existing loop shape; it does NOT compose for free with step-cache on the same run — a perturbed pass changes block *i*'s output relative to the cached anchor, so a generation using both PAG and step-cache needs the perturbed pass to skip the cache entirely (pass `stepCache: null` on that forward call), not share one.
- **Which pipeline first.** Follow the step-cache rollout's own lesson: start with the architecture that has the fewest structural complications, not the most requested one. SD3 (`Sd3Transformer`) is again the cleanest candidate — single dual-stream loop, no double→single transition (unlike Flux/Chroma/HiDream) and no F16-cast-back subtlety (unlike Chroma). Prove the block-index-set + identity-passthrough pattern there before touching the double→single architectures, exactly as the step-cache work sequenced SD3 before HiDream.
- **Blending back.** The 3-pass combine (`eps_pag = eps_cfg + pag_scale * (eps_cond - eps_perturbed)`, §2.5 above) is pure post-hoc tensor math on the three forward passes' outputs — this part reuses `CfgHelper`'s existing shape directly (a new `ApplyPag(Tensor cfgCombined, Tensor cond, Tensor perturbed, float pagScale)` method, same signature style as `ApplyDualCfg`). This half is NOT the hard part of this project; the hard part is the third forward pass's block-selective perturbation above.
- **Not scoped here:** which block index(es) are "the mid block equivalent" per architecture (SDXL's UNet has a literal middle block; a DiT's `Depth`-deep uniform loop doesn't have one structurally distinguished layer) — this needs its own empirical sweep once the mechanism exists, the same way step-cache needed a per-architecture threshold sweep. Don't guess a default before that data exists.

Original C# sketch (kept for the interface-shape idea, but see the concrete fork point above — a real implementation patches the existing per-block SDPA call, it doesn't wrap it in a new interface):

```csharp
public interface IAttentionHook
{
    /// <summary>
    /// Called instead of normal self-attention when PAG is active for this layer.
    /// Returns V directly (identity attention: I * V = V).
    /// </summary>
    Tensor PerturbedAttention(Tensor query, Tensor key, Tensor value);
}
```

The hook should be applied selectively to only the perturbed portion of the batch.

#### Memory Optimization Notes

1. **Sequential CFG** (for low-VRAM): Run unconditional and conditional passes separately, keeping only one set of activations in memory at a time. This halves activation VRAM at the cost of ~10% slower inference (loss of batched parallelism).

2. **CFG on CPU, model on GPU**: For hybrid setups, the CFG combination itself can run on CPU since it's just element-wise ops. The bandwidth cost of copying noise predictions to CPU is negligible.

3. **Skip CFG at later steps**: Some implementations skip the unconditional pass in later denoising steps (when the image is mostly formed) as an optimization. This is model-dependent and may affect quality.

#### Numerical Precision

- The CFG formula itself is numerically stable (simple add/multiply)
- CFG Rescale requires computing standard deviation, which should use the numerically stable two-pass or Welford algorithm for large tensors
- The std division `std_text / std_cfg` can produce infinity if `std_cfg` is near zero (very unlikely in practice, but clamp `std_cfg` to a minimum of 1e-8)
- Sinusoidal embeddings for Flux guidance should be computed in float32 minimum (the exp and trig functions lose precision in float16)

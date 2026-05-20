# Stable Audio Open — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: SharpInference.Audio (Stable Audio pipeline)

## Summary

Stable Audio is Stability AI's family of latent-diffusion text-to-audio / text-to-music models. The pipeline is uniformly three blocks: (1) a **stereo waveform VAE** ("Oobleck VAE") that compresses 44.1 kHz stereo audio by a factor of 2048× into a 64-channel latent at ~21.5 Hz, (2) a **text conditioner** (T5-base for the open models; CLAP for the original closed Stable Audio 1.0/2.0), and (3) a **Diffusion Transformer (DiT)** that denoises the 1D latent sequence with cross-attention to the text tokens and adaLN-modulated global conditioning carrying both the diffusion timestep and two unique-to-this-family scalars: `seconds_start` and `seconds_total`. Those two timing scalars are encoded with Fourier features and let the user generate variable-length audio (0–47 s for Open 1.0, ~11.9 s for Open Small) while preserving timeline coherence.

There are three released open-weight variants relevant to SharpInference: **Stable Audio Open 1.0** (1.06B-param DiT, dpmpp-3m-sde, 100 steps, classical CFG ~ 7, ~47 s output, June 2024); **Stable Audio Open Small** (0.34B DiT, ARC-adversarial post-trained, ping-pong sampler, 8 steps, no CFG, ~11.89 s output, May 2025); and the **stable-audio-tools 2.0 DiT config** (1.06B+, 1536-dim, CLAP text, used internally for "Stable Audio 2.0" but **not released as weights** — Stable Audio 2.0 / 2.5 are API-only). All three share the **same Oobleck VAE** architecture (different checkpoints in the 2.0 line are not public). Open 1.0 and Open Small share the **same VAE weights**. Stable Audio Open uses **EDM-style v-prediction with cosine-shaped sigma schedule** for 1.0; Open Small was **re-trained as a rectified flow** then ARC-distilled. Implementation in pure C# is feasible: the VAE is a 1D snake-activated weight-norm Conv stack (no GroupNorm, ~156M params), the DiT is a standard Stable-Diffusion-3-style DiT with cross-attention + global adaLN (a Flux/SD3 block is ~80% reusable), and T5-base already exists in `SharpInference.Diffusion`.

---

## Detailed Findings

### 1. Variants

| Model | Release | HF path | DiT params | Total params | Audio | Sample rate | License | Approx size |
|---|---|---|---|---|---|---|---|---|
| Stable Audio (closed 1.0) | Sep 2023 | n/a (API only) | ~907M U-Net | ~1.21B | 95 s stereo | 44.1 kHz | Stability commercial | API only |
| Stable Audio 2.0 (closed) | Apr 2024 | n/a (API only) | ~1.06B DiT (config `stable_audio_2_0.json`) | ~1.32B | up to 3 min (≈190 s) stereo | 44.1 kHz | Stability commercial | API only |
| **Stable Audio Open 1.0** | Jun 2024 | `stabilityai/stable-audio-open-1.0` | **1.06B** DiT | 1.32B (156M VAE + 109M T5-base + 1.06B DiT) | **up to 47 s** stereo (1024 latent tokens) | 44.1 kHz | Stability AI Community License (free non-commercial; commercial requires paid license) | **~4.85 GB** safetensors (FP32 full checkpoint) |
| **Stable Audio Open Small** | May 2025 | `stabilityai/stable-audio-open-small` | **0.34B** DiT | ~0.6B (156M VAE + 109M T5-base + 341M DiT) | up to **11.89 s** stereo (256 latent tokens at 21.5 Hz) | 44.1 kHz | Stability AI Community License | ~1.6 GB safetensors |
| Stable Audio 2.5 | Oct 2025 | n/a (API/enterprise) | unknown | unknown | "minutes" of stereo music, faster than 2.0 | 44.1 kHz | enterprise | API only |

**Open models** are the only ones whose weights ship publicly. SAO 1.0's `model.safetensors` is 4.85 GB on the Hub (FP32). Open Small is roughly 1/3 of that. Both ship with a `model_config.json` that mirrors the `stable_audio_tools` config schema and a `model.safetensors` containing all three components keyed under `pretransform.`, `conditioner.conditioners.prompt.`, `diffusion.`. Hugging Face Diffusers also publishes `stabilityai/stable-audio-open-1.0` in a multi-folder layout (`vae/`, `text_encoder/`, `transformer/`, `scheduler/`, `projection_model/`) for the `StableAudioPipeline`.

### 2. Architecture — Component by Component

#### 2.1 Conditioning Encoder (Text)

- **Open 1.0, Open Small:** `google-t5/t5-base` (T5-1.0, 109M parameters, 12 encoder layers, 768 hidden, 12 heads, 64 head dim, SentencePiece tokenizer with `spiece.model`, `model_max_length` = 512). Hidden states taken from `last_hidden_state` (NOT a penultimate layer). The Diffusers reference pipeline pads to `tokenizer.model_max_length`; the original stable-audio-tools `T5Conditioner` defaults to `max_length=128`. Open Small was trained with a tighter context window (the model card example uses prompts ≤ 32 tokens).
- **Open 1.0 negative prompt:** the Diffusers pipeline supports `negative_prompt` for CFG, encoded the same way.
- **Closed 2.0:** `clap_text` with `audio_model_type=HTSAT-base`, `enable_fusion=true`, `use_text_features=true`, `feature_layer_ix=-2` (penultimate text-projection-head feature). `cond_dim=768` matches T5-base's hidden size, so the swap from CLAP→T5 in the Open release did not require changing the DiT's `cond_token_dim`.

The text encoder output is projected into the DiT via a small `projection_model` (linear T5-hidden → `cond_token_dim=768`; identity in Open since hidden sizes already match). The projection_model also produces the timing token embeddings (see §2.4).

#### 2.2 Audio VAE ("Oobleck VAE")

Single VAE used across all Stable Audio variants (the 2.0 and the Open releases share the same architecture; weights differ).

**Verbatim config** (`stable_audio_tools/configs/model_configs/autoencoders/stable_audio_2_0_vae.json`):

```json
{
    "model_type": "autoencoder",
    "sample_size": 65536,
    "sample_rate": 44100,
    "audio_channels": 2,
    "model": {
        "encoder": {
            "type": "oobleck",
            "config": {
                "in_channels": 2, "channels": 128,
                "c_mults": [1, 2, 4, 8, 16],
                "strides": [2, 4, 4, 8, 8],
                "latent_dim": 128, "use_snake": true
            }
        },
        "decoder": {
            "type": "oobleck",
            "config": {
                "out_channels": 2, "channels": 128,
                "c_mults": [1, 2, 4, 8, 16],
                "strides": [2, 4, 4, 8, 8],
                "latent_dim": 64, "use_snake": true,
                "final_tanh": false
            }
        },
        "bottleneck": { "type": "vae" },
        "latent_dim": 64,
        "downsampling_ratio": 2048,
        "io_channels": 2
    }
}
```

Key derived numbers:
- **Encoder pre-bottleneck dim:** 128 (channels × c_mults[-1] = 128 × 16 = 2048 internal at deepest stage, then 1×1 conv to `latent_dim=128` = 2× the bottleneck's 64-channel latent because the VAE outputs `mean` and `logvar` concatenated).
- **Latent dim:** **64 channels** (sampled from N(mean, exp(logvar)) at the bottleneck).
- **Downsampling ratio:** 2 × 4 × 4 × 8 × 8 = **2048 samples per latent step**.
- **Latent sample rate:** 44100 / 2048 ≈ **21.533 Hz**. 47 s of audio → ⌊47 × 21.533⌋ = **1012 latent steps**, rounded up by the DiT to **1024 latent tokens** (which corresponds to exactly 2048 × 1024 = 2 097 152 audio samples ≈ **47.55 s**).
- **VAE training context (sample_size=65536):** the VAE was fine-tuned on **65536-sample crops ≈ 1.486 s** at a time; the architecture is fully convolutional so any length divisible by 2048 is valid at inference.
- **Parameter count:** ≈ 156M total (encoder ~78M, decoder ~78M).
- **Reference-implementation parity:** the diffusers port at `diffusers/models/autoencoders/autoencoder_oobleck.py` uses identical names: `encoder_hidden_size=128`, `downsampling_ratios=[2,4,4,8,8]`, `channel_multiples=[1,2,4,8,16]`, `decoder_channels=128`, `decoder_input_channels=64`, `audio_channels=2`, `sampling_rate=44100`. `pipe.vae.hop_length = prod(downsampling_ratios) = 2048`.

**EncoderBlock structure (verbatim from `stable_audio_tools/models/autoencoders.py`):**
- Stem: `WNConv1d(in_channels=2, out_channels=128, kernel_size=7, padding=3)` (weight_norm Conv1d). No bias quirks.
- 5 × EncoderBlock, where block `i` does:
  1. 3 × `ResidualUnit(c_in)` with dilations `[1, 3, 9]` (each ResidualUnit is `[Snake1d → WNConv1d(k=7,dil=d,padding=3·d) → Snake1d → WNConv1d(k=1)]` with residual add).
  2. Downsample: `Snake1d → WNConv1d(in=c_mults[i]*128, out=c_mults[i+1]*128, kernel_size=2*stride, stride=stride, padding=ceil(stride/2))`.
- Final head: `Snake1d → WNConv1d(in=2048, out=2*64=128, kernel_size=3, padding=1)` (the ×2 produces mean + logvar; `latent_dim` in the encoder config = 128 = 2 × bottleneck 64).

**DecoderBlock structure (mirror of encoder):**
- Stem: `WNConv1d(in=64, out=2048, kernel_size=7, padding=3)`.
- 5 × DecoderBlock in reverse:
  1. Upsample: `Snake1d → WNConvTranspose1d(in=c_mults[i+1]*128, out=c_mults[i]*128, kernel_size=2*stride, stride=stride, padding=ceil(stride/2))`.
  2. 3 × `ResidualUnit` with dilations `[1, 3, 9]`.
- Final head: `Snake1d → WNConv1d(in=128, out=2, kernel_size=7, padding=3)`. **No final tanh.**

**Snake1d activation (exact formula, learnable per-channel `alpha`):**
```
snake(x) = x + (1 / (alpha + 1e-9)) * sin(alpha * x)^2
```
The diffusers port writes this as a single line:
```
hidden_states = hidden_states + (beta + 1e-9).reciprocal() * torch.sin(alpha * hidden_states).pow(2)
```
where `alpha` is `nn.Parameter(torch.ones(1, channels, 1))`. In `stable-audio-tools` it can also be `SnakeBeta` (two learnable params α, β); the released VAEs use the single-α form. **No GroupNorm, no BatchNorm anywhere in the VAE** — only weight normalization on the Conv1d layers.

**Receptive field for chunked decoding:** the paper notes 16 latents on each side as the safe overlap when chunk-decoding for memory.

#### 2.3 Diffusion Transformer (DiT)

**Verbatim DiT config** for the Stable Audio 2.0 line (`stable_audio_tools/configs/model_configs/txt2audio/stable_audio_2_0.json`):

```json
"diffusion": {
    "cross_attention_cond_ids": ["prompt", "seconds_start", "seconds_total"],
    "global_cond_ids": ["seconds_start", "seconds_total"],
    "type": "dit",
    "config": {
        "io_channels": 64,
        "embed_dim": 1536,
        "depth": 24,
        "num_heads": 24,
        "cond_token_dim": 768,
        "global_cond_dim": 1536,
        "project_cond_tokens": false,
        "transformer_type": "continuous_transformer"
    }
}
```

Stable Audio **Open 1.0** uses the same DiT architecture (`embed_dim=1536, depth=24, num_heads=24`) but with `cond_token_dim=768` driven by T5-base instead of CLAP. Together with two timing tokens (see §2.4), the cross-attention key/value sequence is `(T5_tokens + 2) × 768`.

Derived numbers:
- **Head dim:** `embed_dim / num_heads = 1536 / 24 = 64`.
- **MLP ratio:** **4×** (the `ContinuousTransformer.FeedForward` defaults to `mult=4` with SwiGLU; SiLU-gated gated-linear-unit, so `inner_dim = 4 * 1536 = 6144` and the gating splits to 6144 × 2).
- **Per-layer params (rough):** self-attn (4 × 1536²), cross-attn (1536 × 768 + 1536 × 768 + 1536 × 1536 + 1536 × 1536), SwiGLU FFN (~3 × 1536 × 6144 due to gate+up+down), AdaLN modulation linear (1536 × (1536 × 6)). Sums to roughly **44M per block × 24 = 1.06B** matching the paper.
- **Norm:** **bias-less LayerNorm** ("bias-less layernorm has been shown to be more stable") on Q/K projections optional; defaults to off in 1.0, on in Open Small ("we add QK-LayerNorm").
- **RoPE:** **Yes**, applied to `dim_heads // 2 = 32` channels per head via `RotaryEmbedding(max(dim_heads // 2, 32))` along the latent token axis (1024-long). NOT applied to the prepended timing/global tokens — only to the latent stream.
- **Attention type:** **full global** O(N²) self-attention over the 1024 latent tokens. No sliding window, no block-sparse. FlashAttention is optional. Cross-attention sequence length = T5_tokens (≤128) + 2 timing tokens. Note: every block has both self-attention AND cross-attention in Open 1.0; `final_cross_attn_ix` is `-1` (default), so cross-attn runs in **every block**.
- **I/O channels:** **64** (matches VAE latent_dim). Patchify with `patch_size=1` (no patching; each latent step is one token).
- **Prepend global tokens:** Following Diffusers' `StableAudioDiTModel`, the model **prepends 1 global token** (timestep + timing fused) to the sequence: `hidden_states = cat([global_hidden_states, latent_tokens], dim=-2)`. The original `stable_audio_tools` instead injects via `global_cond` to adaLN; both implementations are functionally equivalent for inference but the safetensors are saved against the Diffusers layout for `StableAudioPipeline`.
- **AdaLN modulation:** **6 vectors per block** (scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff). Computed by `Linear(global_cond_dim=1536, 1536) → SiLU → Linear(1536, 6*1536)` then `.chunk(6, dim=-1)`. A learnable zero-init `to_scale_shift_gate` parameter is added to the linear output before chunking (zero-init means each block starts as identity).

**Stable Audio Open Small DiT (verbatim from paper §3.1):**
- `embed_dim = 1024` (down from 1536)
- `depth = 16` (down from 24)
- `num_heads = 16` (head_dim still 64)
- **QK-LayerNorm enabled** (training-stability addition)
- **`seconds_start` removed** — only `seconds_total` is conditioned (the model is duration-controlled but always starts at 0)
- **341M parameters** total in the DiT (vs 1.06B in Open 1.0)
- DiT is `torch.compile`-d for inference

#### 2.4 Timing Conditioning (Stable Audio's signature feature)

Stable Audio is unique in conditioning on **two timestamp scalars** alongside the text prompt:
- `seconds_start` — where in the original recording this clip begins (lets the model handle "cropped from the middle of a song" semantics during training, and at inference defaults to 0).
- `seconds_total` — total **duration of the output clip in seconds** (0–512 range in the config; effectively capped by the trained latent length).

Both are encoded by `NumberConditioner` → `NumberEmbedder` from `stable_audio_tools/models/adp.py`:
1. Clamp to `[min_val=0, max_val=512]`, **min-max normalize to `[0, 1]`**.
2. `NumberEmbedder` applies **Fourier features** (sinusoidal random-projection embedding to dimension `output_dim = cond_token_dim = 768`).
3. The 768-dim embedding is unsqueezed to one **token of length 1**: `(B, 1, 768)`.
4. **Two of these tokens** (one for `seconds_start`, one for `seconds_total`) are **concatenated to the T5 cross-attention sequence**: final cross-attn KV is `cat([T5_tokens(B,L,768), start_token(B,1,768), end_token(B,1,768)], dim=1)` → `(B, L+2, 768)`.
5. **They are also fed into the global conditioning path:** the same Fourier-embedded vectors are concatenated and projected by `Linear(2*768, global_cond_dim=1536)`, summed with the timestep embedding, and fed into the per-block AdaLN modulation linears.

So timing influences the DiT **twice** — as cross-attention tokens AND as adaLN scale/shift/gate biases. In Open Small `seconds_start` is dropped; only `seconds_total` remains in both pathways (so 1 token, not 2).

**Timestep embedding** (the diffusion-time scalar): `FourierFeatures(1, timestep_features_dim=256)` → `Linear(256, 1536) → SiLU → Linear(1536, 1536)` → added to the global timing embedding. Diffusers' `time_proj_dim = 256` matches.

#### 2.5 Hugging Face Diffusers reference layout

`StableAudioDiTModel` (reference Diffusers port) ships with these exact defaults that an implementer should mirror:

| Param | Value |
|---|---|
| `sample_size` | 1024 (latent tokens for SAO 1.0; 256 for Small) |
| `in_channels` | 64 |
| `num_layers` | 24 (16 for Small) |
| `attention_head_dim` | 64 |
| `num_attention_heads` | 24 (16 for Small) |
| `num_key_value_attention_heads` | **12** (cross-attn uses fewer KV heads = MQA-ish; Q is 24 heads but K/V is 12 heads — split to halve cross-attn KV memory) |
| `out_channels` | 64 |
| `cross_attention_dim` | 768 |
| `time_proj_dim` | 256 |
| `global_states_input_dim` | 1536 |
| `cross_attention_input_dim` | 768 |

The 12-KV-head cross-attention is a subtle but important detail — pure-C# implementation must repeat-interleave KV heads to match Q's 24 heads.

---

### 3. Flow Matching / Sampler

**Stable Audio Open 1.0** is **NOT a rectified flow** despite the family using flow matching elsewhere. It is an **EDM-style v-prediction diffusion** model.

- Sampler default: **`dpmpp-3m-sde`** (DPM-Solver++ 3rd-order SDE multistep), `sigma_min=0.3`, `sigma_max=500`, **100 steps**, `cfg_scale=7`.
- Sigma schedule: **poly-exponential** (`K.sampling.get_sigmas_polyexponential` from `k_diffusion`) — schedule defaults `sigma_min=0.01, sigma_max=100` are overridden by the SAO pipeline's `0.3 / 500`.
- The Diffusers port uses `EDMDPMSolverMultistepScheduler` instead (`num_inference_steps=100`, `guidance_scale=7.0`). Output is functionally equivalent.
- Prediction target: **v-prediction** (`x_pred = c_skip * x_t + c_out * model(c_in * x_t, c_noise(sigma), cond)`, EDM preconditioning).
- CFG: applied between conditional (text+timing) and unconditional (empty prompt + same timing) forward passes; **timing is kept in the negative branch** so that the model still respects duration.

**Stable Audio Open Small** is **rectified flow** for the base model, then **ARC-adversarially post-trained** (no distillation):
- Sampler default: **`pingpong`** ("ping-pong sampling" alternating denoise/renoise), **8 steps**, **`cfg_scale=1.0`** (i.e. **CFG disabled** — the ARC contrastive loss removes the need for it, and saves half the VRAM at inference).
- Base pre-training used the **shifted-logit-normal** noise distribution from Stable Diffusion 3 for `p_disc`, and `p_gen ∈ U(log-SNR=-6..2)`. Velocity prediction `v = ε - x_0`. ODE: `x_t = (1-t) x_0 + t ε`, solve `dx_t = -v_θ(x_t, t, c) dt` from `t=1` to `t=0`.
- After ARC, the generator is reparameterized as `G_φ(x_t, t, c) = x_t - t · v_φ(x_t, t, c)` (predicts clean `x_0` indirectly). Ping-pong: `x_{τ_{i-1}} = (1 - τ_{i-1}) x̂_0 + τ_{i-1} ε`.

**Forward link:** see `docs/Research/FLOW_MATCHING_AUDIO.md` for the audio-specific rectified-flow math and noise-schedule details that apply to the Open Small base model and (per Stability's marketing) to Stable Audio 2.5.

---

### 4. Open 2 vs Open 1 Changes

There is **no model named "Stable Audio Open 2.0" on Hugging Face**. The "Open 2" line in this family is:
- **Stable Audio Open Small** (May 2025, ARC paper arXiv:2505.08175) — the **first Open-released model trained with rectified flow**, with adversarial post-training to drop steps from 100 → 8.
- **Stable Audio 2.5** (Oct 2025, closed/API) — Stability's commercial upgrade focused on professional sound production; not open-weight.

Differences from Open 1.0:

| Aspect | Open 1.0 (Jun 2024) | Open Small (May 2025) |
|---|---|---|
| Training objective | v-prediction (EDM) | rectified flow + ARC adversarial post-training |
| DiT embed_dim | 1536 | 1024 |
| DiT depth | 24 | 16 |
| DiT heads | 24 (head_dim 64) | 16 (head_dim 64) |
| DiT params | 1.06B | 0.34B (341M) |
| Total params | 1.32B | ~606M |
| QK-LayerNorm | No | **Yes** |
| `seconds_start` cond | Yes | **Removed** (always 0) |
| `seconds_total` cond | Yes (Fourier features) | Yes (Fourier features) |
| Max duration | 47 s (1024 latent tokens) | **11.89 s** (256 latent tokens) |
| Sampler | dpmpp-3m-sde | **pingpong** |
| Steps | 100 | **8** |
| CFG | 7.0 | **1.0 (disabled)** |
| Negative prompt | Supported | Not used |
| VAE | Oobleck (shared) | **Same Oobleck VAE** (no upgrade) |
| Text encoder | T5-base | T5-base (same) |
| Latency on H100 | ~7 s for 47 s output | ~75 ms for ~12 s output |
| Edge device | n/a | ~7 s on smartphone CPU (FP16, no CFG) |
| New conditioning modes | none | none |
| Audio-to-audio / inpainting | Not in stock pipeline | Not in stock pipeline |
| Stem-aware generation | No | No |
| Style transfer | No | No |

**Audio-to-audio, inpainting, stem generation, and style transfer are advertised for the closed Stable Audio 2.0/2.5 API only.** They are not present in any open-weight model. The codebase contains a `diffusion_cond_inpaint` model type, but no public Stable Audio Open inpainting checkpoint exists.

---

### 5. Inference Pipeline Pseudocode

End-to-end for **text → 30 s of 44.1 kHz stereo WAV** with Stable Audio Open 1.0:

```text
INPUT:
  prompt        : str
  seconds_total : int = 30
  steps         : int = 100
  cfg_scale     : float = 7.0
  negative_prompt : str = ""   // optional
  generator     : RNG

STEP 1 — Tokenize and encode text
  tokens, mask = T5Tokenizer(prompt, max_length=128, padding="max_length", truncation=True)
  // tokens.shape = (1, 128) int64, mask.shape = (1, 128) bool
  text_hidden  = T5Encoder(tokens, attention_mask=mask).last_hidden_state
  // text_hidden.shape = (1, 128, 768) float16
  text_hidden  = ProjectionModel.text_proj(text_hidden)
  // (1, 128, 768) — identity in Open since T5-base is already 768

  // Same for negative prompt (or zeros if empty)
  neg_hidden   = T5Encoder(neg_tokens, neg_mask).last_hidden_state
                  |> ProjectionModel.text_proj
  // (1, 128, 768)

STEP 2 — Encode timing
  seconds_start_norm  = (0 - 0)/(512 - 0) = 0.0
  seconds_total_norm  = (30 - 0)/(512 - 0) = 0.0586
  start_tok = FourierFeatures(seconds_start_norm)   // (1, 1, 768)
  total_tok = FourierFeatures(seconds_total_norm)   // (1, 1, 768)
  cross_kv_cond = concat([text_hidden, start_tok, total_tok], dim=1)
  // (1, 130, 768)
  cross_kv_uncond = concat([neg_hidden, start_tok, total_tok], dim=1)
  // (1, 130, 768)   <-- timing kept in uncond branch
  global_timing = Linear_1536(concat([start_tok, total_tok], dim=-1).squeeze(1))
  // (1, 1536)

STEP 3 — Compute latent shape
  latent_seconds = ceil(seconds_total) * latent_rate_hz   // 30 * 21.533 ≈ 646
  // SAO 1.0 trained on fixed 1024 tokens — pad to 1024:
  latent_len = 1024
  audio_samples = latent_len * 2048 = 2097152   // ≈ 47.55 s

STEP 4 — Initialize noise
  x_T = N(0, 1).sample((1, 64, 1024)) * sigma_max(=500)
  // (1, 64, 1024) float16, layout (B, C, T)

STEP 5 — Build EDM sigma schedule
  sigmas = polyexponential(steps=100, sigma_min=0.3, sigma_max=500, rho=7.0)
  // sigmas: tensor length 101, sigmas[0]=500, sigmas[-1]=0.0

STEP 6 — Denoising loop (dpmpp-3m-sde)
  for i in 0..99:
      sigma_t = sigmas[i]
      sigma_next = sigmas[i+1]

      // EDM preconditioning
      c_skip = sigma_data^2 / (sigma_t^2 + sigma_data^2)   // sigma_data = 1.0
      c_out  = sigma_t * sigma_data / sqrt(sigma_t^2 + sigma_data^2)
      c_in   = 1 / sqrt(sigma_t^2 + sigma_data^2)
      c_noise = 0.25 * log(sigma_t)

      // CFG dual forward
      v_cond   = DiT(c_in * x_t, c_noise, cross_kv=cross_kv_cond,   global=global_timing + time_embed)
      v_uncond = DiT(c_in * x_t, c_noise, cross_kv=cross_kv_uncond, global=global_timing + time_embed)
      v        = v_uncond + cfg_scale * (v_cond - v_uncond)

      // Convert v to x_0 estimate
      x0 = c_skip * x_t + c_out * v

      // DPM++ 3M SDE step from sigma_t to sigma_next
      x_t = dpmpp_3m_sde_step(x_t, x0, sigma_t, sigma_next, history=...)

  // After loop: x_0 ~ (1, 64, 1024)

STEP 7 — VAE decode
  // Optionally chunk-decode for memory: split (1,64,1024) into overlapping
  // chunks of 128 latents with 16-latent overlap on each side, decode, crossfade.
  wav = OobleckDecoder(x_0)
  // (1, 2, 2 097 152) float32 in [-1, 1]

STEP 8 — Trim and write WAV
  wav = wav[:, :, : seconds_total * 44100]   // crop to 30s = 1 323 000 samples
  wav = wav / max(abs(wav))                  // normalize
  WriteWav("out.wav", wav, sample_rate=44100, bit_depth=16, channels=2)
```

**Tensor-shape cheat sheet** (Open 1.0, batch = 1, FP16):

| Stage | Shape | Notes |
|---|---|---|
| T5 input ids | (1, 128) int64 | |
| T5 hidden out | (1, 128, 768) | |
| Timing token | (1, 1, 768) × 2 | |
| Cross-attn KV in | (1, 130, 768) | |
| Global cond (post-Linear) | (1, 1536) | |
| Initial noise latent | (1, 64, 1024) | sigma=500 scaled |
| DiT internal hidden | (1, 1024, 1536) | after BCN → BNC swap |
| Self-attn QKV per head | (1, 24, 1024, 64) | full O(1024²) per layer |
| Cross-attn Q | (1, 24, 1024, 64) | |
| Cross-attn KV | (1, 12, 130, 64) | 12 KV heads (MQA-like) |
| FFN intermediate | (1, 1024, 6144) | SwiGLU |
| Decoder VAE input | (1, 64, 1024) | |
| Decoder VAE output | (1, 2, 2 097 152) | stereo waveform |

---

### 6. Features

| Feature | Open 1.0 | Open Small | Closed 2.0 (API) | Closed 2.5 (API) |
|---|---|---|---|---|
| Text-to-audio | ✓ | ✓ | ✓ | ✓ |
| Variable duration (timing cond) | ✓ (0–47 s) | ✓ (0–11.89 s) | ✓ (up to ~190 s) | ✓ (minutes) |
| Audio-to-audio (init audio + strength) | ✗ in stock pipeline (DIY by VAE-encoding + partial-noise) | ✗ | ✓ | ✓ |
| Negative prompt | ✓ | ✗ (no CFG) | ✓ | ✓ |
| Inpainting / extension | ✗ (`diffusion_cond_inpaint` exists in code but no checkpoint) | ✗ | ✓ (inpaint API) | ✓ |
| Stem-aware generation / stem separation | ✗ | ✗ | ✗ (separate model) | ✓ |
| Style transfer | ✗ | ✗ | ✓ | ✓ |
| Loop / sound-effect | ✓ | ✓ (specialized for SFX/loops) | ✓ | ✓ |
| Vocal generation | Limited (model card warns "cannot generate realistic vocals") | Same warning | ✗ (instrumental focus) | Improved |

Audio-to-audio is implementable on top of Open 1.0 by VAE-encoding the input clip, partially noising it with `sigma = sigma_max * strength`, then resuming the sampler from that step. This is exactly the SD-img2img pattern in 1D and is not exposed in the official inference scripts but is supported by `generate_diffusion_cond(init_audio=..., init_noise_level=...)` in `stable_audio_tools/inference/generation.py`.

---

### 7. Memory and Performance

For Stable Audio Open 1.0, **FP16** inference (Diffusers `torch_dtype=float16`):
- T5-base (FP16): ~218 MB resident weights + small KV during encode.
- VAE encoder + decoder (FP16): ~312 MB weights.
- DiT (FP16): ~2.12 GB weights.
- Activation peak during DiT forward at 1024 tokens, batch 1: ~1.5 GB (self-attn 1024² × 24 heads × 24 layers temporary; FlashAttention drops this to ~300 MB).
- VAE decode peak: ~700 MB unless chunk-decoded.
- **Total: ~5–6 GB VRAM for end-to-end inference at batch 1, FP16, no CFG batching; ~7–8 GB if CFG is run as a 2× batched forward.**

Latency on consumer GPU (single batch, 47 s output, 100 steps, FP16):
- RTX 4090: ~12 s wall-clock (RTF ~ 4×). H100: ~7 s wall-clock per the ARC paper.
- CPU-only: minutes (the project explicitly does not target this).

Stable Audio Open **Small** (8 steps, no CFG, ARC-distilled):
- H100: ~75 ms for ~12 s output (RTF ~ 160×).
- Snapdragon-class CPU: ~7 s for ~12 s output (RTF ~ 1.7×, real-time-ish).
- VRAM: ~1.8 GB FP16 total (single forward branch, no CFG doubling).

---

### 8. Reference Implementations

| Repo | Path | Notes |
|---|---|---|
| Stability-AI/stable-audio-tools | `stable_audio_tools/models/autoencoders.py` (OobleckEncoder, OobleckDecoder, ResidualUnit) | The reference VAE; weight_norm Conv1d + snake. |
| Stability-AI/stable-audio-tools | `stable_audio_tools/models/dit.py` | DiffusionTransformer wrapper around ContinuousTransformer; Fourier timestep features. |
| Stability-AI/stable-audio-tools | `stable_audio_tools/models/transformer.py` | `ContinuousTransformer`, `TransformerBlock`, AdaLN modulation (6-vector chunk), RoPE on `head_dim//2`. |
| Stability-AI/stable-audio-tools | `stable_audio_tools/models/conditioners.py` | `T5Conditioner` (last_hidden_state, default max_length=128), `NumberConditioner` (Fourier features for timing). |
| Stability-AI/stable-audio-tools | `stable_audio_tools/models/adp.py` | `NumberEmbedder` (sinusoidal Fourier-features for the normalized scalar). |
| Stability-AI/stable-audio-tools | `stable_audio_tools/inference/sampling.py` | All samplers; relevant: `dpmpp-3m-sde` for 1.0, `pingpong` for Small, rectified-flow `euler` / `rk4` / `dpmpp`. |
| Stability-AI/stable-audio-tools | `stable_audio_tools/inference/generation.py` | `generate_diffusion_cond` — wraps the loop with CFG and init_audio handling. |
| Stability-AI/stable-audio-tools | `stable_audio_tools/configs/model_configs/autoencoders/stable_audio_2_0_vae.json` | Authoritative VAE config (reproduced verbatim in §2.2). |
| Stability-AI/stable-audio-tools | `stable_audio_tools/configs/model_configs/txt2audio/stable_audio_2_0.json` | Authoritative DiT config (reproduced verbatim in §2.3). |
| Stability-AI/stable-audio-tools | `stable_audio_tools/configs/model_configs/txt2audio/stable_audio_1_0.json` | The closed Stable Audio 1.0 U-Net config (`adp_cfg_1d`, DAC VAE) — included for historical reference; DO NOT confuse with Open 1.0. |
| Stability-AI/stable-audio-tools | `stable_audio_tools/training/arc.py` | ARC adversarial post-training (Open Small). |
| huggingface/diffusers | `src/diffusers/models/transformers/stable_audio_transformer.py` | `StableAudioDiTModel` — clean PyTorch DiT with `num_layers=24, attention_head_dim=64, num_attention_heads=24, num_key_value_attention_heads=12`. Best reference for the **12 KV head** cross-attention. |
| huggingface/diffusers | `src/diffusers/models/autoencoders/autoencoder_oobleck.py` | `AutoencoderOobleck` — explicit Snake1d formula, dilation=[1,3,9] ResidualUnits per block. |
| huggingface/diffusers | `src/diffusers/pipelines/stable_audio/pipeline_stable_audio.py` | `StableAudioPipeline` — T5EncoderModel, ProjectionModel, EDMDPMSolverMultistepScheduler, default steps=100, cfg=7.0. |
| arxiv 2407.14358 | Stable Audio Open paper | Architecture overview, evaluation. |
| arxiv 2505.08175 | "Fast Text-to-Audio Generation with Adversarial Post-Training" | ARC + Open Small architecture deltas (§3.1 has all the numbers). |
| arxiv 2402.04825 | "Fast Timing-Conditioned Latent Audio Diffusion" | Original Stable Audio 1.0 (closed) paper; introduces the timing-conditioning idea. |

---

### 9. Differences Between Implementations

- **stable_audio_tools vs diffusers**: stable_audio_tools injects `global_cond` purely through adaLN modulation; diffusers' `StableAudioDiTModel` instead **prepends a 1-token global state to the latent sequence** AND also uses adaLN. Both yield equivalent forward computations for the same weights (the safetensors layout in `stabilityai/stable-audio-open-1.0` is the diffusers one).
- **Cross-attn KV heads**: stable_audio_tools `ContinuousTransformer` does standard symmetric multi-head attention (Q and KV both 24 heads). Diffusers explicitly splits to `num_key_value_attention_heads=12`. The shipped weights in HF have **separate 12-head KV projections** on cross-attention — implementations must repeat KV heads ×2 to broadcast against the 24 Q heads. Self-attention remains symmetric 24/24.
- **T5 max length**: stable-audio-tools defaults to **128**, diffusers defaults to **`tokenizer.model_max_length` = 512**. The model was trained with 128 — using 512 wastes compute on padding tokens but is numerically identical (mask blocks them).
- **Sigma schedule**: stable-audio-tools uses `K.sampling.get_sigmas_polyexponential(sigma_min=0.3, sigma_max=500, rho=7.0)`; diffusers uses `EDMDPMSolverMultistepScheduler` which generates a slightly different schedule for the same `(sigma_min, sigma_max)`. Both produce qualitatively similar audio; bit-exact matching requires reproducing the exact `get_sigmas_polyexponential` formula.
- **Snake variant**: `stable_audio_tools.models.autoencoders.Snake1d` allows both `Snake` (single learnable α) and `SnakeBeta` (α and β). Released VAE weights use single-α. Diffusers' port also uses single-α but names the parameter `alpha`. **Make sure the C# code uses single-α form**.
- **Closed Stable Audio 1.0** (the older U-Net model, config `stable_audio_1_0.json`) used a **DAC-encoder VAE** with `strides=[4,4,8,8]` and 1024× downsample, **not** the Oobleck VAE. Don't conflate it with **Stable Audio Open 1.0** which uses Oobleck.

---

## Key Numbers / Constants

```text
SAMPLE_RATE              = 44100 Hz
AUDIO_CHANNELS           = 2 (stereo)
VAE_DOWNSAMPLE_RATIO     = 2048   (= 2*4*4*8*8)
VAE_LATENT_CHANNELS      = 64
VAE_LATENT_RATE_HZ       = 44100 / 2048 = 21.5332...
VAE_BASE_CHANNELS        = 128
VAE_C_MULTS              = [1, 2, 4, 8, 16]
VAE_STRIDES              = [2, 4, 4, 8, 8]
VAE_RESBLOCK_DILATIONS   = [1, 3, 9]   // 3 ResidualUnits per stage
VAE_STEM_KERNEL          = 7
VAE_HEAD_OUT_CHANNELS    = 2 * 64 = 128  // mean + logvar
VAE_ENCODER_PARAMS       ≈ 78M
VAE_DECODER_PARAMS       ≈ 78M
VAE_TOTAL_PARAMS         ≈ 156M
SNAKE_EPSILON            = 1e-9

T5_VARIANT               = google-t5/t5-base
T5_HIDDEN_DIM            = 768
T5_PARAMS                = 109M
T5_MAX_LENGTH            = 128 (stable-audio-tools) / 512 (diffusers); train was 128

// Stable Audio Open 1.0 DiT
DIT_EMBED_DIM            = 1536
DIT_DEPTH                = 24
DIT_NUM_HEADS            = 24
DIT_HEAD_DIM             = 64
DIT_KV_HEADS_CROSS_ATTN  = 12
DIT_MLP_RATIO            = 4
DIT_MLP_HIDDEN           = 6144
DIT_IO_CHANNELS          = 64
DIT_COND_TOKEN_DIM       = 768
DIT_GLOBAL_COND_DIM      = 1536
DIT_PARAMS               ≈ 1.06B
DIT_TIME_PROJ_DIM        = 256
ROPE_DIM_PER_HEAD        = 32  // dim_heads // 2
ADALN_VECTORS_PER_BLOCK  = 6   // scale/shift/gate × {self_attn, ffn}
LATENT_LEN_OPEN_1_0      = 1024 tokens (≈ 47.55 s)

// Stable Audio Open Small DiT
SMALL_DIT_EMBED_DIM      = 1024
SMALL_DIT_DEPTH          = 16
SMALL_DIT_NUM_HEADS      = 16
SMALL_DIT_HEAD_DIM       = 64
SMALL_DIT_PARAMS         ≈ 341M
SMALL_QK_LAYERNORM       = true
SMALL_HAS_SECONDS_START  = false (only seconds_total)
SMALL_LATENT_LEN         = 256 tokens (≈ 11.89 s)

// Timing conditioner
TIMING_MIN_VAL           = 0
TIMING_MAX_VAL           = 512  (seconds)
TIMING_FEATURE_TYPE      = Fourier features (sinusoidal random projection)
TIMING_OUTPUT_DIM        = 768  (matches cond_token_dim)

// Sampler defaults — Open 1.0
SAMPLER                  = dpmpp-3m-sde
STEPS                    = 100
CFG_SCALE                = 7.0
SIGMA_MIN                = 0.3
SIGMA_MAX                = 500.0
SIGMA_DATA               = 1.0    (EDM)
RHO                      = 7.0    (polyexponential schedule)

// Sampler defaults — Open Small
SAMPLER                  = pingpong
STEPS                    = 8
CFG_SCALE                = 1.0   (disabled)
PRED_TARGET              = velocity (rectified flow, x_t = (1-t) x_0 + t ε)
PINGPONG_RENOISE_SCHED   = linear (t descending)
```

---

## Data Layouts / Formats

**Waveform tensor:** `(batch, channels=2, samples)` `float32` in `[-1, 1]`. WAV writer should clip and quantize: `int16(clamp(x, -1, 1) * 32767)`.

**VAE latent tensor:** `(batch, channels=64, time)` `float16/float32`. **Time axis is last** in stable-audio-tools convention (`(B, C, T)` like a 1D conv input), but the DiT internally transposes to `(B, T, C)` for attention. Implementers must match the on-disk weight layout: VAE conv layers expect `(B, C, T)`; DiT projections expect `(B, T, C)`.

**Mean/logvar split inside VAE encoder output:** the encoder produces `(B, 128, T)`; split along dim 1 into `mean = enc[:, :64, :]` and `logvar = enc[:, 64:, :]`; sample with `z = mean + exp(0.5 * clamp(logvar, -30, 20)) * eps`. At inference, deterministic mode uses `z = mean` only.

**T5 attention mask:** `(B, L)` bool. Convert to additive: `(1 - mask) * -10000`. Must be applied in cross-attention to prevent the model from attending to PAD tokens (this is critical — the model was trained with masked PADs).

**Timing token shape:** `(B, 1, 768)` per timing scalar. Concatenate to text hidden along dim=1 BEFORE cross-attention. The mask for these tokens is **always 1** (always attend).

**Safetensors layout (Open 1.0, HF Diffusers folder structure):**
```
stable-audio-open-1.0/
  vae/
    diffusion_pytorch_model.safetensors     // OobleckVAE weights
    config.json
  text_encoder/
    model.safetensors                       // T5-base encoder weights
    config.json
  tokenizer/
    spiece.model
    tokenizer_config.json
  projection_model/
    diffusion_pytorch_model.safetensors     // Linear(T5 → cond) + Fourier-feature embedders
    config.json
  transformer/
    diffusion_pytorch_model.safetensors     // StableAudioDiTModel
    config.json
  scheduler/
    scheduler_config.json                   // EDMDPMSolverMultistepScheduler
  model_index.json
```

The stable-audio-tools layout instead packs everything into a single `model.safetensors` (~4.85 GB) plus a `model_config.json`. Loader should accept both.

---

## Algorithm Steps

See §5 for end-to-end pseudocode. Concrete sub-algorithms:

**A. Snake1d forward (per-channel learnable α):**
```text
input  x : (B, C, T)
alpha   : (1, C, 1) learnable
y = x + (1 / (alpha + 1e-9)) * sin(alpha * x)^2
```

**B. Oobleck ResidualUnit forward:**
```text
input x : (B, C, T)
y = WNConv1d(in=C, out=C, kernel=7, dilation=d, padding=3*d)( Snake1d(x) )
y = WNConv1d(in=C, out=C, kernel=1)( Snake1d(y) )
return x + y
```
Three stacked with dilations 1, 3, 9, then downsample (encoder) or upsample (decoder).

**C. Fourier feature embedding for timing scalar `s ∈ [0,1]`:**
```text
// stable-audio-tools NumberEmbedder
// freqs: fixed random projection of shape (output_dim/2,) sampled at init
half = output_dim // 2
emb  = s * freqs * 2*pi
emb  = concat([sin(emb), cos(emb)], dim=-1)   // (output_dim,)
emb  = Linear(output_dim, output_dim)(emb)    // optional learned projection
```
Returns `(B, 1, output_dim)`.

**D. AdaLN modulation per DiT block:**
```text
input cond_global : (B, 1536)
mod = Linear(1536, 1536)( SiLU( Linear(1536, 1536)(cond_global) ) )
mod = mod + to_scale_shift_gate   // (1, 6*1536) learnable, zero-init
scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff = chunk(mod, 6, dim=-1)
// Apply inside block:
//   h = x + gate_self.unsqueeze(1) * SelfAttn( LayerNorm(x) * (1+scale_self.unsqueeze(1)) + shift_self.unsqueeze(1) )
//   h = h + CrossAttn( LayerNorm(h), kv=cross_kv_cond )      // no adaLN on cross-attn norm
//   h = h + gate_ff.unsqueeze(1)   * FFN(      LayerNorm(h) * (1+scale_ff.unsqueeze(1))   + shift_ff.unsqueeze(1) )
```

**E. EDM v→x_0 conversion (Open 1.0):**
```text
c_skip(σ)  = sigma_data^2 / (σ^2 + sigma_data^2)
c_out(σ)   = σ * sigma_data / sqrt(σ^2 + sigma_data^2)
c_in(σ)    = 1 / sqrt(σ^2 + sigma_data^2)
c_noise(σ) = 0.25 * log(σ)
v          = DiT(c_in(σ) * x_σ, c_noise(σ), cond)
x_0_est    = c_skip(σ) * x_σ + c_out(σ) * v
```

**F. Ping-pong step (Open Small):**
```text
for i in range(steps, 0, -1):    // i goes 8,7,...,1
    τ_i      = i / steps          // 1.0 → 1/8
    τ_next   = (i-1) / steps      // 0.0 at end
    x_0_hat  = x_t - τ_i * v_φ(x_t, τ_i, cond)   // generator output
    if τ_next > 0:
        ε    = sample N(0,1) like x_t
        x_t  = (1 - τ_next) * x_0_hat + τ_next * ε
    else:
        x_t  = x_0_hat
return x_t   // == x_0
```

---

## Open Questions

1. **Exact Fourier-feature initialization for `NumberEmbedder`.** Stability's code computes random projections at module init; the trained weights freeze that randomness. Need to confirm the safetensors actually contains the projection matrix (likely `projection_model.start_number_conditioner.embedder.weight` etc.) and whether the diffusers `ProjectionModel` exposes them under different names. **Action:** dump the safetensors index and grep for "embedder" / "fourier".
2. **`final_cross_attn_ix` for Open 1.0.** The config does not set it explicitly; default is `-1` (cross-attn in every block). Need to confirm at weight-load by checking whether `transformer_blocks.{i}.attn2.*` exists for all 24 blocks.
3. **QK-LayerNorm in Open Small** — the paper says "we add QK-LayerNorm" but doesn't specify whether it's per-head RMSNorm or LayerNorm, and whether on Q only or both Q and K. The diffusers `StableAudioDiTModel` for Open 1.0 likely doesn't have these params; for Open Small we may need separate `q_norm` / `k_norm` modules. **Action:** inspect Open Small's safetensors key list once weights are downloaded.
4. **Stable Audio 2.0 / 2.5 architecture.** No open release. The closed 2.0 uses CLAP text encoder per the config, and was extended to ~190 s outputs. 2.5 details are not public; Stability's marketing implies faster generation and "complex" outputs — likely rectified flow + larger DiT, no public confirmation. **For SharpInference, plan only for the open variants.**
5. **Chunk-decode crossfade strategy.** Paper says 16-latent overlap on each side is sufficient. Exact crossfade window (linear? Hann?) is not specified — diffusers' AutoencoderOobleck does not implement chunking. **Action:** mirror stable-audio-tools' `decode_audio` helper.
6. **Mean/logvar split — order in concat.** stable-audio-tools' `VAEBottleneck` does `mean, logvar = enc.chunk(2, dim=1)` (mean first, logvar second). Confirm safetensors layout uses the same ordering when saving.
7. **Inpainting checkpoint.** `diffusion_cond_inpaint` model_type exists in the codebase but no public Stable Audio Open inpaint weights have been released. Skip implementing the inpaint path until weights surface.

---

## Implementation Notes for SharpInference

### Package placement
- New package: **`SharpInference.Audio.StableAudio`** under `src/SharpInference.Audio.StableAudio/`. Depends on `SharpInference.Core`, `SharpInference.Diffusion` (for T5 + scheduler infrastructure + DiT block primitives), `SharpInference.Audio.Vocoders` (the future home of iSTFTNet / Kokoro / ACE-Step VAEs — share the snake activation and weight-norm Conv1d primitives there).

### Component-by-component implementation plan

**1. T5-base encoder — reuse.**
The existing `SharpInference.Diffusion` T5 encoder (used for Flux, SD3, etc.) supports `t5-base` already. Wire the SAO tokenizer to the same SentencePiece path (`spiece.model`). Verify `last_hidden_state` (not penultimate) is returned. Max length should default to 128 to match training.

**2. Oobleck VAE — new code, but share primitives with iSTFTNet / Kokoro / ACE-Step.**

Required new primitives (under `SharpInference.Audio.Layers/`):
- `WeightNormConv1D` — Conv1D where `weight = g * v / ||v||`; at load time can be fused to a single Conv1D weight (`fused_weight = g * v / ||v||_per_outchannel`). Recommend **fusing at load** so the forward path is plain Conv1D.
- `WeightNormConvTranspose1D` — same idea on transposed conv.
- `Snake1D` — per-channel learnable `alpha`, formula `x + sin(alpha*x)^2 / (alpha + 1e-9)`. Already needed by ACE-Step too. Make this a CUDA kernel (PTX) — it's elementwise, trivial.
- `OobleckResidualUnit` — 2× Conv1D with snake between, residual add. Compose dilations [1, 3, 9].
- `OobleckEncoderBlock` / `OobleckDecoderBlock` — composes 3 ResidualUnits + downsample/upsample Conv.
- `OobleckEncoder` / `OobleckDecoder` — stems + 5 blocks.
- `OobleckVAE` — wraps encoder + decoder + KL bottleneck. `Encode(wav) -> (mean, logvar)`, `Decode(z) -> wav`, `Reparameterize(mean, logvar) -> z`.

**Memory note:** at 47s × 44.1 kHz × 2 channels × FP32 = 16.6 MB per waveform — trivial. Latents are 64 × 1024 × 4 = 256 KB. The decoder's intermediate activation peak is at the highest-resolution stage just before output (2 channels × 2.1M samples × 128 internal). Plan **chunk-decode** with 128-latent chunks and 16-latent overlap, crossfaded linearly. This caps activation memory at ~50 MB.

**3. DiT — reuse Flux/SD3 blocks heavily, but build a 1D variant.**

The DiT is structurally **a 1D version of SD3's MM-DiT minus the dual-stream image/text branching** — single stream over latent tokens with cross-attention to a separate text+timing token bank, plus adaLN with 6-vector chunking. Reusable from `SharpInference.Diffusion`:
- **AdaLN modulation** (6-chunk variant from SD3) — direct reuse.
- **RoPE** — reuse Flux's `RotaryEmbedding`, configure for 1D over 1024 positions, applied to first `head_dim//2 = 32` channels per head.
- **SwiGLU FFN** — reuse SD3's `GatedFeedForward` (gate+up+down with SiLU gating, `mult=4`).
- **bias-less LayerNorm** — reuse the SD3 norm.

Build new (under `SharpInference.Audio.StableAudio.Models/`):
- `FourierFeatures1D` — random-projection sinusoidal embedding; load the projection matrix from safetensors. **Make it CUDA kernel** — single matmul + sin + cos.
- `TimestepEmbed` — `FourierFeatures1D(1, 256) → Linear(256, 1536) → SiLU → Linear(1536, 1536)`.
- `NumberConditioner` — clamps to [0,512], normalizes, calls a `FourierFeatures1D(1, 768)` then optional linear; outputs `(B, 1, 768)`.
- `StableAudioDiTBlock` — composes the reused AdaLN + self-attn (with optional QK-LN for Open Small) + cross-attn-with-12-KV-heads + FFN. **The 12-KV-head cross-attention is the only structurally new attention pattern**; either repeat-interleave KV heads at projection time (2× KV memory waste) or write a small CUDA kernel that broadcasts.
- `StableAudioDiT` — stem `Conv1D(64, 1536, k=1)` (effectively a Linear over channels), 24 (or 16) blocks, head `Conv1D(1536, 64, k=1)`. Plus the "prepend global token" trick from diffusers if loading diffusers-format weights, or pure-adaLN if loading stable-audio-tools-format weights.

**4. Sampler — extend `SharpInference.Diffusion.Schedulers`.**

- **dpmpp-3m-sde** (for Open 1.0) — port from k_diffusion. Order-3 multistep DPM-Solver++ with SDE noise injection. Needs `polyexponential` sigma schedule `(sigma_min=0.3, sigma_max=500, rho=7, steps=100)`. This sampler is NOT yet in `SharpInference.Diffusion`. Reuse the existing EDM preconditioning helpers. Cross-link `docs/Research/DIFFUSION_SCHEDULERS.md`.
- **pingpong** (for Open Small) — trivial: denoise to `x_0_hat`, renoise with linear schedule. New code, but tiny (~20 lines).

**5. WAV writer.**
Reuse whatever `SharpInference.Audio.Vocoders` already has (Kokoro / iSTFTNet need it). Standard 16-bit PCM WAV, sample_rate=44100, channels=2, little-endian. **No** float WAV mode by default (writers like Audacity expect int16 for compatibility).

### Loader strategy

Two safetensors formats coexist:
- **stable-audio-tools format:** single `model.safetensors` keyed `pretransform.encoder.*`, `pretransform.decoder.*`, `conditioner.conditioners.prompt.model.*`, `conditioner.conditioners.seconds_start.embedder.*`, `model.*` (DiT). Plus a sibling `model_config.json`.
- **diffusers format:** multi-folder with separate safetensors per subfolder.

Recommend a **single loader** that detects layout by file presence, maps both to a common in-memory dictionary, then constructs the components. The HF-Hub bound user will more often grab the diffusers layout; researchers will grab the stable-audio-tools single-file. Both need to work.

### Validation tolerances vs Python reference

- VAE encode/decode: per-sample MSE < 1e-4 at FP32, < 1e-2 at FP16.
- T5 hidden state: per-token cosine similarity > 0.999 at FP16.
- DiT forward: per-element cosine > 0.99 at FP16 over the (1, 64, 1024) output. The sin in Snake and the SwiGLU make exact bit-equality unrealistic.
- End-to-end audio: FAD (Fréchet Audio Distance) < 0.5 vs reference on 100-prompt eval set, **or** PEAQ ODG > -1.0. For unit tests, use fixed seed + sampler and compare waveform STFT cosine > 0.95.

### Risks and TBDs

- **The 12-KV-head cross-attention** is unusual; misimplementing it as standard 24-head MHA loads the wrong weight shapes and produces silent garbage. Validate by printing the cross-attn KV projection weight shape on first load: expect `(12*64, 768) = (768, 768)` not `(24*64, 768) = (1536, 768)`. Note both happen to be 768×768 because 12×64=768 — so the **shape alone won't catch it**. Must check by symbol of behavior: 12 vs 24 heads in the attention reshape.
- **QK-LayerNorm for Open Small** — until weights are inspected, treat as an unknown sub-component; the load path must handle missing `q_norm`/`k_norm` weights gracefully (Open 1.0 has none).
- **`seconds_start` for Open Small** — code paths must allow that conditioner to be absent. Cleanest is to drive the model via a `ConditioningConfig` record that lists which scalars exist.
- **Mono input handling** — model is stereo-only. For mono input audio (audio-to-audio), duplicate to L+R channels. For mono output preference, simple post-mix `0.5*(L+R)`.
- **Pure-C# CUDA Snake1d kernel**: the `sin` is expensive. A fused `snake_act` PTX kernel that does `x + __sinf(a*x)^2 / (a + eps)` in one pass is essential — naive impl would be 3 kernel launches. Use `__fmaf_rn` for the final add. Snake appears in every Oobleck Conv layer, so this is hot.

### Cross-references

- VAE primitives: see `docs/Research/VAE_ARCHITECTURE.md` (image VAEs — different but shares concepts), `docs/Research/HIFIGAN_VOCODER.md` (Conv1D + weight-norm patterns), `docs/Research/KOKORO_ARCHITECTURE.md` (existing Conv1D vocoder in the codebase).
- DiT primitives: `docs/Research/SD3_ARCHITECTURE.md` (AdaLN-Single + SwiGLU), `docs/Research/FLUX_ARCHITECTURE.md` (RoPE configuration).
- Schedulers: `docs/Research/DIFFUSION_SCHEDULERS.md` (EDM preconditioning; dpmpp-3m-sde needs to be added).
- Flow matching for the rectified-flow Open Small path: see `docs/Research/FLOW_MATCHING_AUDIO.md` (companion document being built in parallel).
- Text encoders: `docs/Research/T5_MEMORY_STRATEGY.md`, `docs/Research/TEXT_ENCODERS.md`, `docs/Research/TOKENIZERS.md`.

# Stable Audio Open — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: HartsyInference.Audio (Stable Audio pipeline)

> **Stub.** The narrative walkthrough, restated pseudocode and resolved open questions were
> removed on 2026-08-06 — this model is built and verified, so the C# is the source of truth for
> *how it works*. What remains is what the code cannot tell you: upstream provenance, reference
> constants to diff a suspect port against, where implementations disagree, and bring-up traps.
> Full history is in git. Parity evidence: `docs/Checklists/PARITY_VERIFICATION.md`.

## Summary

Stable Audio is Stability AI's family of latent-diffusion text-to-audio / text-to-music models. The pipeline is uniformly three blocks: (1) a **stereo waveform VAE** ("Oobleck VAE") that compresses 44.1 kHz stereo audio by a factor of 2048× into a 64-channel latent at ~21.5 Hz, (2) a **text conditioner** (T5-base for the open models; CLAP for the original closed Stable Audio 1.0/2.0), and (3) a **Diffusion Transformer (DiT)** that denoises the 1D latent sequence with cross-attention to the text tokens and adaLN-modulated global conditioning carrying both the diffusion timestep and two unique-to-this-family scalars: `seconds_start` and `seconds_total`. Those two timing scalars are encoded with Fourier features and let the user generate variable-length audio (0–47 s for Open 1.0, ~11.9 s for Open Small) while preserving timeline coherence.

There are three released open-weight variants relevant to HartsyInference: **Stable Audio Open 1.0** (1.06B-param DiT, dpmpp-3m-sde, 100 steps, classical CFG ~ 7, ~47 s output, June 2024); **Stable Audio Open Small** (0.34B DiT, ARC-adversarial post-trained, ping-pong sampler, 8 steps, no CFG, ~11.89 s output, May 2025); and the **stable-audio-tools 2.0 DiT config** (1.06B+, 1536-dim, CLAP text, used internally for "Stable Audio 2.0" but **not released as weights** — Stable Audio 2.0 / 2.5 are API-only). All three share the **same Oobleck VAE** architecture (different checkpoints in the 2.0 line are not public). Open 1.0 and Open Small share the **same VAE weights**. Stable Audio Open uses **EDM-style v-prediction with cosine-shaped sigma schedule** for 1.0; Open Small was **re-trained as a rectified flow** then ARC-distilled. Implementation in pure C# is feasible: the VAE is a 1D snake-activated weight-norm Conv stack (no GroupNorm, ~156M params), the DiT is a standard Stable-Diffusion-3-style DiT with cross-attention + global adaLN (a Flux/SD3 block is ~80% reusable), and T5-base already exists in `HartsyInference.Diffusion`.

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

## Implementation Notes for HartsyInference

### Package placement
- New package: **`HartsyInference.Audio.StableAudio`** under `src/HartsyInference.Audio.StableAudio/`. Depends on `HartsyInference.Core`, `HartsyInference.Diffusion` (for T5 + scheduler infrastructure + DiT block primitives), `HartsyInference.Audio.Vocoders` (the future home of iSTFTNet / Kokoro / ACE-Step VAEs — share the snake activation and weight-norm Conv1d primitives there).

### Component-by-component implementation plan

**1. T5-base encoder — reuse.**
The existing `HartsyInference.Diffusion` T5 encoder (used for Flux, SD3, etc.) supports `t5-base` already. Wire the SAO tokenizer to the same SentencePiece path (`spiece.model`). Verify `last_hidden_state` (not penultimate) is returned. Max length should default to 128 to match training.

**2. Oobleck VAE — new code, but share primitives with iSTFTNet / Kokoro / ACE-Step.**

Required new primitives (under `HartsyInference.Audio.Layers/`):
- `WeightNormConv1D` — Conv1D where `weight = g * v / ||v||`; at load time can be fused to a single Conv1D weight (`fused_weight = g * v / ||v||_per_outchannel`). Recommend **fusing at load** so the forward path is plain Conv1D.
- `WeightNormConvTranspose1D` — same idea on transposed conv.
- `Snake1D` — per-channel learnable `alpha`, formula `x + sin(alpha*x)^2 / (alpha + 1e-9)`. Already needed by ACE-Step too. Make this a CUDA kernel (PTX) — it's elementwise, trivial.
- `OobleckResidualUnit` — 2× Conv1D with snake between, residual add. Compose dilations [1, 3, 9].
- `OobleckEncoderBlock` / `OobleckDecoderBlock` — composes 3 ResidualUnits + downsample/upsample Conv.
- `OobleckEncoder` / `OobleckDecoder` — stems + 5 blocks.
- `OobleckVAE` — wraps encoder + decoder + KL bottleneck. `Encode(wav) -> (mean, logvar)`, `Decode(z) -> wav`, `Reparameterize(mean, logvar) -> z`.

**Memory note:** at 47s × 44.1 kHz × 2 channels × FP32 = 16.6 MB per waveform — trivial. Latents are 64 × 1024 × 4 = 256 KB. The decoder's intermediate activation peak is at the highest-resolution stage just before output (2 channels × 2.1M samples × 128 internal). Plan **chunk-decode** with 128-latent chunks and 16-latent overlap, crossfaded linearly. This caps activation memory at ~50 MB.

**3. DiT — reuse Flux/SD3 blocks heavily, but build a 1D variant.**

The DiT is structurally **a 1D version of SD3's MM-DiT minus the dual-stream image/text branching** — single stream over latent tokens with cross-attention to a separate text+timing token bank, plus adaLN with 6-vector chunking. Reusable from `HartsyInference.Diffusion`:
- **AdaLN modulation** (6-chunk variant from SD3) — direct reuse.
- **RoPE** — reuse Flux's `RotaryEmbedding`, configure for 1D over 1024 positions, applied to first `head_dim//2 = 32` channels per head.
- **SwiGLU FFN** — reuse SD3's `GatedFeedForward` (gate+up+down with SiLU gating, `mult=4`).
- **bias-less LayerNorm** — reuse the SD3 norm.

Build new (under `HartsyInference.Audio.StableAudio.Models/`):
- `FourierFeatures1D` — random-projection sinusoidal embedding; load the projection matrix from safetensors. **Make it CUDA kernel** — single matmul + sin + cos.
- `TimestepEmbed` — `FourierFeatures1D(1, 256) → Linear(256, 1536) → SiLU → Linear(1536, 1536)`.
- `NumberConditioner` — clamps to [0,512], normalizes, calls a `FourierFeatures1D(1, 768)` then optional linear; outputs `(B, 1, 768)`.
- `StableAudioDiTBlock` — composes the reused AdaLN + self-attn (with optional QK-LN for Open Small) + cross-attn-with-12-KV-heads + FFN. **The 12-KV-head cross-attention is the only structurally new attention pattern**; either repeat-interleave KV heads at projection time (2× KV memory waste) or write a small CUDA kernel that broadcasts.
- `StableAudioDiT` — stem `Conv1D(64, 1536, k=1)` (effectively a Linear over channels), 24 (or 16) blocks, head `Conv1D(1536, 64, k=1)`. Plus the "prepend global token" trick from diffusers if loading diffusers-format weights, or pure-adaLN if loading stable-audio-tools-format weights.

**4. Sampler — extend `HartsyInference.Diffusion.Schedulers`.**

- **dpmpp-3m-sde** (for Open 1.0) — port from k_diffusion. Order-3 multistep DPM-Solver++ with SDE noise injection. Needs `polyexponential` sigma schedule `(sigma_min=0.3, sigma_max=500, rho=7, steps=100)`. This sampler is NOT yet in `HartsyInference.Diffusion`. Reuse the existing EDM preconditioning helpers. Cross-link `docs/Research/DIFFUSION_SCHEDULERS.md`.
- **pingpong** (for Open Small) — trivial: denoise to `x_0_hat`, renoise with linear schedule. New code, but tiny (~20 lines).

**5. WAV writer.**
Reuse whatever `HartsyInference.Audio.Vocoders` already has (Kokoro / iSTFTNet need it). Standard 16-bit PCM WAV, sample_rate=44100, channels=2, little-endian. **No** float WAV mode by default (writers like Audacity expect int16 for compatibility).

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

# AudioLDM 2 — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: HartsyInference.Audio (AudioLDM2 pipeline)

> **Stub.** The narrative walkthrough, restated pseudocode and resolved open questions were
> removed on 2026-08-06 — this model is built and verified, so the C# is the source of truth for
> *how it works*. What remains is what the code cannot tell you: upstream provenance, reference
> constants to diff a suspect port against, where implementations disagree, and bring-up traps.
> Full history is in git. Parity evidence: `docs/Checklists/PARITY_VERIFICATION.md`.

## Summary

AudioLDM 2 (Liu et al., 2023) is a text-to-audio latent diffusion model that generates music, sound effects, and (in the speech variants) intelligible speech from natural-language prompts. It produces 16 kHz mono waveforms (typically ~10.24 s per generation) by denoising in a compact mel-spectrogram latent space and vocoding the decoded mel with HiFi-GAN. The model is architecturally unusual: instead of feeding text embeddings directly into the UNet, it uses a **two-stage conditioning pipeline** — (1) CLAP and FLAN-T5-Large jointly encode the prompt, (2) a small **GPT-2** autoregressively produces a fixed-length "AudioMAE-style" continuous feature sequence from those embeddings. The UNet then cross-attends to both the GPT-2 output *and* the original CLAP/T5 text features via two parallel cross-attention streams. Diffusion is classic Gaussian (eps-prediction, DDIM) at 200 steps with CFG 3.5. The full pipeline is: text → (CLAP + T5) → projection → GPT-2 (8 tokens) → UNet denoise (200 steps) → VAE decode → mel → HiFi-GAN → 16 kHz waveform.

This file covers the model architecture and inference pipeline. The HiFi-GAN vocoder (a SpeechT5HifiGan variant with non-standard upsampling) is documented in [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md). Mel preprocessing reference parameters are in [MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md). Classifier-free guidance details are in [CFG_AND_GUIDANCE.md](CFG_AND_GUIDANCE.md). DDIM is in [DIFFUSION_SCHEDULERS.md](DIFFUSION_SCHEDULERS.md). T5 encoder implementation notes are in [TEXT_ENCODERS.md](TEXT_ENCODERS.md).

Sources:
- Paper: [AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining (arXiv:2308.05734)](https://arxiv.org/abs/2308.05734)
- Reference repo: [haoheliu/AudioLDM2](https://github.com/haoheliu/AudioLDM2)
- Diffusers pipeline: [`pipeline_audioldm2.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/audioldm2/pipeline_audioldm2.py)
- Diffusers modeling: [`modeling_audioldm2.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/audioldm2/modeling_audioldm2.py)
- Weights: [`cvssp/audioldm2`](https://huggingface.co/cvssp/audioldm2), [`cvssp/audioldm2-large`](https://huggingface.co/cvssp/audioldm2-large), [`cvssp/audioldm2-music`](https://huggingface.co/cvssp/audioldm2-music)

License: cc-by-nc-sa-4.0 (non-commercial — note this is incompatible with paid HartsyInference use; flag in the pipeline docs).

## Variants

All public checkpoints share the same encoders (CLAP, T5, GPT-2, projection model), the same VAE, and the same SpeechT5HifiGan vocoder. They differ only in the UNet weights and (where applicable) `transformer_layers_per_block`.

| Checkpoint | Task | UNet params | Total params | UNet `transformer_layers_per_block` | Training audio | Sample rate | Default length |
|---|---|---|---|---|---|---|---|
| `cvssp/audioldm2` | general (sfx + music + some speech) | 350 M | ~1.1 B | 1 | 1,150 k hours | 16 kHz | 10.24 s |
| `cvssp/audioldm2-large` | general | 750 M | ~1.5 B | 2 | 1,150 k hours | 16 kHz | 10.24 s |
| `cvssp/audioldm2-music` | music only | 350 M | ~1.1 B | 1 | 665 k hours (music) | 16 kHz | 10.24 s |
| `cvssp/audioldm2-speech-gigaspeech` | TTS (GigaSpeech) | 350 M | ~1.1 B | 1 | speech corpus | 16 kHz | 10.24 s |
| `cvssp/audioldm2-speech-ljspeech` | TTS (LJSpeech) | 350 M | ~1.1 B | 1 | LJSpeech | 16 kHz | 10.24 s |

There is also a `audioldm_48k` upstream variant in the reference repo (not on diffusers): 48 kHz output, 256 mel bins, hop 480, `n_fft` 2048, latent embed dim 16, latent time 128, latent freq 32. **Not** wired into the diffusers pipeline — we'll target the diffusers configs for HartsyInference v1 and revisit 48k later.

The base / large / music checkpoints are the focus of this doc; the speech checkpoints additionally use a VITS-based phoneme encoder in place of T5, which the diffusers `AudioLDM2Pipeline` supports but is a separate code path. We will only implement the **CLAP + T5** general/music path in v1.

## VAE (`AutoencoderKL`, mel-spectrogram variant)

Standard diffusers `AutoencoderKL` with a 1-channel image (mel) instead of 3-channel RGB.

| Field | Value |
|---|---|
| `in_channels` / `out_channels` | 1 / 1 |
| `latent_channels` | 8 |
| `block_out_channels` | (128, 256, 512) |
| `down_block_types` | `DownEncoderBlock2D × 3` |
| `up_block_types` | `UpDecoderBlock2D × 3` |
| `layers_per_block` | 2 |
| `norm_num_groups` | 32 |
| `act_fn` | `silu` |
| `sample_size` | 1024 |
| `scaling_factor` | **0.4110932946205139** (≠ SD's 0.18215) |
| `force_upcast` | true |

`vae_scale_factor = 2 ^ (len(block_out_channels) - 1) = 2^2 = 4`. (Note: **4, not 8** — there are only 3 down-blocks. This differs from SD's VAE.)

Input mel: `(B, 1, mel_time, mel_freq) = (B, 1, 1024, 64)` at 10.24 s. Encoded latent: `(B, 8, 256, 16)`. The decoder is the inverse: latent `(B, 8, 256, 16)` → mel `(B, 1, 1024, 64)`.

Latent normalization at encoding time: `latent = encoder(mel).sample() * scaling_factor`. At decode: `mel = decoder(latent / scaling_factor)`. (We only need decode at inference — encoding is for audio-to-audio remixing, not in the v1 pipeline.)

The VAE kernels are identical to SD's `AutoencoderKL` (we have these); only the channel counts and block depths change.

## Music vs Speech Variants

Same architecture, different training data and (for some variants) tokenizer:
- **General (`audioldm2`, `audioldm2-large`)**: trained on 1,150 k hours of mixed sound effects, music, and some speech from AudioSet and other sources. Best all-rounder; mediocre at intelligible speech.
- **Music (`audioldm2-music`)**: trained on 665 k hours of music only. Higher fidelity instrumental output, but cannot generate sound effects or speech.
- **Speech (`audioldm2-speech-*`)**: trained on the named speech corpus. **Uses a VITS phoneme encoder** in place of T5 — the diffusers `AudioLDM2Pipeline` handles this by branching on `text_encoder_2.config.model_type`. Quality is below dedicated TTS models (Kokoro, F5-TTS) and the model has no speaker control. **Recommendation: do NOT prioritize the speech variant** — HartsyInference already has Kokoro / F5-TTS for TTS.

Inference parameters are identical across variants.

## HartsyInference Implementation Notes

This pipeline is roughly 70% reuse from existing HartsyInference components and 30% new.

**Reuse:**
- T5EncoderModel — already implemented (F-Lite / SD3 / Flux pipelines).
- AutoencoderKL — kernels reused; new config (1 channel in/out, 8 latent channels, 3 down-blocks instead of 4).
- UNet 2D — reuse SD1.5 UNet kernels (ResBlock2D, Transformer2D, CrossAttn, GroupNorm). Modifications: (a) accept 8 in/out channels, (b) different `block_out_channels`, (c) route two `encoder_hidden_states` streams per the `cross_attention_dim[block][sublayer]` table, (d) the input "image" is shaped `(time, mel_freq)`; no math changes.
- HiFiGAN generator — reuse `HifiGanGenerator` from the vocoder layer; new config with 5 upsample stages, mel bins 64, sample rate 16 kHz. See [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md).
- DDIM scheduler — already implemented. Verify `set_alpha_to_one=false` + `steps_offset=1` parity.
- DPM-Solver++ scheduler — already implemented (Z-Image / SD3); plug-and-play.

**New:**
- **CLAP text encoder** (RoBERTa-base + projection MLP). Tokenizer: RoBERTa BPE (need to add to `HartsyInference.ModelAssets.Tokenizers`). Model: ~12-layer post-LN transformer; 768 → 512 projection head. **No audio tower needed for v1.**
- **GPT-2 small** (~124 M). Need to implement:
  - Learned token + positional embeddings (`wte` + `wpe`).
  - 12 × decoder-only transformer blocks (pre-LN, GELU-new, causal self-attn).
  - Final LayerNorm.
  - **No `lm_head` evaluation needed** — we read the last layer's hidden state directly.
  - **Continuous-prefix mode**: feed `inputs_embeds` (skip `wte`). The pipeline never tokenizes anything for GPT-2.
  - **KV-cache**: required for performance. Prefix encoded once on iteration 0, then 8 single-step extensions.
  - **No sampling**: deterministic; no temperature / top-p / top-k logic needed.
  - This is the first decoder-only Transformer in HartsyInference. Architecture is similar to dotLLM's Llama (RoPE → swap for learned positional, RMSNorm → LayerNorm, SwiGLU → GELU MLP). Treat it as the foundation kernel for any future small autoregressive transformer; consider a `HartsyInference.Transformer.Gpt2` package shared with future work.
- **AudioLDM2ProjectionModel** (~80 K params). Two linears + four learned vectors + the SOS/EOS insertion logic. Trivial — implement inline in the AudioLDM2 pipeline class.
- **Dual-stream cross-attention routing** inside the UNet transformer block. Per-sublayer table maps to one of two `encoder_hidden_states` tensors. Add a `int[][] CrossAttnStreamIndex` config to the UNet block and switch the K/V tensor accordingly. Implement once in `Transformer2DBlock`; existing single-stream pipelines pass a length-1 table.

**Validation targets** (vs HuggingFace diffusers Python at FP32 on CPU; tolerance per [BENCHMARKING.md](BENCHMARKING.md)):
- CLAP text features: max |Δ| < 1e-4 on a 32-prompt set.
- T5 encoder: max |Δ| < 1e-4 (existing target).
- Projection model output (post SOS/EOS insertion): exact (no nonlinearity beyond linears).
- GPT-2 forward (single step): max |Δ| < 1e-4 per hidden state.
- GPT-2 8-step continuous generation: max |Δ| < 5e-4 (accumulated).
- UNet single forward at fixed timestep / latent / conditioning: max |Δ| < 1e-3 (FP32) on each output channel.
- VAE decode: max |Δ| < 1e-3 on the mel.
- Vocoder: max |Δ| < 1e-3 on waveform sample values (with the same input mel).
- End-to-end: not bit-exact (DDIM noise sampling differs); compare on **CLAP audio score** of the generated waveform vs the prompt — should match Python within ±0.01 CLAP score over a 32-prompt benchmark.

**Package placement** (one folder per package under `src/`, GPU behind `IBackend`):
- `HartsyInference.Audio.AudioLDM2` — pipeline class, projection model, UNet 2D dual-stream override.
- `HartsyInference.TextEncoders.Clap` — new package (text-only for v1; can grow to add audio tower later).
- `HartsyInference.TextEncoders.T5` — existing.
- `HartsyInference.Transformer.Gpt2` — new (consider naming to allow future reuse; keep tiny).
- `HartsyInference.Vocoder.HifiGan` — existing; add the 5-stage 16 kHz config preset.
- `HartsyInference.Vae` — existing; add the 1-channel mel-VAE config preset.

**Build order** (suggested for [Checklists/](../Checklists/)):
1. CLAP text encoder + RoBERTa tokenizer (CPU + CUDA).
2. GPT-2 small (CPU + CUDA, with KV-cache, continuous `inputs_embeds` mode).
3. AudioLDM2ProjectionModel.
4. UNet 2D dual-stream cross-attention extension.
5. Mel-VAE config preset and validation.
6. SpeechT5HifiGan 5-stage preset and validation.
7. Pipeline glue + DDIM/DPMSolver++ wiring.
8. End-to-end validation against HF diffusers reference outputs.

## Known Gotchas

- `langauge_model_dim` (sic) — typo in the upstream projection model config field name. Preserve it for safetensors compatibility; expose with the corrected spelling at the C# API layer.
- VAE `scaling_factor` is `0.4110932946205139` — **not** SD's 0.18215. Bake into the AudioLDM2 VAE config preset.
- VAE has only 3 down-blocks → `vae_scale_factor = 4` (not 8). All downstream shape arithmetic depends on this.
- The vocoder is shape-sensitive: input mel must be `(B, T, 64)` (T = `audio_length_in_s × 100`), in log-magnitude (natural log), with **no per-feature normalization** (`normalize_before=false`).
- GPT-2 `max_new_tokens=8` is **baked into UNet training**. Don't expose it as a user parameter.
- Per the diffusers source, the UNet supports at most 4 cross-attention sublayers per block. HartsyInference's `Transformer2DBlock` should enforce this and fail loudly otherwise.
- License is cc-by-nc-sa-4.0 (non-commercial). The HartsyInference loader should expose the license string at load time so end-user apps can warn / gate accordingly. Music-only and speech-only variants share this license.
- The reference repo's separate 48 kHz variant uses a different VAE (16 latent channels, different mel bins) and is **not** API-compatible — defer to a separate config preset.

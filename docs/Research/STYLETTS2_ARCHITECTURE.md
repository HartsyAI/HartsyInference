# StyleTTS 2 — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: HartsyInference.Audio (StyleTTS 2 pipeline, Kokoro extension)

> **Stub.** The narrative walkthrough, restated pseudocode and resolved open questions were
> removed on 2026-08-06 — this model is built and verified, so the C# is the source of truth for
> *how it works*. What remains is what the code cannot tell you: upstream provenance, reference
> constants to diff a suspect port against, where implementations disagree, and bring-up traps.
> Full history is in git. Parity evidence: `docs/Checklists/PARITY_VERIFICATION.md`.

## Summary

StyleTTS 2 (Li et al., 2023, [arXiv:2306.07691](https://arxiv.org/abs/2306.07691)) is the parent architecture of Kokoro. It produces 24 kHz speech from text via a five-stage pipeline — PLBERT phoneme encoding → text encoder → prosody predictor (duration / F0 / energy) → length regulation → iSTFTNet (or HiFi-GAN) decoder. Voice identity is carried by a **256-dim style vector** (concatenation of two 128-dim halves: an acoustic style for the decoder, and a prosodic style for the predictor), exactly as in Kokoro. The two architectural pieces Kokoro **removed** are: (1) a **diffusion-based style sampler** — a small 1D transformer-UNet that denoises a 256-d Gaussian latent into a style vector, conditioned on PLBERT text features and (optionally) a reference style; (2) a **speech encoder (StyleEncoder)** — a 2D-Conv ResNet that extracts a 128-d style vector from a reference mel spectrogram. With these two components, StyleTTS 2 supports zero-shot voice cloning, random style sampling, and style-transfer modes that Kokoro cannot do natively. Released checkpoints are LJSpeech (single-speaker, 24 kHz, ~148M params) and LibriTTS (multi-speaker, 24 kHz, ~148M params).

This file documents only what differs from Kokoro. The shared components (PLBERT, TextEncoder, ProsodyPredictor, iSTFTNet decoder, phoneme vocab, voice-pack split convention) are documented in [KOKORO_ARCHITECTURE.md](KOKORO_ARCHITECTURE.md) and are reused unchanged. The G2P stage (espeak-ng IPA) is in [G2P_PHONEMIZATION.md](G2P_PHONEMIZATION.md). Decoder details are in [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md). Mel preprocessing is in [MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md). Karras / V / ADPM2 sampling primitives are in [DIFFUSION_SCHEDULERS.md](DIFFUSION_SCHEDULERS.md). Classifier-free guidance details are in [CFG_AND_GUIDANCE.md](CFG_AND_GUIDANCE.md).

Sources:
- Paper: [arXiv:2306.07691](https://arxiv.org/abs/2306.07691)
- Reference repo: [yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2)
- Weights: [yl4579/StyleTTS2-LibriTTS](https://huggingface.co/yl4579/StyleTTS2-LibriTTS), [yl4579/StyleTTS2-LJSpeech](https://huggingface.co/yl4579/StyleTTS2-LJSpeech)
- Notable forks: [IIEleven11/StyleTTS2](https://github.com/IIEleven11/StyleTTS2) (finetuning), [Stylish-TTS](https://github.com/Stylish-TTS/stylish-tts) (community fork)

License: MIT (yl4579/StyleTTS2 codebase + weights).

## Key Numbers / Constants

| Constant | Value | Notes |
|----------|-------|-------|
| Total inference params (LibriTTS) | ~148 M | After stripping training-only modules. |
| Total inference params (LJSpeech) | ~148 M | Similar topology, minor decoder difference. |
| Sample rate | 24 kHz | Both checkpoints (LJSpeech upsampled from 22.05k training). |
| Mel n_fft / hop / win / n_mels | 2048 / 300 / 1200 / 80 | Standard StyleTTS 2 mel params (LJSpeech config). |
| Style vector dim | 256 | 128 acoustic ⊕ 128 prosodic. |
| StyleEncoder dim_in / max_conv_dim | 64 / 512 | Conv2d ResNet width. |
| Diffusion latent shape | (B, 1, 256) | 256-d style as a length-1 sequence. |
| Diffusion transformer layers | 3 | `args.diffusion.transformer.num_layers`. |
| Diffusion transformer heads / head dim | 8 / 64 | |
| Diffusion sigma_min / sigma_max / rho | 1e-4 / 3.0 / 9.0 | Karras schedule. |
| Diffusion num_steps (typical) | 5 – 10 | 5 = fast, 10 = more diverse. |
| Diffusion CFG embedding_scale | 1.0 – 2.0 | 1.0 = no guidance, 2.0 = strong. |
| Diffusion `embedding_mask_proba` | 0.1 | CFG dropout rate (training-time). |
| Sigma data | 0.2 | Karras normalization. |
| Vocab size | 178 | Same as Kokoro. |
| Max position embeddings | 512 | PLBERT context limit. |
| Decoder upsample (LJSpeech, iSTFTNet) | [10, 6] + iSTFT(20, 5) | Hop product 300. |
| Decoder upsample (LibriTTS, HiFi-GAN) | [10, 5, 3, 2] | Hop product 300. |
| LJSpeech weight file | `epoch_2nd_00100.pth` (~720 MB) | FP32 .pth. |
| LibriTTS weight file | `epochs_2nd_00020.pth` (~745 MB) | FP32 .pth. |

## Data Layouts / Formats

### Reference-style extraction (Mode A)
```
ref_audio (16-bit WAV)  → 24 kHz resample → mono
mel = MelSpectrogram(n_fft=2048, hop=300, win=1200, n_mels=80)(ref_audio)
mel = log(mel + 1e-5)  → normalize  → shape (1, 1, 80, T_mel)
s_acoustic = style_encoder(mel)              # (1, 128)
s_prosodic = predictor_encoder(mel)          # (1, 128)
ref_s      = concat([s_acoustic, s_prosodic], dim=-1)  # (1, 256)
```

### Diffusion sampler call
```
Inputs:
  noise:           (1, 1, 256)        Gaussian N(0, I)
  embedding:       (1, ≤512, 768)     PLBERT text features (bert_dur)
  features:        (1, 256) or None   reference style (Mode A), None for Mode B
  embedding_scale: float              CFG scale, 1.0 – 2.0
  num_steps:       int                5 – 10

Process (ADPM2 + Karras schedule):
  sigmas = KarrasSchedule(1e-4, 3.0, 9.0, num_steps+1)
  x = noise * sigmas[0]
  for i in 0..num_steps-1:
    x = adpm2_step(x, sigmas[i], sigmas[i+1], net, conditioning)
  return x   # (1, 1, 256)

Output:
  s_pred:          (1, 256)           sampled style vector
```

### Diffusion network conditioning (StyleTransformer1d, multispeaker)
```
input_x:         (B, 256, 1)          noised style as a 1-length sequence
time_emb:        (B, 256)             sinusoidal(t) → MLP → 256
context_text:    (B, ≤512, 768)       PLBERT cross-attention keys/values
context_style:   (B, 256) or fixed    reference style → AdaLayerNorm γ/β

Per transformer layer:
  x = SelfAttention(x, time=time_emb, style=context_style)   # AdaLN before
  x = CrossAttention(x, context=context_text)
  x = FeedForward(x, time=time_emb)
```

### Inference-only re-packed weights (target safetensors layout)
```
HartsyInference StyleTTS 2 v1 safetensors:
  bert.*                          PLBERT (shared with Kokoro)
  bert_encoder.weight             Linear(768, 512)
  text_encoder.*                  TextEncoder
  predictor.*                     ProsodyPredictor
  decoder.*                       Decoder (iSTFTNet or HiFi-GAN per config)
  style_encoder.*                 StyleEncoder (Mode A)
  predictor_encoder.*             second StyleEncoder (Mode A)
  diffusion.*                     StyleTransformer1d + fixed-embedding for CFG (Modes A/B/C)
Sidecar JSON: model_config.json with {decoder_type, sample_rate, n_mels, mel params, ...}
```

## Reference Implementations

- [yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2) — Official PyTorch reference.
- [yl4579/StyleTTS2-LJSpeech](https://huggingface.co/yl4579/StyleTTS2-LJSpeech) — LJSpeech checkpoint.
- [yl4579/StyleTTS2-LibriTTS](https://huggingface.co/yl4579/StyleTTS2-LibriTTS) — LibriTTS checkpoint.
- [arXiv:2306.07691](https://arxiv.org/abs/2306.07691) — Paper.
- [IIEleven11/StyleTTS2](https://github.com/IIEleven11/StyleTTS2) — Active fine-tuning fork with clearer training docs.
- [Stylish-TTS/stylish-tts](https://github.com/Stylish-TTS/stylish-tts) — Community refactor / fork with ONNX export work.
- [archinetai/audio-diffusion-pytorch](https://github.com/archinetai/audio-diffusion-pytorch) — Source of `AudioDiffusionConditional`, `StyleTransformer1d`, `Transformer1d`, `KDiffusion`, `ADPM2Sampler`, `KarrasSchedule` (StyleTTS 2 vendors a snapshot of this repo under `Modules/diffusion/`).
- [Karras et al. 2022, "Elucidating the Design Space of Diffusion-Based Generative Models"](https://arxiv.org/abs/2206.00364) — Source for the Karras schedule + ADPM2 sampler.
- [KOKORO_ARCHITECTURE.md](KOKORO_ARCHITECTURE.md) — Companion doc for the shared PLBERT + TextEncoder + Predictor + Decoder stack.

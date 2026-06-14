# StyleTTS 2 — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: HartsyInference.Audio (StyleTTS 2 pipeline, Kokoro extension)

## Summary

StyleTTS 2 (Li et al., 2023, [arXiv:2306.07691](https://arxiv.org/abs/2306.07691)) is the parent architecture of Kokoro. It produces 24 kHz speech from text via a five-stage pipeline — PLBERT phoneme encoding → text encoder → prosody predictor (duration / F0 / energy) → length regulation → iSTFTNet (or HiFi-GAN) decoder. Voice identity is carried by a **256-dim style vector** (concatenation of two 128-dim halves: an acoustic style for the decoder, and a prosodic style for the predictor), exactly as in Kokoro. The two architectural pieces Kokoro **removed** are: (1) a **diffusion-based style sampler** — a small 1D transformer-UNet that denoises a 256-d Gaussian latent into a style vector, conditioned on PLBERT text features and (optionally) a reference style; (2) a **speech encoder (StyleEncoder)** — a 2D-Conv ResNet that extracts a 128-d style vector from a reference mel spectrogram. With these two components, StyleTTS 2 supports zero-shot voice cloning, random style sampling, and style-transfer modes that Kokoro cannot do natively. Released checkpoints are LJSpeech (single-speaker, 24 kHz, ~148M params) and LibriTTS (multi-speaker, 24 kHz, ~148M params).

This file documents only what differs from Kokoro. The shared components (PLBERT, TextEncoder, ProsodyPredictor, iSTFTNet decoder, phoneme vocab, voice-pack split convention) are documented in [KOKORO_ARCHITECTURE.md](KOKORO_ARCHITECTURE.md) and are reused unchanged. The G2P stage (espeak-ng IPA) is in [G2P_PHONEMIZATION.md](G2P_PHONEMIZATION.md). Decoder details are in [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md). Mel preprocessing is in [MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md). Karras / V / ADPM2 sampling primitives are in [DIFFUSION_SCHEDULERS.md](DIFFUSION_SCHEDULERS.md). Classifier-free guidance details are in [CFG_AND_GUIDANCE.md](CFG_AND_GUIDANCE.md).

Sources:
- Paper: [arXiv:2306.07691](https://arxiv.org/abs/2306.07691)
- Reference repo: [yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2)
- Weights: [yl4579/StyleTTS2-LibriTTS](https://huggingface.co/yl4579/StyleTTS2-LibriTTS), [yl4579/StyleTTS2-LJSpeech](https://huggingface.co/yl4579/StyleTTS2-LJSpeech)
- Notable forks: [IIEleven11/StyleTTS2](https://github.com/IIEleven11/StyleTTS2) (finetuning), [Stylish-TTS](https://github.com/Stylish-TTS/stylish-tts) (community fork)

License: MIT (yl4579/StyleTTS2 codebase + weights).

## Detailed Findings

### 1. Released Checkpoints

| Checkpoint | Speaker | Sample rate | Style mode | Params (total) | Weight file | Repo size |
|---|---|---|---|---|---|---|
| `yl4579/StyleTTS2-LibriTTS` | multi-speaker (LibriTTS train-clean-100 + train-clean-360, ~245 speakers) | 24 kHz | zero-shot via reference audio or diffusion | ~148 M | `Models/LibriTTS/epochs_2nd_00020.pth` (~745 MB) | 774 MB total (incl. `reference_audio.zip`, 2.92 MB) |
| `yl4579/StyleTTS2-LJSpeech` | single-speaker (LJSpeech, female) | 24 kHz (upsampled from 22.05 kHz training) | random style via diffusion (no speech-encoder voice cloning, single speaker) | ~148 M | `Models/LJSpeech/epoch_2nd_00100.pth` (~720 MB) + `Models/LJSpeech/config.yml` | 750 MB total |

Both `.pth` files are PyTorch pickle, FP32, containing a dict of state_dicts (`bert`, `bert_encoder`, `predictor`, `decoder`, `text_encoder`, `style_encoder`, `predictor_encoder`, `diffusion`, `text_aligner`, `pitch_extractor`, `mpd`, `msd`, `wd`). The training-only modules (`text_aligner`, `pitch_extractor`, `mpd`, `msd`, `wd` — discriminators and pretrained encoders used during stage-1 / stage-2 training) **can be stripped at packaging time**; only `bert`, `bert_encoder`, `text_encoder`, `predictor`, `decoder`, `style_encoder`, `predictor_encoder`, and `diffusion` are needed for inference. Inference-only state strips to roughly ~590 MB FP32 (~295 MB FP16).

**Param breakdown (approximate, from architecture, both checkpoints):**

| Component | Params | Notes |
|---|---|---|
| PLBERT (CustomAlbert) | ~30 M | Same as Kokoro (hidden 768, 12 shared layers). |
| BERT linear projection (768→512) | ~0.4 M | Same as Kokoro. |
| TextEncoder | ~5 M | Same as Kokoro (Embed + Conv1d×3 + BiLSTM). |
| ProsodyPredictor (DurationEncoder + AdainResBlk1d chains) | ~16 M | Same as Kokoro. |
| iSTFTNet Decoder | ~54 M | Same as Kokoro topology; differs slightly per config (LibriTTS may use HiFiGAN). |
| **StyleEncoder (speech encoder)** | ~13 M | **New vs Kokoro.** 2D Conv ResNet over mel. |
| **PredictorEncoder** (second StyleEncoder) | ~13 M | **New vs Kokoro.** Identical topology. |
| **Diffusion StyleTransformer1d** | ~17 M | **New vs Kokoro.** 3-layer transformer over 256-d latent. |
| **Total inference** | ~148 M | LJSpeech may be slightly smaller depending on `decoder.type`. |

**Decoder type differs by config:**
- LJSpeech (`Configs/config.yml`): `decoder.type: istftnet`, `upsample_rates: [10, 6]`, `gen_istft_n_fft: 20`, `gen_istft_hop_size: 5`. Hop product = 10 × 6 × 5 = 300, matching Kokoro.
- LibriTTS (`Configs/config_libritts.yml`): `decoder.type: hifigan`, `upsample_rates: [10, 5, 3, 2]`. Hop product = 10 × 5 × 3 × 2 = 300. Both produce 24 kHz output with 300-sample hop, but the LibriTTS variant uses full HiFi-GAN upsampling (no iSTFT shortcut) — slightly heavier than Kokoro's iSTFTNet.

### 2. Architecture Differences from Kokoro

| Aspect | Kokoro | StyleTTS 2 |
|---|---|---|
| Style source | Pre-extracted `.pt` voicepack indexed by token length (shape `(511, 1, 256)`) | (a) Pre-extracted from reference audio via **StyleEncoder + PredictorEncoder**, OR (b) sampled from the **diffusion style sampler**, OR (c) blended (clone + diffusion-perturb) |
| Speech encoder (style extractor) | Not released | Released — `StyleEncoder` (2D-Conv ResNet over mel) + a second identical `PredictorEncoder` |
| Style diffusion model | None | `StyleTransformer1d` (multispeaker) / `Transformer1d` (single-speaker), 3 layers, 8 heads, 64 head dim, conditioned on PLBERT text features and optional reference style vector |
| Zero-shot voice cloning | No | Yes — feed any reference mel → 256-d style → synthesize |
| Random style sampling | No (single fixed voice per `.pt`) | Yes — sample noise → diffuse → 256-d style |
| Style transfer (content vs voice) | No | Yes — diffuse with reference style from speaker A while content (text) is from speaker B |
| Decoder | Always iSTFTNet (`[10, 6]` upsample + iSTFT) | iSTFTNet (LJSpeech config) **or** full HiFi-GAN (LibriTTS config, `[10, 5, 3, 2]`) |
| PLBERT | CustomAlbert (12 shared layers, hidden 768) | Same — CustomAlbert with the multilingual PLBERT checkpoint (`Utils/PLBERT/`) |
| Training discriminators on disk | None (inference-only release) | MPD + MSD + a WavLM-based SLM discriminator `wd` (stage-2 adversarial). Strip at packaging. |

**The four model classes shared with Kokoro are bit-identical in topology** — `CustomAlbert` (PLBERT), `TextEncoder`, `ProsodyPredictor` (DurationEncoder + duration head + F0/N AdainResBlk1d chains), and `Decoder` (iSTFTNet) all have the same hidden_dim=512, n_layer=3, style_dim=128, n_token=178, max_dur=50. HartsyInference can reuse the Kokoro modules verbatim and just swap in the StyleTTS 2 weight loader.

### 3. Diffusion Style Sampler

The diffusion sampler is the key novelty of StyleTTS 2 over earlier StyleTTS / VITS-style systems. It models the distribution of style vectors given the input text (and optionally a reference style), enabling sampling of plausible style vectors at inference time.

**Latent.** A single 256-d vector — the concatenation of the two halves of the StyleTTS 2 style vector (`s_acoustic[128] ⊕ s_prosodic[128]`). In the diffusion code this is shaped `(batch, channels=256, length=1)` because the underlying transformer is 1D and treats the style as a length-1 sequence.

**Network: `StyleTransformer1d` (multispeaker) / `Transformer1d` (single-speaker).** Defined in `Modules/diffusion/modules.py`. The model is a small 1D transformer (NOT a convolutional U-Net), wired as:

```
build_model(...):
    diffusion = AudioDiffusionConditional(
        in_channels=1,
        embedding_max_length=bert.config.max_position_embeddings,   # 512
        embedding_features=bert.config.hidden_size,                  # 768
        embedding_mask_proba=args.diffusion.embedding_mask_proba,    # 0.1
        channels=args.style_dim * 2,                                 # 256
        context_features=args.style_dim * 2,                         # 256 (ref-style cond)
    )
```

Inside, `AudioDiffusionConditional` builds (when multispeaker):

```
StyleTransformer1d(
    channels                 = 256,                # the 256-d style latent
    context_embedding_features = 768,              # PLBERT hidden size
    context_features         = 256,                # ref-style vector (optional via masking)
    num_layers               = 3,                  # args.diffusion.transformer.num_layers
    num_heads                = 8,                  # args.diffusion.transformer.num_heads
    head_features            = 64,                 # args.diffusion.transformer.head_features
    multiplier               = 2,
)
```

The transformer block uses time embeddings via `TimePositionalEmbedding` (sinusoidal time embedding → MLP), self-attention, and cross-attention to the PLBERT text features. Reference style is injected via `AdaLayerNorm` modulation in the `StyleTransformer1d` variant (the single-speaker `Transformer1d` omits the style modulation path).

**Classifier-free guidance.** During training, the conditioning embedding is randomly dropped with probability `embedding_mask_proba=0.1` (a `FixedEmbedding` replaces the dropped condition). At inference, the model is called twice and combined: `out = (1 - scale) * unconditional + scale * conditional`. Typical `embedding_scale` is **1.0 – 2.0** (notebook uses 1, 1.5, 2). Higher CFG produces more expressive / stylized speech at the cost of diversity.

**Diffusion type.** `KDiffusion` (Karras et al. 2022 "Elucidated Diffusion"). Sigma distribution at train time is `LogNormalDistribution(mean=args.diffusion.dist.mean, std=args.diffusion.dist.std)` with `sigma_data=0.2`.

**Sampler at inference.** `Demo/Inference_LibriTTS.ipynb` uses:

```python
sampler = DiffusionSampler(
    model.diffusion.diffusion,
    sampler=ADPM2Sampler(),
    sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
    clamp=False,
)
s_pred = sampler(
    noise=torch.randn((1, 256)).unsqueeze(1).to(device),   # (1, 1, 256) latent
    embedding=bert_dur,                                     # (1, seq_len, 768) PLBERT text features
    embedding_scale=embedding_scale,                        # CFG scale, 1.0 – 2.0
    features=ref_s,                                         # (1, 256) reference style or None
    num_steps=diffusion_steps,                              # 5 – 10 typical
).squeeze(1)                                                # → (1, 256)
```

| Sampler parameter | Value | Notes |
|---|---|---|
| `noise` shape | `(B, 1, 256)` | Standard Gaussian. |
| `sigma_min` | 1e-4 | Karras schedule lower bound. |
| `sigma_max` | 3.0 | Karras schedule upper bound. |
| `rho` | 9.0 | Karras schedule curvature exponent. |
| Sampler | `ADPM2Sampler` (second-order Heun-like, Algorithm 2 from Karras et al.) | `AEulerSampler` and `KarrasSampler` also defined but not used at inference. |
| `num_steps` | 5 (typical) – 10 (more diverse) | Tiny — the latent is only 256-d so a handful of steps suffices. |
| `embedding_scale` (CFG) | 1.0 – 2.0 | 1.0 = no guidance (use the conditional output directly), higher = more text-faithful, less stylistic diversity. |
| `clamp` | False | No post-step clamping of the latent. |

**Inference cost.** Per call: `num_steps × 2` (uncond + cond) transformer forward passes over a `(1, 1, 256)` latent with `(1, ≤512, 768)` cross-attention. With 5 steps this is ~10 lightweight transformer evals — negligible compared to the iSTFTNet decoder.

### 4. Speech Encoder (Style Extractor)

The `StyleEncoder` in `models.py` lines ~187–209 maps a reference mel spectrogram to a 128-d style vector. **The full 256-d style vector is the concatenation of two such encoders run in parallel**: `style_encoder` (the acoustic-style half, fed to the decoder) and `predictor_encoder` (the prosodic-style half, fed to the prosody predictor). The two encoders have **identical topology and independent weights**.

**Input.** Mel spectrogram, shape `(B, 1, n_mels=80, T_mel)`. The reference mel is treated as a 2D image. Reference audio length is unconstrained — at least ~3 s is recommended for stable voice cloning; longer references improve quality.

**Topology.** A small 2D-Conv ResNet:

| Stage | Layer | Channels | Spatial |
|---|---|---|---|
| Stem | `spectral_norm(Conv2d(1, dim_in=64, k=3, s=1, p=1))` | 1 → 64 | (80, T) |
| Block 1 | `ResBlk(64, 128, downsample='half', spectral_norm)` | 64 → 128 | (40, T/2) |
| Block 2 | `ResBlk(128, 256, downsample='half')` | 128 → 256 | (20, T/4) |
| Block 3 | `ResBlk(256, 512, downsample='half')` | 256 → 512 | (10, T/8) |
| Block 4 | `ResBlk(512, 512, downsample='half')` (capped at `max_conv_dim=512`) | 512 → 512 | (5, T/16) |
| Tail | `LeakyReLU(0.2)` → `spectral_norm(Conv2d(512, 512, k=5, s=1, p=0))` | 512 → 512 | (1, T/16 − 4) |
| Pool | `AdaptiveAvgPool2d(1)` → `LeakyReLU(0.2)` → flatten | 512 | (1, 1) |
| Head | `Linear(512, style_dim=128)` (`self.unshared`) | 128 | — |

Notes:
- `dim_in` defaults to 64 in the LibriTTS config (it is 64 in `Configs/config_libritts.yml` and `Configs/config.yml`).
- `max_conv_dim` defaults to 512 (same as `hidden_dim`).
- `ResBlk` is the StarGAN-v2 / MUNIT-style residual block (Conv → IN → LeakyReLU → Conv → IN, with a parallel 1×1 shortcut, then AvgPool2d for the `'half'` downsample). All convolutions are wrapped in `spectral_norm`.
- The tail's `Conv2d(..., k=5, p=0)` shrinks the time axis by 4 frames before the adaptive pool.
- This is **not** an attention-pool encoder — it's pure conv + global average pool. The phrasing "attention pool" in some descriptions is incorrect for the released StyleTTS 2 code; it's `AdaptiveAvgPool2d(1)`.

**Two-encoder convention.** At inference, the reference 256-d style is built as:

```python
mel = preprocess(reference_audio)               # (1, 80, T_mel)
ref_s_acoustic  = model.style_encoder(mel.unsqueeze(1))      # (1, 128)
ref_s_prosodic  = model.predictor_encoder(mel.unsqueeze(1))  # (1, 128)
ref_s = torch.cat([ref_s_acoustic, ref_s_prosodic], dim=-1)  # (1, 256)
```

Compare to Kokoro, which uses `ref_s[:, :128]` for the decoder and `ref_s[:, 128:]` for the predictor — same convention, but Kokoro ships the result of these two encoders baked into the voicepack file rather than computing them at runtime.

### 5. Inference Modes

StyleTTS 2 supports three inference modes via different combinations of the speech encoder and the diffusion sampler:

**Mode A — Zero-shot voice cloning (LibriTTS only):**
1. G2P the input text → phoneme tokens (espeak-ng IPA).
2. Compute mel of the reference audio → `ref_s_clone = StyleEncoder(mel) ⊕ PredictorEncoder(mel)` (256-d).
3. PLBERT-encode the phonemes → `bert_dur` (1, seq, 768).
4. Either skip diffusion and use `ref_s_clone` directly, **or** run the diffusion sampler with `features=ref_s_clone` and a small `embedding_scale` (1.0 – 1.5) to produce a slightly perturbed style `s_pred` — the latter often sounds more natural because it lets the model adjust style to the requested text.
5. Synthesize via the standard StyleTTS 2 / Kokoro pipeline.

**Mode B — Random / unconditional style sampling:**
1. G2P + PLBERT as above → `bert_dur`.
2. Run the diffusion sampler with `features=None` (no reference style) and `embedding_scale` ∈ {1.5, 2.0} → 256-d `s_pred`.
3. Synthesize.

   This is the **only mode applicable to LJSpeech** (the speaker is fixed; the diffusion sampler controls intonation / prosodic style only).

**Mode C — Style transfer (long-form synthesis):**

In `Demo/Inference_LibriTTS.ipynb` `LFinference`:
1. Compute `s_prev` = style from the previous sentence's reference (or previous synthesis).
2. For the next sentence, run the sampler with the new text's `bert_dur` and `features=s_prev` but mix the prior style into the prediction using `alpha`, `beta`, and `t` (interpolation parameters) — typical values `alpha=0.3, beta=0.7, diffusion_steps=10, embedding_scale=1.5`.
3. The interpolation produces a sequence of style vectors that vary across sentences while staying close to the target voice, useful for paragraph-length synthesis without unnatural style discontinuities at sentence boundaries.

A degenerate version of Mode C (`STinference`) takes a reference voice's style and a different speaker's text content to produce "voice X reads content from speaker Y" output.

### 6. Phoneme Tokenization

StyleTTS 2 uses **espeak-ng IPA** as its text frontend, with the same 178-token vocabulary as Kokoro (the Kokoro vocab was derived from StyleTTS 2 plus extensions for non-English phonemes). Both LJSpeech and LibriTTS were trained on espeak-ng output for English. The tokenization step is identical to Kokoro:

```
phoneme_string  = espeak-ng --ipa "Hello world"  →  "həlˈoʊ wˈɜːɹld"
token_ids       = [vocab[c] for c in phoneme_string if c in vocab]
input_ids       = [0] + token_ids + [0]    # BOS/EOS = pad token 0
```

**Pure-C# replacement.** Cross-reference [G2P_PHONEMIZATION.md](G2P_PHONEMIZATION.md). The same per-language IPA-dictionary plan documented for Kokoro applies unchanged to StyleTTS 2 — both models share the same 178-token vocab and the same espeak-ng phoneme convention. For StyleTTS 2 specifically:
- LJSpeech: English-only, dictionary + simple OOV fallback is sufficient.
- LibriTTS: English-only, same approach.

(The multilingual PLBERT shipped in `Utils/PLBERT/` would support other languages, but neither released checkpoint was trained on non-English audio.)

### 7. HuggingFace Files

**`yl4579/StyleTTS2-LibriTTS`** (774 MB total):

| File | Size | Purpose |
|---|---|---|
| `Models/LibriTTS/epochs_2nd_00020.pth` | ~745 MB | Stage-2 (final) checkpoint, FP32 PyTorch pickle. Dict of state_dicts: `bert`, `bert_encoder`, `predictor`, `decoder`, `text_encoder`, `style_encoder`, `predictor_encoder`, `diffusion`, plus training-only `text_aligner`, `pitch_extractor`, `mpd`, `msd`, `wd`. |
| `Models/LibriTTS/config.yml` | ~3 KB | Hyperparameters (must be reproduced in our config — model is not self-describing). |
| `reference_audio.zip` | 2.92 MB | Demo reference clips for zero-shot cloning. |
| `.gitattributes` | 113 B | LFS pointers. |

**`yl4579/StyleTTS2-LJSpeech`** (750 MB total):

| File | Size | Purpose |
|---|---|---|
| `Models/LJSpeech/epoch_2nd_00100.pth` | ~720 MB | Stage-2 (final) checkpoint, FP32 PyTorch pickle. Same dict structure as LibriTTS but trained on LJSpeech only. The StyleEncoder is still present in the checkpoint (it's used during training); for single-speaker inference it's only useful if you want to extract a "different reading" style from a held-out LJSpeech clip. |
| `Models/LJSpeech/config.yml` | ~3 KB | Hyperparameters. |
| `.gitattributes` | 72 B | LFS pointers. |

**Auxiliary modules** (downloaded from the StyleTTS 2 repo, NOT from the HF release):
- `Utils/ASR/epoch_00080.pth` (~13 MB) — text aligner, training-only, can be omitted.
- `Utils/JDC/bst.t7` (~30 MB) — F0 / pitch extractor, training-only, can be omitted.
- `Utils/PLBERT/step_1000000.t7` (~110 MB) — PLBERT pretrained weights. **Required** at first-stage training and at model construction time in the official code, but the PLBERT weights are also serialized inside `epochs_2nd_*.pth` (the `bert` key), so once we strip and re-pack the inference checkpoint we don't need to ship `step_1000000.t7` separately.

**Inference-only re-packed format (recommended for HartsyInference shipping):**

Strip the checkpoint to just `{bert, bert_encoder, text_encoder, predictor, decoder, style_encoder, predictor_encoder, diffusion}` and convert PyTorch pickle → safetensors. Approximate sizes:
- FP32: ~590 MB.
- FP16: ~295 MB.
- INT8 (decoder + diffusion): TBD, probably ~200 MB total.

### 8. Memory and Performance

The StyleTTS 2 paper reports a real-time factor (RTF) of about **0.137** on a single NVIDIA Tesla V100 GPU for the LJSpeech model — i.e., it generates 1 s of audio in ~0.14 s wall-clock (roughly 7× faster than realtime). The LibriTTS multispeaker model with zero-shot diffusion is similar order of magnitude. Per-step diffusion cost is negligible compared to iSTFTNet / HiFi-GAN decode.

**Expected HartsyInference numbers (target):**

| Setting | VRAM (FP16) | RTF (RTX 4070-class) | Notes |
|---|---|---|---|
| LJSpeech, 5 diffusion steps, 5 s utterance | ~700 MB | ~0.10 – 0.15 | Inference-only weights ~295 MB FP16 + transient activations. |
| LibriTTS, 5 diffusion steps, zero-shot clone, 5 s utterance | ~750 MB | ~0.15 – 0.20 | Adds StyleEncoder + PredictorEncoder forward pass on the reference (~50 ms one-time per reference). |
| LibriTTS, 10 diffusion steps + CFG, 10 s utterance | ~800 MB | ~0.20 – 0.30 | CFG doubles diffusion cost (uncond + cond). |

Reference encoding is one-time per voice — cache the 256-d `ref_s` and reuse across many utterances of the same speaker.

### 9. C# Implementation Notes

1. **Reuse Kokoro modules verbatim.** PLBERT (CustomAlbert with weight sharing), `TextEncoder`, `ProsodyPredictor`, and the iSTFTNet `Decoder` are bit-identical. Move them under `HartsyInference.Audio/Modules/StyleTTS2Shared/` and have both the Kokoro pipeline and the StyleTTS 2 pipeline reference them. The LibriTTS variant additionally needs a **full HiFi-GAN decoder branch** (`upsample_rates [10, 5, 3, 2]` without the iSTFT shortcut) — implement as a config switch on the existing decoder class, not a separate file.

2. **New module: `StyleEncoder` (and `PredictorEncoder`).** Pure 2D Conv2d / `ResBlk2d` / `AdaptiveAvgPool2d(1)` / `Linear`. Components:
   - `Conv2d` with `kernel_size=3, padding=1` and a `spectral_norm` wrapper (at inference time, spectral_norm collapses to a regular matmul — we can pre-fold it during weight conversion and ship plain conv weights).
   - `ResBlk2d` with `'half'` downsample (Conv + InstanceNorm2d + LeakyReLU(0.2) + AvgPool2d(2), with 1×1 shortcut).
   - `AdaptiveAvgPool2d(output=1)` — `mean` over all spatial dims, trivial.
   - `Linear(512, 128)` head.
   Both encoders are tiny (~13 M params each); FP32 is fine. Add `Conv2d` + `InstanceNorm2d` + `AvgPool2d` ops to `HartsyInference.Core` if not already present (they are needed by other models too).

3. **New module: Diffusion style sampler.** Three sub-tasks:
   - **`StyleTransformer1d` (or `Transformer1d`) network.** A 1D transformer with 3 layers, 8 heads, 64 head dim, self-attention + cross-attention to a `(seq=≤512, dim=768)` PLBERT context, and AdaLayerNorm for reference-style modulation. Reuse our existing transformer block from `HartsyInference.Core/Modules/Transformer` — add an AdaLayerNorm variant.
   - **Time embedding.** `TimePositionalEmbedding`: sinusoidal `(B, dim)` from `t ∈ [0, 1]`, followed by a 2-layer MLP → time features. Standard, identical to Stable Diffusion's time embedding.
   - **Sampler.** Implement `ADPM2Sampler` (second-order Heun / DPM-2) and `KarrasSchedule(sigma_min, sigma_max, rho, num_steps)` in `HartsyInference.Audio/Diffusion/`. Cross-reference [DIFFUSION_SCHEDULERS.md](DIFFUSION_SCHEDULERS.md) — these are well-documented samplers shared with image diffusion.
   - **CFG.** Trivial two-forward-pass mixing with `embedding_scale`. Cross-reference [CFG_AND_GUIDANCE.md](CFG_AND_GUIDANCE.md). The "unconditional" branch uses a `FixedEmbedding` learned during training — a single `(1, 1, 256)` constant that replaces `bert_dur`. Extract it from the diffusion state_dict's `to_embedding.fixed_embedding.embedding.weight` (or similar — verify key during weight conversion).

4. **Weight conversion script.** One-time offline tool:
   - Load `.pth` with Python (or a pure-C# pickle reader — we already have one for safetensors metadata; PyTorch pickle is more involved).
   - Drop training-only keys (`text_aligner`, `pitch_extractor`, `mpd`, `msd`, `wd`).
   - Fold `spectral_norm` parametrizations into plain conv weights (`u`/`v` vectors are stored separately under `xxx.weight_orig`, `xxx.weight_u`, `xxx.weight_v`; the inference weight is `weight_orig / max(σ, 1)` where σ = `u^T W v`).
   - Write FP16 safetensors.
   - Emit a small JSON manifest mapping our flat tensor names to the original `.pth` hierarchy (so we can re-run conversion if naming evolves).

5. **Inference API surface.** Three entry points on the StyleTTS 2 pipeline class:
   - `Synthesize(text, voicepack256)` — Kokoro-compatible path (pre-extracted style). No diffusion call.
   - `Synthesize(text, referenceAudio, diffusionSteps=5, cfgScale=1.0)` — zero-shot voice cloning. Internally extracts the reference style, runs diffusion (optionally), then runs the standard pipeline.
   - `Synthesize(text, diffusionSteps=10, cfgScale=2.0, seed?)` — random style sampling (no reference). Single-speaker LJSpeech mode and "novel voice" mode for LibriTTS.

6. **Mel preprocessing for the StyleEncoder.** Reference audio → 24 kHz mono → mel with `n_fft=2048, hop=300, win=1200, n_mels=80, fmin=0, fmax=12000` (StyleTTS 2's audio_helpers default). Cross-reference [MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md). The mel is log-magnitude (`log(mel + 1e-5)`) and normalized by the same dataset stats used at training (mean ≈ -4, std ≈ 4 — verify by extracting from the training code). Mel input to the StyleEncoder is shape `(1, 1, 80, T_mel)`.

7. **Deterministic random style.** When `seed` is provided, seed our PRNG and produce the diffusion `noise = randn((1, 1, 256))` deterministically. Same seed + same text + same config → bit-identical audio. Validate against the Python reference within 1e-3 PCM tolerance for fixed `(seed, num_steps=5, embedding_scale=1)`.

8. **Reuse the Kokoro voicepack file format for ad-hoc StyleTTS 2 voices.** Once a user has cloned a voice via Mode A, cache the 256-d result as a Kokoro-compatible `.pt` (or our internal flat format) so subsequent runs skip the speech-encoder step. The voicepack convention (`ref_s[:128]` decoder, `ref_s[128:]` predictor) is identical.

9. **LibriTTS HiFi-GAN decoder differs slightly from iSTFTNet.** The LibriTTS config uses `decoder.type: hifigan` with 4 upsampling stages (`[10, 5, 3, 2]`) instead of `istftnet`'s 2 stages (`[10, 6]`) + iSTFT tail. Both have hop-product 300. We need to support both decoder topologies — either as two separate decoder classes or one parameterized class. Recommend one parameterized class with `is_istft: bool` flag, since the upsample/ResBlock chain is otherwise identical.

10. **Validation plan.** For each released checkpoint, generate 50 reference utterances with the Python reference (fixed seed, fixed text, fixed reference audio) and store as ground-truth WAV. HartsyInference output must match within 1e-3 PCM tolerance per sample (or, for the diffusion-sampled style, within a documented MOS-equivalent tolerance — diffusion sampling is float-order-sensitive, so exact-match might require matching the exact ordering of CUDA reductions; bit-exact is a stretch goal).

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

## Algorithm Steps

### Mode A — Zero-shot voice cloning (LibriTTS)
```
1. G2P text → phoneme tokens          (see G2P_PHONEMIZATION.md)
2. Tokenize phonemes → input_ids      [0, *ids, 0]  shape (1, L)
3. Reference audio → mel              shape (1, 1, 80, T_mel)
4. ref_s = concat(style_encoder(mel), predictor_encoder(mel))   shape (1, 256)
5. PLBERT encode input_ids            shape (1, L, 768)
6. (Optional) Diffusion sampler:
     noise = randn((1, 1, 256))
     s_pred = ADPM2_sample(noise, embedding=bert_dur, features=ref_s,
                           num_steps=5, embedding_scale=1.0)
     ref_s := s_pred  # replace
7. bert_encoder(bert_dur) → (1, L, 512) → transpose → (1, 512, L)
8. Duration prediction with s=ref_s[:, 128:]      → durations
9. Alignment matrix → length-regulated features (1, 512, T_frames)
10. F0_pred, N_pred = predictor.F0Ntrain(en, s=ref_s[:, 128:])
11. text_encoder(input_ids) → expand by alignment → asr (1, 512, T_frames)
12. audio = decoder(asr, F0_pred, N_pred, ref_s[:, :128])    24 kHz
```

### Mode B — Random style sampling (LJSpeech or LibriTTS, no reference)
```
1. G2P + tokenize as in Mode A.
2. PLBERT encode → bert_dur.
3. Diffusion sampler:
     noise = randn((1, 1, 256))
     s_pred = ADPM2_sample(noise, embedding=bert_dur, features=None,
                           num_steps=10, embedding_scale=2.0)
     ref_s = s_pred
4–12. Identical to Mode A from step 7 onward.
```

### Mode C — Long-form style continuation
```
Given a sequence of sentences [t_1, t_2, ..., t_N] and a starting reference style s_0
(either from Mode A reference extraction or a Mode B random sample):

  s_prev = s_0
  for t_i in sentences:
     bert_dur_i = PLBERT(tokenize(G2P(t_i)))
     s_pred_i = ADPM2_sample(noise=randn(), embedding=bert_dur_i,
                             features=s_prev, num_steps=10, embedding_scale=1.5)
     # interpolate: stay close to s_prev for continuity
     s_i = alpha * s_prev + (1 - alpha) * s_pred_i        # alpha ≈ 0.3
     synthesize(t_i, s_i)
     s_prev = s_i

The notebook's LFinference function additionally splits long text on sentence
boundaries and concatenates the resulting waveforms with small crossfades.
```

### Spectral-norm folding (one-time offline)
```
For each parametrized layer in the .pth:
  W_orig = state[f"{prefix}.weight_orig"]            # (out, in, ...)
  u      = state[f"{prefix}.weight_u"]               # (out,)
  v      = state[f"{prefix}.weight_v"]               # (in*kh*kw,)  flattened
  σ      = u @ W_orig.reshape(out, -1) @ v
  W_inf  = W_orig / max(σ, 1)
  → emit W_inf as the plain "weight" key for HartsyInference.
```

## Open Questions

- [ ] Exact per-component parameter counts (the ~148 M total is from architecture estimation; the official paper does not publish a per-component breakdown). Verify by loading the checkpoint and summing.
- [ ] Whether the `FixedEmbedding` used for CFG dropout is a single 768-d vector tiled across the sequence dimension, or a learned `(max_len=512, 768)` matrix. Verify the exact key name / shape in the `diffusion.*` state_dict.
- [ ] Whether the LibriTTS HiFi-GAN decoder shares the same `SourceModuleHnNSF` (harmonic + noise excitation) as the iSTFTNet decoder, or omits it. Check `Modules/hifigan.py` vs `Modules/istftnet.py`.
- [ ] Mel normalization stats (mean / std) used at training — confirm exact values used in `audio_helpers` / `meldataset.py`. Required for bit-exact StyleEncoder output.
- [ ] Whether community fine-tunes (e.g., `IIEleven11/StyleTTS2-finetune-experiments`) change any architectural hyperparameters or only the weights. Probably weights only — if so, our loader can accept any compatible checkpoint without code changes.
- [ ] Whether to expose all three Karras-family samplers (`ADPM2Sampler`, `KarrasSampler`, `AEulerSampler`) or just `ADPM2Sampler` (the inference default). Recommend just `ADPM2Sampler` for v1.
- [ ] Behavior with `num_steps=1` or `num_steps=2`. Notebook uses 5+; ultra-fast inference may benefit from distillation later.

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

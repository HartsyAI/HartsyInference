# MeloTTS — Architecture Research Notes

> Status: Complete | Last Updated: 2026-06-27 | Needed Before: HartsyInference.Audio (MeloTTS pipeline, also stage 1 of OpenVoice)
>
> **2026-06-27 implementation corrections (verified against the real English-v3 checkpoint + melo source):**
> (1) the text-encoder embedding scale multiplies the WHOLE sum incl. both BERT projections by sqrt(hidden), not just the id embeddings;
> (2) non-Chinese languages (incl. English) put their 768-dim BERT on the `ja_bert` slot with the 1024 `bert` slot = zeros (no padding);
> (3) English-v3 uses a TRANSFORMER coupling flow (TransformerCouplingBlock), not WaveNet residual coupling;
> (4) the C# text encoder is validated BIT-EXACT (m_p/logs_p corr 1.000000) vs the reference with these corrections. See [[melotts-build]] memory.

> **Stub.** The narrative walkthrough, restated pseudocode and resolved open questions were
> removed on 2026-08-06 — this model is built and verified, so the C# is the source of truth for
> *how it works*. What remains is what the code cannot tell you: upstream provenance, reference
> constants to diff a suspect port against, where implementations disagree, and bring-up traps.
> Full history is in git. Parity evidence: `docs/Checklists/PARITY_VERIFICATION.md`.

## Summary

MeloTTS (MyShell AI, 2024) is a high-quality multilingual TTS family based on **VITS** ([arXiv:2106.06103](https://arxiv.org/abs/2106.06103)) with the **Bert-VITS2** auxiliary-BERT extension. It is also released under MIT, ~70M parameters per language variant, and the **stage-1 acoustic model used by OpenVoice v2**. The architecture is end-to-end (text + speaker → waveform with no separate vocoder file): a phoneme TextEncoder (Transformer, 6 layers, hidden=192) fuses phoneme embeddings with tone, language, and BERT auxiliary features into a prior distribution; a Stochastic Duration Predictor + deterministic Duration Predictor jointly predict per-phoneme frame counts; a 4-layer normalizing Flow (residual coupling with WaveNet blocks) inverts a Gaussian sample into latents; and a **HiFi-GAN V1 generator** upsamples those latents to 44.1 kHz waveform. A per-language pretrained BERT (different model per language) is concatenated into the text encoder at inference; the BERT is run on the original orthographic text (not on phonemes), aligned to phoneme tokens via the language's G2P front-end.

Each language is shipped as a **separate ~208 MB checkpoint.pth** at HuggingFace under `myshell-ai/MeloTTS-<Language>`, with English-v3 ("EN-Newest") being the latest single-speaker English variant. The seven public language variants are: `MeloTTS-English` (5 accents: US, BR, IN, AU, Default), `MeloTTS-English-v2`, `MeloTTS-English-v3` (single-speaker "EN-Newest"), `MeloTTS-Chinese` (with ZH+EN code-switch), `MeloTTS-Spanish`, `MeloTTS-French`, `MeloTTS-Japanese`, `MeloTTS-Korean`.

This file covers the model architecture and inference path. The HiFi-GAN-V1 generator is documented in [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md). The per-language G2P front-ends are in [G2P_PHONEMIZATION.md](G2P_PHONEMIZATION.md). Mel/STFT preprocessing (used only for the BERT speaker condition path and during training) in [MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md). The auxiliary BERT encoders share infrastructure with [TEXT_ENCODERS.md](TEXT_ENCODERS.md).

Sources: [myshell-ai/MeloTTS](https://github.com/myshell-ai/MeloTTS) (MIT), [myshell-ai HuggingFace org](https://huggingface.co/myshell-ai), [VITS paper (arXiv:2106.06103)](https://arxiv.org/abs/2106.06103), [VITS2 (arXiv:2307.16430)](https://arxiv.org/abs/2307.16430), [Bert-VITS2 GitHub](https://github.com/fishaudio/Bert-VITS2), [jaywalnut310/vits reference impl](https://github.com/jaywalnut310/vits).

## HuggingFace files (per variant)

All variants ship the same four-file layout: `checkpoint.pth` (PyTorch pickle, model + optimizer state stripped at release), `config.json`, `README.md`, `.gitattributes`. There is **no safetensors release**; the project ships only `.pth`.

| Variant | HF path | checkpoint.pth | config.json | License |
|---------|---------|----------------|-------------|---------|
| English (multi-accent) | `myshell-ai/MeloTTS-English` | 208 MB | 3.49 kB | MIT |
| English v2 | `myshell-ai/MeloTTS-English-v2` | 208 MB | ~3 kB | MIT |
| English v3 ("EN-Newest") | `myshell-ai/MeloTTS-English-v3` | 208 MB | 3.41 kB | MIT |
| Chinese (ZH+EN mix) | `myshell-ai/MeloTTS-Chinese` | 208 MB | 2.30 kB | MIT |
| Spanish | `myshell-ai/MeloTTS-Spanish` | 208 MB | 3.43 kB | MIT |
| French | `myshell-ai/MeloTTS-French` | 208 MB | 3.41 kB | MIT |
| Japanese | `myshell-ai/MeloTTS-Japanese` | 208 MB | 3.43 kB | MIT |
| Korean | `myshell-ai/MeloTTS-Korean` | 208 MB | 3.41 kB | MIT |

`checkpoint.pth` is a pickled dict with two top-level keys: `"model"` (state_dict for `SynthesizerTrn` including discriminator state) and `"iteration"`/`"learning_rate"`/`"optimizer"` (training scaffolding — ignore at inference). The generator (`dec.*`), text encoder (`enc_p.*`), flow (`flow.*`), and duration predictors (`sdp.*`, `dp.*`) plus the speaker embedding (`emb_g.*`) are the only sub-trees the C# loader needs.

Per-language **auxiliary BERT models are NOT bundled** in these 208 MB checkpoints — they are downloaded separately from HuggingFace at runtime. Disk budget for BERTs:

| BERT | Size | Notes |
|------|------|-------|
| `bert-base-uncased` | ~440 MB | English |
| `hfl/chinese-roberta-wwm-ext-large` | ~1.3 GB | Chinese (it's a *large* RoBERTa) |
| `tohoku-nlp/bert-base-japanese-v3` | ~450 MB | Japanese, +UniDic dictionary needed by tokenizer |
| `dbmdz/bert-base-french-europeana-cased` | ~440 MB | French |
| `dccuchile/bert-base-spanish-wwm-uncased` | ~440 MB | Spanish |
| `kykim/bert-kor-base` | ~440 MB | Korean |

Chinese is the heavy outlier (~5x larger than the synthesizer itself). For a CPU-targeted distribution, document that downloading the BERT is a one-time per-language cost.

## Memory and performance

**Parameters (synthesizer only, inference-only modules)**: roughly 70M per checkpoint. Breakdown estimate (from inspection of config + module dims):

| Module | Approx params |
|--------|--------------|
| TextEncoder (6-layer FFT + BERT proj) | ~5 M |
| Flow (4 × ResidualCouplingLayer, WN(192,k=5,nl=4)) | ~10 M |
| Stochastic Duration Predictor (4 × ConvFlow + DDS) | ~1 M |
| Deterministic Duration Predictor | <1 M |
| Generator (HiFi-GAN-V1 with extra stage) | ~50 M |
| Speaker embedding (256 × 256) | ~65 K |
| **Total (inference)** | **~67 M** |

(The 208 MB on-disk file is ~50 MB heavier than the inference graph would warrant — the difference is the posterior encoder + duration discriminator + multi-period discriminator weights left in the checkpoint.)

**Plus** the BERT auxiliary (110 M for base-uncased, 340 M for Chinese RoBERTa-large).

**VRAM (FP32, inference)**: ~250 MB peak for the synthesizer alone at ~10 s utterance. With BERT loaded, ~700 MB for non-Chinese, ~1.6 GB for Chinese. FP16 halves both.

**RTF (real-time factor) on consumer hardware**: MeloTTS authors advertise **"fast enough for CPU real-time inference"** (>1× faster than realtime on a modern x86 CPU for the synthesizer). On a single CPU thread the synthesizer is ~0.3–0.5 RTF for English. On a 3090/4090 GPU the synthesizer is ~0.02 RTF. BERT inference dominates first-token latency; for short utterances BERT is roughly half the runtime.

**Inference is non-streaming** — the entire duration plan must be computed before the flow + generator can run. Streaming is not supported in the reference code.

## C# implementation notes

1. **VITS is a new model family for HartsyInference** — Kokoro (StyleTTS2) and AudioLDM2 are the closest existing references, but the flow + duration predictor structure is different. The required new building blocks:

   - **WaveNet residual block** (`modules.py::WN`): dilated Conv1d + gated activation `tanh ⊙ sigmoid` + residual + skip, with global conditioning (`g`) projected to `2*channels` and added pre-activation. Used in the flow's coupling layers. **Implement once**, reuse across flow, posterior encoder (if we ever train), and SDP's `cond`.

   - **Residual coupling layer** (`modules.py::ResidualCouplingLayer`): channel split, transform via WN, add to other half, swap (`Flip`). Trivial wrapper around WN.

   - **Stochastic Duration Predictor**: needs `ConvFlow` (piecewise rational-quadratic spline coupling, the [Neural Spline Flows](https://arxiv.org/abs/1906.04032) layer), `DDSConv` (depthwise-separable Conv stack), `Log`, `Flip`, `ElementwiseAffine`. The spline math (`piecewise_rational_quadratic_transform` in `transforms.py`) is the most novel piece — port carefully and validate at FP64 first against PyTorch.

   - **Generator**: standard HiFi-GAN V1 — see [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md). Only modification: 5 upsample stages (not 4), with `[8,8,2,2,2]` and kernels `[16,16,8,2,2]`. The `Generator.forward` injects the speaker `g` at every upsample via a `cond_layer = Conv1d(gin, channels_after_upsample, 1)`.

   - **Path / alignment generator** (`commons.py::generate_path`): converts integer duration tensor `(B, 1, T)` into a one-hot expansion matrix `(B, 1, T_audio, T)`. Easier to implement as a `Repeat` op directly — see implementation pattern from KOKORO_ARCHITECTURE.md.

2. **Auxiliary BERT** can reuse our existing BERT infrastructure (see [TEXT_ENCODERS.md](TEXT_ENCODERS.md)). Each language needs a different model — we should **lazily download** per language on first use, not bundle. Chinese RoBERTa-large is the heavy outlier; for a "MeloTTS lite" distribution we could ship a smaller Chinese BERT (e.g. `bert-base-chinese` at 110 M) at some quality cost.

   The two-slot input layout (`bert` @ 1024 ch + `ja_bert` @ 768 ch) must be preserved exactly. For non-Chinese non-Japanese languages, zero-pad the 768-dim BERT hidden to 1024 (trailing zeros) before passing into the model. Validate this matches the reference by comparing intermediate text-encoder outputs against PyTorch.

3. **Phoneme front-ends** are the biggest engineering item — see [G2P_PHONEMIZATION.md](G2P_PHONEMIZATION.md). Each language is a separate sub-project. Recommended order:
   1. Korean (rules over jamo, trivial — port `g2pkk`)
   2. English (CMUDict lookup + small fallback)
   3. Spanish, French (lexicon + rules)
   4. Chinese (jieba.NET + pinyin dict + tone extraction)
   5. Japanese (only if MeCab+UniDic port is in scope — large)

4. **Multiple variants = one model class, different weights**. Treat exactly like our diffusion model variants: a single `MeloTTSModel` class parameterised by config.json's hyperparameters, with a `variant: enum` switch that controls only G2P + BERT selection. Don't subclass per language.

5. **Checkpoint loading**: `.pth` is a Python pickle. Two options:
   - Build a minimal "torch.load"-compatible pickle reader in C# (we likely already have one for Kokoro). Filter top-level keys; pull only `model.*` and remap key prefixes (`enc_p.`, `flow.`, `dec.`, `sdp.`, `dp.`, `emb_g.`).
   - **Preferred**: convert to safetensors offline at packaging time. Strip discriminators and posterior encoder during conversion (~50 MB saved per variant). This is the pattern from KOKORO_ARCHITECTURE.md item 3.

6. **Determinism**: `noise_scale`, `noise_scale_w`, and the SDP's internal `randn` calls all consume randomness. To reproduce reference output bit-for-bit we need a deterministic RNG seedable from the caller. Use a PCG / xoshiro RNG in `HartsyInference.Core.Random`, NOT `System.Random` (not deterministic across .NET versions). The PyTorch `randn_like` semantics is per-call so the call order matters — log and match it exactly.

7. **`ceil(w)` and `Math.Round`**: same warning as Kokoro. `w_ceil = torch.ceil(w * x_mask)` uses standard ceiling — use `Math.Ceiling`. Be careful with `length_scale = 1/speed` rounding for very small speed values (clamp `speed ∈ [0.1, 10]`).

8. **Alignment expansion via Repeat, not matmul**: the `m_p_expanded = matmul(attn, m_p.T).T` operation is mathematically a per-frame Gather/Repeat. Implement directly: for each input frame `t`, repeat `m_p[:, :, t]` `w_ceil[t]` times along the output axis. Saves a `(B, T_audio, T)` materialization which is enormous for long utterances (200 phonemes × 1000 frames = 200K float entries).

9. **HiFi-GAN-V1 generator**: see [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md) for the full op list (ResBlock1 with dilations [1,3,5], MRF averaging, LeakyReLU slope 0.1, weight-norm removed at inference). The only MeloTTS-specific tweak is the 5-stage `[8,8,2,2,2]` upsample schedule and per-stage speaker injection.

10. **Validation strategy** (port from Kokoro plan):
    - Stage 1: TextEncoder alone — feed deterministic phoneme ids + zero BERT + fixed speaker, compare `(m_p, logs_p)` against PyTorch within 1e-4.
    - Stage 2: SDP — fix RNG seed, sample logw, compare per-phoneme durations exactly.
    - Stage 3: Flow inverse — fix `z_p`, invert, compare `z` within 1e-4.
    - Stage 4: Generator — fix `z` and `g`, compare waveform within 1e-3 PCM (tolerance because LeakyReLU rounding can drift).
    - Stage 5: End-to-end on canonical test sentences from `melo/test_one.py`.

11. **OpenVoice v2 stage-1 reuse**: OpenVoice v2 uses **MeloTTS-English-v3 (EN-Newest)** as its base TTS, then applies a tone-color converter on top. For the OpenVoice integration, we just need MeloTTS-English-v3 to run cleanly and expose its 44.1 kHz output buffer to the next stage. Speaker-control is irrelevant in that mode (single speaker only).

## Key Numbers / Constants

| Constant | Value | Notes |
|----------|-------|-------|
| Total inference parameters | ~67 M | per variant, excluding BERT |
| On-disk checkpoint size | ~208 MB | includes discriminators (training-only) |
| Sample rate | 44,100 Hz | all variants |
| `n_fft` / `filter_length` | 2048 | STFT for posterior encoder (training only) |
| `hop_length` | 512 | matches product of upsample rates |
| `win_length` | 2048 | STFT window |
| Mel spec bins | 1025 | `n_fft/2 + 1`, used as PosteriorEncoder input dim |
| Inter / hidden channels | 192 | flow latent + text encoder hidden |
| Filter channels (FFN) | 768 | text encoder FFN inner dim |
| `gin_channels` | 256 | speaker embedding dim |
| Text encoder layers | 6 | FFT blocks |
| Text encoder heads | 2 | multi-head attention |
| Text encoder kernel | 3 | FFN Conv1d kernel |
| Text encoder window | 4 | relative position window |
| Flow layers (residual) | 4 | each layer = WN + Flip |
| Flow WN sub-layers | 4 | inside each coupling layer's WN |
| Flow WN kernel | 5 | dilated Conv1d kernel |
| Transformer flow layers | 3 | if `use_transformer_flow=True` |
| Posterior encoder layers | 16 | training only — skip |
| SDP flow layers | 4 | ConvFlow + Flip pairs |
| SDP DDS layers | 3 | depthwise-separable Conv1d stack |
| SDP kernel size | 3 | |
| Duration predictor kernel | 3 | DP only |
| Dropout (most modules) | 0.1 | |
| Dropout (SDP, DP) | 0.5 | |
| Speaker embedding slots | 256 | mostly unused per variant |
| Upsample rates | [8,8,2,2,2] | product = 512 = hop_length |
| Upsample kernel sizes | [16,16,8,2,2] | |
| ResBlock kernel sizes | [3,7,11] | HiFi-GAN MRF |
| ResBlock dilations | [[1,3,5],[1,3,5],[1,3,5]] | HiFi-GAN MRF |
| `upsample_initial_channel` | 512 | halved at each upsample stage |
| `n_speakers` slot count | 256 | shared across variants |
| `num_languages` | 8 or 10 | varies per variant config |
| `num_tones` | 11 or 16 | varies per variant config |
| Default `sdp_ratio` | 0.2 | SDP/DP mix |
| Default `noise_scale` | 0.667 | prior sample temperature |
| Default `noise_scale_w` | 0.8 | SDP noise temperature |
| Default `length_scale` | 1.0 | inverse speed |
| Add-blank | true | always interleave blank id=0 between phonemes |

## Data Layouts / Formats

### Input phoneme tensor
```
phones: (1, T)  int64      — ids into config.symbols
tones:  (1, T)  int64      — 0..num_tones-1
lang:   (1, T)  int64      — 0..num_languages-1
length: (1,)    int64      — = T
```
With `add_blank=true`, T = 2*N + 3 where N = phoneme count from G2P (extra +3 from BOS/EOS blank + interleaved blanks).

### BERT feature tensor
```
bert:    (1, 1024, T)  float32  — Chinese RoBERTa-large hidden, or other-lang BERT zero-padded to 1024
ja_bert: (1, 768,  T)  float32  — Japanese BERT hidden (else zeros)
```
Hidden states are word-pooled (typically penultimate transformer layer mean-pooled per word), then repeated by `word2ph[i]` for each word's phoneme count. With add_blank, blank tokens get a copy of the adjacent word's BERT.

### Speaker tensor
```
sid: (1,)   int64    — index into emb_g
g:   (1, 256, 1)  float32  — emb_g(sid), broadcast across time
```

### Prior tensors (TextEncoder output)
```
m_p:     (1, 192, T_phoneme)  float32
logs_p:  (1, 192, T_phoneme)  float32
x_mask:  (1, 1,   T_phoneme)  float32 — 1 where valid, 0 where padded
```

### Duration tensors
```
logw:   (1, 1, T_phoneme)  float32  — log-duration in mel-frame units
w_ceil: (1, 1, T_phoneme)  float32  — integer-valued (kept as float for matmul)
```

### Alignment / expanded prior
```
attn:           (1, 1, T_audio_frames, T_phoneme)  float32  — one-hot
m_p_expanded:   (1, 192, T_audio_frames)  float32
logs_p_expanded:(1, 192, T_audio_frames)  float32
y_mask:         (1, 1,   T_audio_frames)  float32
```

### Latent and waveform
```
z_p:    (1, 192, T_audio_frames)  float32  — sampled from prior
z:      (1, 192, T_audio_frames)  float32  — after flow^-1
audio:  (1, 1,   T_audio_frames * 512)  float32  — generator output, range [-1, 1]
        — squeeze to 1-D, write to WAV at 44100 Hz
```

### Checkpoint .pth (PyTorch pickle)
```
Top-level dict:
  "model"        -> { full SynthesizerTrn state_dict with all submodules }
  "iteration"    -> int (drop)
  "learning_rate"-> float (drop)
  "optimizer"    -> { Adam state (drop) }
Inference-relevant keys under "model":
  enc_p.*        — TextEncoder
  enc_q.*        — PosteriorEncoder (DROP — training only)
  flow.*         — ResidualCouplingBlock (or TransformerCouplingBlock)
  dp.*           — DurationPredictor
  sdp.*          — StochasticDurationPredictor
  dec.*          — Generator (HiFi-GAN-V1)
  emb_g.*        — Speaker embedding
  ref_enc.*      — ReferenceEncoder (DROP — only used if n_speakers <= 0)
```

## Reference Implementations

- [myshell-ai/MeloTTS](https://github.com/myshell-ai/MeloTTS) — Official Python/PyTorch reference, MIT.
- [myshell-ai HuggingFace org](https://huggingface.co/myshell-ai) — All published model weights.
- [jaywalnut310/vits](https://github.com/jaywalnut310/vits) — Original VITS reference (MIT). MeloTTS's `models.py`, `modules.py`, `attentions.py` are direct descendants.
- [daniilrobnikov/vits2](https://github.com/daniilrobnikov/vits2) — VITS2 reference; source of the transformer-flow option and noise-scaled MAS.
- [fishaudio/Bert-VITS2](https://github.com/fishaudio/Bert-VITS2) — Direct upstream of MeloTTS. Source of the auxiliary-BERT extension and the two-slot BERT input layout.
- [VITS paper (arXiv:2106.06103)](https://arxiv.org/abs/2106.06103) — Kim et al., "Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech".
- [VITS2 paper (arXiv:2307.16430)](https://arxiv.org/abs/2307.16430) — Kong et al., architectural improvements over VITS.
- [Neural Spline Flows (arXiv:1906.04032)](https://arxiv.org/abs/1906.04032) — The piecewise rational quadratic spline used in `ConvFlow` inside the SDP.
- [HiFi-GAN paper (arXiv:2010.05646)](https://arxiv.org/abs/2010.05646) — Kong et al., the generator used as-is by MeloTTS (with one extra upsample stage).
- [OpenVoice v2](https://github.com/myshell-ai/OpenVoice) — Downstream consumer; uses MeloTTS-English-v3 as its stage-1 acoustic model.

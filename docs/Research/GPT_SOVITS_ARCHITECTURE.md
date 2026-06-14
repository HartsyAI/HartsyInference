# GPT-SoVITS — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: HartsyInference.Audio (GPT-SoVITS pipeline)

## Summary

GPT-SoVITS ([RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS), 2024-2025, MIT-licensed) is a few-shot voice-cloning TTS system that became the de-facto standard in the Chinese AI / anime community for character voice cloning. It supports Chinese (Mandarin + Cantonese), English, Japanese, and Korean. A 3-10 second reference audio clip (5 s typical) is sufficient for zero-shot cloning; with ~1 minute of fine-tuning data it can produce highly faithful character voices.

The system is a **two-stage cascade**:

1. **Stage 1 (GPT / "T2S" / s1)** — A small causal Transformer ("GPT") that takes (phoneme IDs + BERT contextual features + reference semantic prefix) and autoregressively predicts a sequence of **discrete HuBERT VQ codes** ("semantic tokens"). This is *not* a language model — it is a text-to-semantic decoder, structurally GPT-shaped but specialised for speech.
2. **Stage 2 (SoVITS / "VITS-VC" / s2)** — A VITS-derived `SynthesizerTrn`: posterior encoder + residual-coupling normalising flow + HiFi-GAN decoder + duration / stochastic-duration predictor. At inference it consumes semantic tokens plus a speaker embedding (extracted from reference audio) and synthesises a 32 kHz (v1/v2/v2Pro), 24 kHz (v3) or 48 kHz (v4) waveform.

Supporting models:
- **chinese-hubert-base** — Chinese HuBERT-base used purely as a feature extractor; its 9th transformer layer output is **VQ-quantised** by `enc_p.ssl_proj + Quantizer` inside SoVITS to produce the discrete "semantic tokens" the GPT stage operates on. At inference, this is what encodes the *reference* audio.
- **chinese-roberta-wwm-ext-large** — RoBERTa large (1024-dim, 24 layers) used to embed input text. Only used for Chinese (and Chinese-English mix); Japanese/English/Korean code paths pass zeros for the BERT feature.
- **G2PW (ONNX)** — Conditional Weighted Softmax BERT (INTERSPEECH 2022) for Mandarin polyphone disambiguation.

This file covers GPT-SoVITS specifically. The HiFi-GAN decoder family is in [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md). G2P / phonemisation is in [G2P_PHONEMIZATION.md](G2P_PHONEMIZATION.md). HuBERT/Conformer encoder mechanics are cross-referenced from [PARAKEET_ARCHITECTURE.md](PARAKEET_ARCHITECTURE.md). Mel preprocessing is in [MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md).

Sources: [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS), [lj1995/GPT-SoVITS (HF)](https://huggingface.co/lj1995/GPT-SoVITS), [v2 features wiki](https://github.com/RVC-Boss/GPT-SoVITS/wiki/GPT%E2%80%90SoVITS%E2%80%90v2%E2%80%90features-(%E6%96%B0%E7%89%B9%E6%80%A7)), [v3/v4 features wiki](https://github.com/RVC-Boss/GPT-SoVITS/wiki/GPT%E2%80%90SoVITS%E2%80%90v3v4%E2%80%90features-(%E6%96%B0%E7%89%B9%E6%80%A7)), [DeepWiki overview](https://deepwiki.com/RVC-Boss/GPT-SoVITS), [Medium architecture write-up (axinc-ai)](https://medium.com/axinc-ai/gpt-sovits-a-zero-shot-speech-synthesis-model-with-customizable-fine-tuning-e4c72cd75d87), [Medium inference walk-through (alex)](https://medium.com/@alex1923221/gpt-sovits-audio-inference-process-analysis-bce1f8d3ec20), [OpenVINO blog](https://blog.openvino.ai/blog-posts/openvino-enable-digital-human-tts-gpt-sovits).

## Detailed Findings

### Variants

| Version | Release | Output SR | Key Change | GPT ckpt | SoVITS ckpt | Vocoder |
|---------|---------|-----------|------------|----------|-------------|---------|
| **v1** | early 2024 | 32 kHz | Initial release. 2k hours Chinese training data. Only CN + EN. | `s1bert25hz-2kh-...ckpt` (~155 MB) | `s2G488k.pth` (~93.5 MB) + `s2D488k.pth` (~155 MB discriminator) | HiFi-GAN inside SoVITS (integrated) |
| **v2** | mid 2024 | 32 kHz | 5k hours, added JA + KO + Cantonese. Larger phoneme vocab (732 vs 322). Better zero-shot timbre similarity. **Default for general use.** | `s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt` (~155 MB) (v2 GPT) | `s2G2333k.pth` (~93.5 MB) + `s2D2333k.pth` (~155 MB) | HiFi-GAN integrated |
| **v2Pro / v2ProPlus** | late 2024 | 32 kHz | Architectural refinements (better speaker conditioning, refined flow) at v2's VRAM/speed envelope but quality approaching v4. ProPlus is larger variant. | as v2 | `s2Gv2Pro*.pth` / `s2Gv2ProPlus*.pth` | HiFi-GAN integrated |
| **v3** | early 2025 | 24 kHz | New "shortcut-CFM-DiT" (Diffusion-Transformer + Conditional Flow Matching) replaces VITS posterior + flow as the acoustic backbone. Outputs mel; mel → wave by **nvidia/BigVGAN-v2 24 kHz** vocoder. Better tonal accuracy than v2. Suffers from "metallic artefact" with small fine-tune datasets due to non-integer upsampling between SoVITS internal SR and 24 kHz. | `s1v3.ckpt` (~155 MB) | `s2Gv3.pth` (~700 MB) | `models--nvidia--bigvgan_v2_24khz_100band_256x/` (BigVGAN v2, ~120 MB) |
| **v4** | mid 2025 | **48 kHz** | Same shortcut-CFM-DiT acoustic core as v3 but with author-trained custom vocoder at integer upsampling ratios. Fixes v3's metallic artefacts. Native 48 kHz output. | `s1v3.ckpt` (shared with v3) | `gsv-v4-pretrained/s2v4.pth` (~700 MB) | `gsv-v4-pretrained/vocoder.pth` (~200 MB) |

All checkpoints live under [lj1995/GPT-SoVITS](https://huggingface.co/lj1995/GPT-SoVITS) (official). Common mirrors: [kevinwang676/GPT-SoVITS-v4](https://huggingface.co/kevinwang676/GPT-SoVITS-v4), [kevinwang676/GPT-SoVITS-v-3](https://huggingface.co/kevinwang676/GPT-SoVITS-v-3).

**Recommended order of implementation for HartsyInference**: v2 first (most stable, most fine-tunes in the wild, simplest architecture — pure VITS), then v2Pro, then v4. Skip v1 (superseded) and v3 (superseded by v4).

### Overall Architecture

```
                       (reference audio, 5 s)
                              |
                  ┌───────────┴───────────┐
                  v                       v
         chinese-hubert-base       SoVITS posterior path
         (9th-layer hidden 768)    (linear spectrogram → z_ref)
                  |                       |
        SoVITS enc_p.ssl_proj             |        speaker embed
        + VectorQuantizer                 |        (ref_enc / GST-style)
                  |                       |             |
        prompt_semantic IDs               +-------------+
        (codebook size 1024)                            |
                  |                                     |
                  | (semantic prefix)                   |
                  v                                     |
        ┌─────────────────────┐                         |
text →  │  Stage 1: GPT (t2s) │                         |
        │  - phoneme embed    │                         |
        │  - BERT embed       │                         |
        │  - 24-layer causal  │                         |
        │    Transformer      │                         |
        │  - hidden 512       │                         |
        │  - 16 heads         │                         |
        │  - vocab=1025       │                         |
        │  - EOS=1024         │                         |
        └─────────┬───────────┘                         |
                  | predicted semantic token IDs        |
                  v                                     |
        ┌──────────────────────────────────────┐        |
        │ Stage 2: SoVITS SynthesizerTrn       │<───────┘
        │  - codebook embed (semantic IDs→192) │
        │  - text encoder (CFM-DiT in v3/v4)   │
        │  - flow (residual coupling)          │
        │  - HiFi-GAN decoder                  │
        │  - speaker conditioning              │
        └─────────┬────────────────────────────┘
                  v
              waveform @ 32 / 24 / 48 kHz
```

### Stage 1 — GPT (T2S) Model

File: [`GPT_SoVITS/AR/models/t2s_model.py`](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/GPT_SoVITS/AR/models/t2s_model.py). Config: [`GPT_SoVITS/configs/s1.yaml`](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/GPT_SoVITS/configs/s1.yaml), `s1longer.yaml` (v1 ≤24 layers, v2 = 24 layers).

**Class**: `Text2SemanticDecoder` (training: `Text2SemanticLightningModule`).

**Architecture**: standard pre-norm decoder-only causal Transformer with:

| Hyperparameter | v1 default | v2 / v2Pro |
|----------------|------------|------------|
| `embedding_dim` (model dim) | 512 | 512 |
| `hidden_dim` | 512 | 512 |
| `num_head` | 8 | 16 |
| `num_layers` | 12 | 24 |
| FFN inner | 2048 | 2048 |
| Phoneme vocab (`phoneme_vocab_size`) | 322 | 732 (v2 expanded multilingual set) |
| Semantic vocab (`vocab_size`) | 1025 (1024 codebook + 1 EOS) | 1025 |
| `EOS` token id | 1024 | 1024 |
| Position encoding | learned + sinusoidal mix (`SinePositionalEmbedding`) | same |
| Activation | GELU | GELU |
| Norm | LayerNorm (pre-norm) | LayerNorm (pre-norm) |
| Causal mask | yes, on semantic part only (text part is bidirectional via the BERT input) | same |

**Inputs to forward (training)** are four tensors:
- `phoneme_ids` (1, P_text) — phoneme indices from the text frontend
- `bert_feature` (1, 1024, P_text) — RoBERTa-large hidden state per phoneme (zeros if non-Chinese)
- `prompt_semantic` (1, P_prompt) — semantic IDs from HuBERT-quantised *reference* audio
- `target_semantic` (1, P_target) — semantic IDs to teacher-force

**Concatenation strategy**: Inside the model the inputs are concatenated as:

```
[BERT/phoneme stream | prompt_semantic | target_semantic]
```

The text portion is `phoneme_embed + bert_proj(bert_feature)`. The semantic portion is `semantic_embed(prompt_semantic)`. A causal mask blocks attention from text to semantic and from semantic to future semantic tokens. EOS terminates generation.

**Inference**: greedy / top-k / top-p sampling with KV cache. The original code has three exported sub-modules used by the OpenVINO / mobile path: `t2s_encoder` (text + BERT → context), `first_stage_decoder` (consume prompt prefix, produce first new token), `stage_decoder` (incremental one-step decode). For HartsyInference we can fuse these — there's no compile-graph reason to split them.

**Sampling defaults** (from `inference_webui.py`): `top_k=15`, `top_p=1.0`, `temperature=1.0`, `repetition_penalty=1.35`, `early_stop_num=1500` (hard cap on generated tokens). EOS detection is "token == 1024 OR cumulative duration exceeds reference scale".

### Stage 2 — SoVITS SynthesizerTrn (v1 / v2 / v2Pro)

File: [`GPT_SoVITS/module/models.py`](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/GPT_SoVITS/module/models.py). Config: [`GPT_SoVITS/configs/s2.json`](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/GPT_SoVITS/configs/s2.json).

The SoVITS module is a near-vanilla **VITS** (Kim et al. 2021) `SynthesizerTrn` with one important modification: the "text encoder" input is **discrete HuBERT codes**, not phonemes. (Phonemes are consumed by the GPT in stage 1; the SoVITS stage is essentially a semantic-token → waveform synthesiser conditioned on a speaker embedding.)

Default config values (`s2.json`):

| Section | Field | Value | Notes |
|---------|-------|-------|-------|
| `data` | `sampling_rate` | `32000` | v1/v2/v2Pro |
| `data` | `filter_length` (n_fft) | `2048` | linear spec FFT size |
| `data` | `hop_length` | `640` | linear spec hop (= 32000/50 → 50 fps) |
| `data` | `win_length` | `2048` | linear spec window |
| `data` | `n_mel_channels` | `128` | mel bands (used by some discriminator paths) |
| `data` | `n_speakers` | `300` | total speakers in pretrain table (most are unused at inference) |
| `model` | `inter_channels` | `192` | latent z dim (VITS standard) |
| `model` | `hidden_channels` | `192` | text encoder hidden |
| `model` | `filter_channels` | `768` | text encoder FFN inner |
| `model` | `n_heads` | `2` | text encoder attention heads |
| `model` | `n_layers` | `6` | text encoder transformer layers |
| `model` | `kernel_size` | `3` | text encoder convs |
| `model` | `p_dropout` | `0.1` | |
| `model` | `resblock` | `"1"` | HiFi-GAN ResBlock1 |
| `model` | `resblock_kernel_sizes` | `[3, 7, 11]` | per-ResBlock kernels |
| `model` | `resblock_dilation_sizes` | `[[1,3,5], [1,3,5], [1,3,5]]` | per-ResBlock dilations |
| `model` | `upsample_rates` | `[10, 8, 2, 2, 2]` | 5-stage upsample: 10·8·2·2·2 = 640 ✓ (matches hop_length) |
| `model` | `upsample_initial_channel` | `512` | first decoder channel count |
| `model` | `upsample_kernel_sizes` | `[16, 16, 8, 2, 2]` | matched upsample kernels |
| `model` | `n_layers_q` | `3` | WaveNet residual layers in posterior encoder |
| `model` | `use_spectral_norm` | `false` | |
| `model` | `gin_channels` | `512` | speaker embed dim (broadcast to every component) |
| `model` | `semantic_frame_rate` | `"25hz"` (v2) / `"25hz"` (v1) | semantic token rate |
| `model` | `freeze_quantizer` | `true` (inference) | VQ codebook is frozen post-pretrain |

**Sub-components**:

1. **`enc_p` (`TextEncoder`)** — Embeds the discrete semantic ID stream (vocab=1024, dim 192) and refines through a 6-layer Transformer (192 dim, 2 heads, 768 FFN). At training time, this encoder is fed via `enc_p.ssl_proj(hubert_features)` to project HuBERT 768-d to 192-d, then through a `Quantizer` (residual VQ, 1 codebook, 1024 codes). At inference there is no audio side — only the GPT-predicted IDs are fed via the embedding table. **This is the critical detail**: the SoVITS posterior is conditioned on integer IDs, not continuous HuBERT features.

2. **`enc_q` (`PosteriorEncoder`)** — Used only in training. Takes a linear spectrogram (n_fft//2+1 = 1025 bins) → WaveNet stack (`n_layers_q=3`, 192 hidden, kernel 5, dilation rate 1) → outputs `(mean, logvar)` for the latent `z`. Speaker-conditioned via `gin_channels`.

3. **`flow` (`ResidualCouplingBlock`)** — 4 `ResidualCouplingLayer`s, each a WaveNet (192 hidden, 4 dilation layers, kernel 5). Bijective mapping between posterior latent `z` and prior latent `z_p`. Speaker-conditioned.

4. **`dec` (`Generator`, HiFi-GAN)** — Takes latent `z` and speaker embed, upsamples 640× to waveform via 5 transposed convs interleaved with `ResBlock1` blocks. Final `Conv1d` to 1 channel, tanh activation. Output: float waveform in [-1, 1].

5. **`dp` (`StochasticDurationPredictor`)** — Used by some configs; consists of `Log` flow + 4 coupling layers with `ConvFlow`, plus a `dds_conv` stack. In GPT-SoVITS the role of explicit duration prediction is reduced because the GPT already determines the semantic token sequence length, but the SDP weights are kept for compatibility with the upstream VITS code.

6. **`ref_enc` (reference-audio speaker embedder)** — A small Conv2D stack (typically 6 blocks 32→32→64→64→128→128 with stride-2 downsampling) over the mel spectrogram of the reference audio, followed by a GRU and a linear → `gin_channels`. Outputs a single speaker vector (1, 512). Variants:
   - **v1**: simple `ref_enc` (no GST)
   - **v2**: same `ref_enc`, slightly larger
   - **v2Pro/ProPlus**: adds a learned multi-token attention head (GST-style) for finer timbre control

### Stage 2 — v3 / v4 Acoustic Backbone (Shortcut-CFM-DiT)

In v3 and v4 the **flow + posterior encoder + HiFi-GAN-inside-SoVITS** is replaced by a **Diffusion Transformer (DiT) trained with shortcut Conditional Flow Matching** that outputs a mel spectrogram, plus an external vocoder.

- v3 mel → wave: pretrained `nvidia/bigvgan_v2_24khz_100band_256x` (24 kHz, 100 mel bands, 256× upsample factor).
- v4 mel → wave: author-trained vocoder (`vocoder.pth`) at 48 kHz with integer upsampling rates (fixes v3 metallic artefacts).

DiT depth / width and number of CFM sampling steps:
- Default CFM steps: 32 (v3 / v4). Configurable via `Sample Steps` parameter (4, 8, 16, 32, 64, 128). Lower steps → faster, lower quality.
- Conditioning: GPT-predicted semantic IDs + speaker embed + (optional) duration prior.

For HartsyInference we are deferring v3/v4 until v2 ships. The CFM-DiT requires implementing flow-matching ODE solvers — see [FLOW_MATCHING_AUDIO.md](FLOW_MATCHING_AUDIO.md) for solver design (Euler / midpoint). The BigVGAN v2 vocoder is documented in [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md).

### HuBERT Feature Extractor

File: [`GPT_SoVITS/feature_extractor/cnhubert.py`](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/GPT_SoVITS/feature_extractor/cnhubert.py). Model: [`lj1995/GPT-SoVITS/chinese-hubert-base`](https://huggingface.co/lj1995/GPT-SoVITS/tree/main/chinese-hubert-base).

The Chinese HuBERT base is the standard Facebook HuBERT-base architecture (`fairseq` / HF `Hubert`):

| Hyperparameter | Value |
|----------------|-------|
| Hidden size | 768 |
| Transformer layers | 12 |
| Attention heads | 12 |
| FFN inner | 3072 |
| Conv feature extractor | 7 layers, (512,512,512,512,512,512,512) channels, strides (5,2,2,2,2,2,2), kernels (10,3,3,3,3,2,2) → 50 Hz output rate from 16 kHz input |
| Input | mono 16 kHz audio |
| Param count | ~95 M |
| Weight file size | ~360 MB (FP32) |

**Critical**: GPT-SoVITS uses the **9th transformer layer** hidden state (0-indexed: `output_layer=9`), not the final layer. This is the layer chosen during the original HuBERT VQ-VAE training as the most "semantic" representation.

**Quantisation**: the 9th-layer features (T, 768) are passed through `enc_p.ssl_proj` (Conv1d 768→192) and then a `Quantizer` (residual VQ, single codebook of 1024 vectors, dim 192). The quantiser returns integer codes ∈ [0, 1023] at 25 Hz (downsampled 2× from HuBERT's 50 Hz via a stride-2 conv inside `ssl_proj`). These codes are the "semantic tokens" the GPT operates on.

Cross-reference: HuBERT's transformer blocks share architectural patterns with the Conformer encoders documented in [PARAKEET_ARCHITECTURE.md](PARAKEET_ARCHITECTURE.md), but HuBERT uses standard Transformer blocks (no convolution module) plus a 7-layer 1D conv feature extractor on raw audio. Implement HuBERT as a separate module — do not try to share code with Conformer.

### BERT (chinese-roberta-wwm-ext-large)

Model: [`hfl/chinese-roberta-wwm-ext-large`](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large), shipped under `GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/`.

| Hyperparameter | Value |
|----------------|-------|
| Hidden size | 1024 |
| Layers | 24 |
| Heads | 16 |
| FFN inner | 4096 |
| Vocab | 21,128 (Chinese characters + WordPiece subwords) |
| Max position | 512 |
| Param count | ~325 M |
| Weight file size | ~1.3 GB (FP32) |

**Usage**: per Chinese character, the BERT hidden state (the 3rd-to-last layer, mean-pooled across the subword tokens that compose the character) is expanded to match the per-phoneme stream and concatenated into the GPT input. For **non-Chinese languages (EN, JA, KO), the BERT feature is zeros** — the GPT was trained with zero-BERT on non-Chinese batches, so this works correctly.

**Tradeoff for HartsyInference**: BERT-large is 1.3 GB. We can:
- Ship a quantised (INT8 or Q4) version for the Chinese-only path.
- For purely English/Japanese pipelines, skip loading BERT entirely and pass zeros — measurable quality drop on Chinese, no effect on EN/JA/KO.

### Text Frontend (per-language phonemisation)

File: [`GPT_SoVITS/text/cleaner.py`](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/GPT_SoVITS/text/cleaner.py), `chinese.py`, `english.py`, `japanese.py`, `korean.py`, `symbols.py`.

GPT-SoVITS does **NOT** use IPA. The unified phoneme vocabulary mixes per-language symbols into one table (`symbols.py`):
- 322 tokens in v1 (CN + EN only).
- 732 tokens in v2 (CN + EN + JA + KO + Cantonese).

Per-language tokenisation:

**Chinese (Mandarin)**:
- Text normalisation: numbers→Chinese, currency, dates, units (custom regex rules in `chinese.py`).
- Word segmentation: `jieba` (Chinese word segmenter).
- G2P: per-character lookup in a pinyin dict + **G2PW** ONNX model for polyphone disambiguation (`g2pw` is a BERT-based classifier that picks the correct pronunciation for ambiguous characters).
- Pinyin → phoneme: each pinyin syllable is split into (initial, final) pairs plus a tone digit (1-5). Example: `"ni3 hao3"` → `["n", "i3", "h", "ao3"]`. Tones are baked into the vowel symbol (e.g., `"ao3"` is one token, distinct from `"ao1"`).
- BERT features required.

**Cantonese (yue)**:
- Uses `pycantonese` / jyutping. Romanisation + tones (1-6) similar to pinyin handling. Added in v2.

**English**:
- Text normalisation: numbers, abbreviations, contractions, currency (custom rules in `english.py`).
- G2P: [`g2p_en`](https://pypi.org/project/g2p-en/) — CMU dict lookup for known words, neural Seq2Seq fallback for OOV.
- Output is **ARPABET** phonemes with stress digits: `"HH AH0 L OW1"` etc. The phoneme strings are then mapped to tokens in the unified vocab.

**Japanese**:
- G2P: [`pyopenjtalk`](https://pypi.org/project/pyopenjtalk/) — wrapper around the OpenJTalk text-to-speech preprocessor (Mecab + Naist-jdic).
- Output is a sequence of romaji/kana-mora phonemes. Pitch accent marks are **not** preserved in current GPT-SoVITS (cited as a source of "unnatural intonation" in JA per axinc-ai write-up; this is a known limitation).

**Korean**:
- Custom rule-based jamo decomposition + romanisation. Less polished than CN/JA/EN paths; some communities recommend retraining for serious Korean use.

**Phoneme cleaner**: `cleaner.clean_text(text, language)` dispatches to the per-language module, runs G2P, and returns:
- `phones` — list of phoneme strings
- `word2ph` — list `len(text)` of "how many phones did each character/word contribute" (used for aligning BERT features to phoneme grid)
- `norm_text` — normalised text for BERT input

**ARPABET prefix convention**: ARPABET symbols are prefixed with `@` to ensure uniqueness against single-letter Chinese pinyin symbols (some collide with English single uppercase letters).

For HartsyInference: see [G2P_PHONEMIZATION.md](G2P_PHONEMIZATION.md) for the broader G2P strategy. Specific to GPT-SoVITS we need:
- A Chinese pinyin dictionary (~70k characters → pinyin entries). Convert the jieba dict + pypinyin CSVs to a flat C# `Dictionary<int, string[]>` at packaging time.
- G2PW ONNX or a ported pure-C# version (it is a small BERT classifier; could be replaced by a simpler dictionary-merge heuristic at ~95% accuracy if we don't want a runtime BERT).
- `g2p_en` CMU dict + a small fallback NN (or just drop OOV-to-phoneme accuracy and require known words).
- `pyopenjtalk` mecab+naist data — non-trivial. The simplest path is to ship a precomputed lattice / lexicon for the most common ~100k Japanese tokens and fall back to character-by-character.

### Reference Audio Path

```
reference.wav  (mono, any SR)
   |
   |  resample → 16 kHz
   v
chinese-hubert-base (output_layer=9)
   |  → (T, 768)
   v
enc_p.ssl_proj  Conv1d 768→192, stride 2
   |  → (T/2, 192) at 25 Hz
   v
enc_p.quantizer  residual VQ, 1024 codes
   |  → integer IDs ∈ [0, 1023]
   v
prompt_semantic  (used as prefix for GPT generation)


reference.wav
   |
   |  resample → 32 kHz (v1/v2)
   v
linear spectrogram (n_fft=2048, hop=640, win=2048)
   |  → (1025, T)
   v
ref_enc  Conv2D stack + GRU
   |  → speaker embedding (1, 512)
   v
(broadcast to gin_channels across SoVITS components)
```

A **prompt_text** (transcript of the reference audio in the same language) is required for v3/v4 and strongly recommended for v2/v2Pro — it is prepended to the target text so the GPT sees `[ref_phonemes, ref_BERT] + [target_phonemes, target_BERT]` and conditions accordingly.

### Inference Pipeline (end-to-end)

```python
# Pseudocode aligned with inference_webui.py

1. Load models (one-time):
    cnhubert.load(ckpt_path)              # ~360 MB FP32
    bert = AutoModel.from_pretrained("chinese-roberta-wwm-ext-large")  # ~1.3 GB
    t2s  = Text2SemanticLightningModule.load(s1_ckpt)                  # ~155 MB
    vq   = SynthesizerTrn.load(s2_ckpt)                                # ~93.5 MB (v2)

2. Pre-process reference audio (once per reference):
    ref_wav = load_and_resample(ref_path, 16000)
    pad with 0.3 s of silence at end
    ssl_content = cnhubert(ref_wav, output_layer=9)         # (1, T, 768)
    prompt_semantic = vq.extract_latent(ssl_content)        # (1, T') integer IDs
    spec = spectrogram(load_and_resample(ref_path, 32000))  # (1, 1025, T_spec)
    refer_g = vq.ref_enc(spec)                              # (1, 512)

3. Pre-process target text:
    phones, word2ph, norm_text = clean_text(target_text, lang)
    phone_ids = [symbol_to_id[p] for p in phones]
    if lang == "zh":
        bert_feat = get_bert_feature(norm_text, word2ph)    # (1024, len(phones))
    else:
        bert_feat = zeros((1024, len(phones)))

    prompt_phones, prompt_word2ph, prompt_norm = clean_text(prompt_text, lang)
    prompt_phone_ids = [...]
    prompt_bert_feat = get_bert_feature(prompt_norm, prompt_word2ph) if lang == "zh" else zeros(...)

    # Concatenate prompt + target for in-context conditioning
    all_phone_ids  = concat([prompt_phone_ids,  phone_ids])
    all_bert_feat  = concat([prompt_bert_feat,  bert_feat], dim=1)

4. GPT inference (stage 1):
    pred_semantic = t2s.infer_panel(
        all_phone_ids, all_bert_feat, prompt_semantic,
        top_k=15, top_p=1.0, temperature=1.0, repetition_penalty=1.35,
        early_stop_num=1500,
    )
    # Generation stops at EOS=1024 or early_stop_num.
    # Strip the prompt_semantic prefix from pred_semantic.

5. SoVITS decoding (stage 2):
    audio = vq.decode(
        pred_semantic,                                 # (1, T_pred)
        torch.LongTensor(phone_ids),                   # for length aux
        refer_g,                                       # speaker
        speed=1.0,
    )
    # audio: (1, 1, num_samples) at 32 kHz

6. Post-process:
    audio = audio.squeeze().cpu().numpy() * 32767     # int16 scale
    write_wav("out.wav", 32000, audio.astype(int16))
```

Total cold-load memory at FP32: ~1.9 GB (BERT 1.3 + HuBERT 0.36 + GPT 0.155 + SoVITS 0.094). At FP16, ~950 MB. Steady-state inference VRAM (v2) is around 1.5-2 GB.

### Few-shot Voice Cloning

GPT-SoVITS' few-shot quality scaling:

| Reference length | Quality |
|------------------|---------|
| < 3 s | Marginal — timbre OK but prosody often drifts |
| 5 s | **Sweet spot** — used in all promotional demos |
| 10 s | Slightly more stable; diminishing returns |
| > 30 s | Truncated to first 30 s internally (`max_seconds` limit) |

For higher fidelity, **fine-tuning** is the preferred workflow:
- 1 minute of audio → ~30 minutes of training on a 4090 → very strong voice match.
- Both GPT and SoVITS are fine-tuned. SoVITS LoRA is supported in newer versions.
- Fine-tunes are typically saved as full `s1` and `s2` checkpoints (each ~100-150 MB).

The zero-shot path described above runs against the base `s1`/`s2` checkpoints; the fine-tune path swaps in user-trained replacements with no architectural change.

### HuggingFace File Inventory (lj1995/GPT-SoVITS, main repo)

```
chinese-hubert-base/
    config.json                                                  ~1 KB
    preprocessor_config.json                                     ~0.3 KB
    pytorch_model.bin                                           ~360 MB    Chinese HuBERT-base FP32 weights

chinese-roberta-wwm-ext-large/
    config.json                                                  ~0.5 KB
    pytorch_model.bin                                          ~1.30 GB    RoBERTa-large Chinese FP32
    tokenizer.json                                              ~10 MB
    vocab.txt                                                   ~110 KB

gsv-v2final-pretrained/
    s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt           ~155 MB    v2 GPT (T2S)
    s2G2333k.pth                                                ~93.5 MB   v2 SoVITS generator
    s2D2333k.pth                                                ~155 MB    v2 SoVITS discriminator (training only)

s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt              ~155 MB    v1 GPT
s2G488k.pth                                                    ~93.5 MB   v1 SoVITS G
s2D488k.pth                                                    ~155 MB    v1 SoVITS D (training only)

s1v3.ckpt                                                      ~155 MB    v3 GPT (also used for v4)
s2Gv3.pth                                                      ~700 MB    v3 SoVITS (DiT-CFM)
models--nvidia--bigvgan_v2_24khz_100band_256x/                ~120 MB    v3 vocoder

gsv-v4-pretrained/
    s2v4.pth                                                    ~700 MB    v4 SoVITS (DiT-CFM)
    vocoder.pth                                                 ~200 MB    v4 author vocoder, 48 kHz

v2Pro/
    s2Gv2Pro.pth                                                ~100 MB    v2Pro SoVITS
    s2Gv2ProPlus.pth                                            ~150 MB    v2Pro+ SoVITS

G2PWModel/
    g2pW.onnx                                                   ~200 MB    Chinese polyphone classifier
    bopomofo_to_pinyin_wo_tune_dict.json                        ~120 KB
    char_bopomofo_dict.json                                     ~1 MB
    pos_dict.json, ...
```

**Configs** (small JSON/YAML, in the GitHub repo not HF):
- `GPT_SoVITS/configs/s1.yaml`, `s1longer.yaml`, `s1longer-v2.yaml` — GPT configs per version
- `GPT_SoVITS/configs/s2.json`, `s2v3.json`, `s2v4.json` — SoVITS configs per version
- `GPT_SoVITS/text/symbols.py`, `symbols2.py` — phoneme vocabularies

## Key Numbers / Constants

### Stage 1 GPT (v2 default)

| Constant | Value | Notes |
|----------|-------|-------|
| `embedding_dim` | 512 | Model hidden / embedding dim |
| `num_layers` | 24 | Decoder blocks (v1: 12) |
| `num_head` | 16 | Attention heads (v1: 8) |
| `phoneme_vocab_size` | 732 | Multilingual unified phoneme set (v1: 322) |
| `semantic_vocab_size` | 1025 | 1024 HuBERT codes + 1 EOS |
| `EOS` | 1024 | Stop token |
| BERT input dim | 1024 | RoBERTa-large hidden |
| Max generation length | 1500 | `early_stop_num` safety cap |
| Param count | ~150 M | Including embeddings |

### Stage 2 SoVITS (v1/v2/v2Pro)

| Constant | Value | Notes |
|----------|-------|-------|
| `sampling_rate` | 32000 | Output Hz |
| `filter_length` (n_fft) | 2048 | Linear spec FFT |
| `hop_length` | 640 | Frame stride (= 50 Hz mel rate; 25 Hz semantic rate after VQ stride-2) |
| `win_length` | 2048 | STFT window |
| `n_mel_channels` | 128 | Mel bands |
| `inter_channels` | 192 | VITS latent z dim |
| `hidden_channels` | 192 | Text encoder hidden |
| `filter_channels` | 768 | Text encoder FFN inner |
| `n_heads` (text enc) | 2 | |
| `n_layers` (text enc) | 6 | |
| `kernel_size` | 3 | |
| `gin_channels` | 512 | Speaker embedding dim |
| `n_layers_q` | 3 | Posterior encoder WaveNet layers |
| `n_flow_layers` | 4 | Residual coupling layers (each is a WaveNet) |
| `resblock` | "1" | HiFi-GAN ResBlock1 |
| `resblock_kernel_sizes` | [3, 7, 11] | |
| `resblock_dilation_sizes` | [[1,3,5],[1,3,5],[1,3,5]] | |
| `upsample_rates` | [10, 8, 2, 2, 2] | Product = 640 = hop_length |
| `upsample_initial_channel` | 512 | |
| `upsample_kernel_sizes` | [16, 16, 8, 2, 2] | |
| HuBERT VQ codebook | 1024 | dim 192, single codebook |
| Semantic frame rate | 25 Hz | After stride-2 ssl_proj on 50 Hz HuBERT |
| Param count (SoVITS only) | ~25 M | Generator only (excludes discriminator) |

### Stage 2 SoVITS (v3/v4)

| Constant | Value | Notes |
|----------|-------|-------|
| Output SR | 24000 (v3) / 48000 (v4) | |
| Mel bands | 100 (v3 BigVGAN-v2) / 100 (v4) | |
| Default CFM steps | 32 | Configurable 4-128 |
| Param count (s2v4) | ~250 M | |

### HuBERT

| Constant | Value | Notes |
|----------|-------|-------|
| Input SR | 16000 | Audio resampled to 16 kHz |
| Hidden | 768 | |
| Layers | 12 | |
| Heads | 12 | |
| FFN | 3072 | |
| `output_layer` | 9 | **0-indexed**; the 9th layer (i.e. 10th if 1-indexed) is the semantic representation used |
| Frame rate | 50 Hz | After feature extractor stride 320 from 16 kHz |
| Param count | ~95 M | |

### Performance (RTF, v2 ProPlus per upstream issue #2579)

| Hardware | RTF |
|----------|-----|
| RTX 4090 | 0.014 |
| RTX 4060 Ti | 0.028 |
| Apple M4 (CPU) | 0.526 |

RTF = seconds-of-compute per second-of-audio (lower is faster). 0.014 means 70× real-time on a 4090. CPU RTF > 0.5 means slower than real-time on most laptops.

### Memory

| Configuration | Approx VRAM |
|---------------|-------------|
| v2 FP32 (full pipeline loaded) | 1.9 GB |
| v2 FP16 | 950 MB |
| v2 FP16 without BERT (EN/JA/KO only) | 250 MB |
| v4 FP16 | 1.8 GB (DiT is larger) |

## Data Layouts / Formats

### Phoneme Token Sequence (input to GPT)

```
Input text:      "你好世界"
Phonemes:        ["n", "i3", "h", "ao3", "sh", "i4", "j", "ie4"]
phone_ids:       LongTensor (1, 8)
word2ph:         [2, 2, 2, 2]  # each char produces 2 phonemes
bert_feat:       FloatTensor (1, 1024, 8)  # RoBERTa hidden expanded per-phoneme
```

### Semantic Token Sequence (GPT output / SoVITS input)

```
Shape: LongTensor (1, T_sem)
Values: integers in [0, 1023], plus EOS=1024 at end
Frame rate: 25 Hz (1 token = 40 ms of audio)
Typical length: 5 s of speech → 125 tokens
```

### Reference Speaker Embedding

```
Shape: FloatTensor (1, 512)  # gin_channels = 512
Broadcast: added/concat into every flow + decoder block's conditioning input
Extracted from: linear spectrogram of 32 kHz reference audio
```

### Audio Output

```
Shape: FloatTensor (1, num_samples) in [-1, 1]
Sample rate: 32000 Hz (v1/v2/v2Pro) / 24000 (v3) / 48000 (v4)
Convert to int16 by * 32767, write WAV
```

### GPT Checkpoint Layout (`s1*.ckpt`)

PyTorch Lightning checkpoint, contains:
```
{
  "epoch": int,
  "global_step": int,
  "pytorch-lightning_version": "...",
  "state_dict": {
      "model.ar_text_embedding.weight": (732, 512),
      "model.bert_proj.weight": (512, 1024),
      "model.bert_proj.bias": (512,),
      "model.ar_audio_embedding.weight": (1025, 512),
      "model.ar_text_position.alpha": (...),
      "model.ar_audio_position.alpha": (...),
      "model.h.layers.0.self_attn.in_proj_weight": (1536, 512),
      ...
      "model.ar_predict_layer.weight": (1025, 512),
  },
  "hyper_parameters": { ... full config dump ... },
  "loops": ...,
  "callbacks": ...,
}
```

For HartsyInference we only need `state_dict` and `hyper_parameters`. Convert offline to safetensors + JSON config.

### SoVITS Checkpoint Layout (`s2G*.pth`)

```
{
  "weight": {
      "enc_p.text_embedding.weight": (1025, 192),
      "enc_p.ssl_proj.weight": (192, 768, 2),    # 768→192 with kernel 2 stride 2
      "enc_p.quantizer.vq.codebooks": (1, 1024, 192),
      "enc_p.encoder.attn_layers.0.conv_q.weight": (192, 192, 1),
      ...
      "enc_q.pre.weight": (192, 1025, 1),
      "enc_q.enc.in_layers.0.weight": (...),
      ...
      "flow.flows.0.enc.in_layers.0.weight": (...),
      ...
      "dec.conv_pre.weight": (512, 192, 7),
      "dec.ups.0.weight": (512, 256, 16),
      "dec.resblocks.0.convs1.0.weight": (...),
      ...
      "dec.conv_post.weight": (1, 32, 7),
      "ref_enc.convs.0.weight_v": (...),
      "ref_enc.gru.weight_ih_l0": (...),
      "dp.flows.0.bias": (...),
      ...
  },
  "iteration": int,
  "learning_rate": float,
  "optimizer": {...},
  "config": {...},
  "info": "..."
}
```

Only `weight` is needed at inference. Discriminator (`s2D*.pth`) is training-only.

## Algorithm Steps

### Full Inference Pipeline (v2, end-to-end)

```
1. TEXT FRONTEND (per target language, see G2P_PHONEMIZATION.md)
   Input: target_text, target_lang, prompt_text (optional), prompt_lang
   For Chinese:
     a. Text normalisation (numbers, currency, dates → Chinese)
     b. jieba word segmentation
     c. Per-character pinyin lookup
     d. G2PW polyphone disambiguation for ambiguous chars
     e. Pinyin → (initial, tonal-final) phoneme tokens
   For English:
     a. Text normalisation (numbers, abbreviations)
     b. g2p_en CMU lookup + neural fallback → ARPABET
   For Japanese:
     a. pyopenjtalk → mora phonemes (no pitch accents currently)
   For Korean:
     a. jamo decomposition → romanised phonemes
   Output: phones (list[str]), word2ph (list[int]), norm_text (str)

2. BERT FEATURE EXTRACTION (Chinese only)
   a. Tokenise norm_text with RoBERTa tokenizer
   b. Run RoBERTa-large forward → hidden states
   c. Take 3rd-to-last layer (transformer index -3)
   d. Mean-pool subword tokens per character
   e. Expand per character → per phoneme via word2ph
   Output: bert_feat shape (1024, len(phones))
   For non-Chinese: zeros(1024, len(phones))

3. REFERENCE AUDIO ENCODING (once per reference)
   a. Load ref_wav, resample to 16 kHz
   b. Append 0.3 s silence to end (avoids cut-off artefacts)
   c. cnhubert.forward → 9th-layer hidden (T, 768)
   d. vq_model.enc_p.ssl_proj(hidden) → (T/2, 192)
   e. vq_model.enc_p.quantizer.quantize → integer IDs (T/2,) ← prompt_semantic
   f. Load ref_wav, resample to 32 kHz
   g. linear_spectrogram(ref_wav, n_fft=2048, hop=640, win=2048) → (1025, T_spec)
   h. vq_model.ref_enc(spec) → (1, 512) ← refer_g (speaker embed)

4. GPT INFERENCE (stage 1)
   Inputs:
     all_phone_ids = concat([prompt_phones, target_phones])
     all_bert_feat = concat([prompt_bert, target_bert], dim=-1)
     prompt_semantic (from step 3e)
   Process:
     a. ar_text_embedding(all_phone_ids) → (1, P, 512)
     b. bert_proj(all_bert_feat.T) → (1, P, 512)
     c. text_input = text_emb + bert_emb + pos_enc
     d. ar_audio_embedding(prompt_semantic) → (1, P_prompt, 512) + pos_enc
     e. Concatenate text_input || prompt_audio_input
     f. Loop for n in 1..early_stop_num:
          - Forward through 24 causal decoder layers (with KV cache)
          - Take last position → ar_predict_layer → (1025,) logits
          - Sample (top_k=15, top_p=1.0, temp=1.0, rep_pen=1.35)
          - If sampled == 1024 (EOS): break
          - Append sampled ID, embed it, continue
   Output: pred_semantic (1, T_pred) — predicted semantic IDs

5. SOVITS DECODING (stage 2)
   Inputs:
     pred_semantic (1, T_pred)
     refer_g (1, 512)
     phone_ids (1, P)  # only for length sanity
   Process:
     a. enc_p.text_embedding(pred_semantic) → (1, T_pred, 192)
     b. enc_p.encoder (6-layer transformer) → (1, T_pred, 192) ← prior_mean, prior_logvar
     c. Sample z_p = mean + sigma * eps  (eps ~ N(0,1))
        At inference, eps=0 is common (deterministic) or eps~small for variety
     d. flow.reverse(z_p, g=refer_g) → z  (4-layer residual coupling, reversed)
     e. dec(z, g=refer_g) → waveform via:
          - conv_pre 192→512 channels
          - For each of 5 upsample stages:
              ConvTranspose1d (rate ×10/×8/×2/×2/×2)
              3 ResBlock1 (kernels 3,7,11; dilations [1,3,5])
              Sum block outputs / 3
          - LeakyReLU + conv_post 1 channel + tanh
   Output: audio (1, 1, num_samples) at 32 kHz, range [-1, 1]

6. POSTPROCESS
   a. Squeeze, multiply by 32767, cast int16
   b. Write WAV at 32000 Hz
```

### Semantic Token Extraction Detail (training-time, repeated at inference for ref audio)

```python
# Inside SynthesizerTrn.extract_latent (or equivalent in cnhubert+vq path):

def extract_latent(ssl_features):  # ssl_features: (B, T, 768)
    x = ssl_features.transpose(1, 2)          # (B, 768, T)
    x = enc_p.ssl_proj(x)                     # Conv1d(768, 192, k=2, s=2): (B, 192, T/2)
    quantized, codes, commit_loss = \
        enc_p.quantizer(x)                    # codes: (B, T/2) int in [0, 1023]
    return codes
```

This is the canonical way to produce "what HuBERT semantic IDs does this audio correspond to" — required for (a) building the prompt prefix, (b) any tooling that needs to align text to semantic tokens for fine-tuning data prep.

## Open Questions

- [ ] Exact param count breakdown for v3/v4 DiT — only know s2v4.pth is ~700 MB total. Need to introspect the checkpoint to split DiT vs vocoder.
- [ ] Whether the GPT stage supports KV-cache slicing for streaming output (chunk-wise decode). The upstream `infer_panel` is synchronous; streaming would require us to emit partial waveforms before EOS. Likely doable since semantic tokens are causal and SoVITS decoder is fully convolutional → can decode partial sequences.
- [ ] Cross-lingual transfer behaviour: how well does a Chinese-fine-tuned voice synthesise English text? (Anecdotally good; needs validation.)
- [ ] BERT-zeros fallback for Chinese: does setting BERT=0 for a CN target degrade quality only mildly, or catastrophically? If only mildly, we can ship a no-BERT minimal build at ~600 MB total.
- [ ] Exact tokeniser sharing between Chinese RoBERTa subwords and the symbols vocab — `word2ph` is the bridge but the construction logic is non-trivial.
- [ ] Whether v2Pro and ProPlus are checkpoint-compatible with v2 SoVITS code or require schema changes. Issue #2191 suggests text_embedding shape changed between (322, 192) and (732, 192) — confirm what else changes for Pro.
- [ ] Whether `g2pW` ONNX can be replaced by a dictionary heuristic at acceptable accuracy. (Important: we want a no-ONNX-runtime path.)

## Implementation Notes for HartsyInference

1. **Build order**: implement and validate components in this sequence to keep cycle time short:
   1. Text frontend (CN pinyin + EN ARPABET) → symbol vocab → validate against Python `clean_text` on a fixed corpus.
   2. HuBERT encoder (12-layer Transformer + 7-layer conv feature extractor) → validate 9th-layer hidden matches reference within 1e-4.
   3. VQ quantiser → validate semantic IDs exactly match reference (deterministic; bit-exact required).
   4. SoVITS posterior + flow + decoder → validate decoder output waveform matches reference within 1e-3 PCM tolerance (use deterministic eps=0).
   5. GPT (T2S) → validate logits match reference at first generation step within 1e-3 (with same seed, same sampling, same KV cache state).
   6. End-to-end → MOS-equivalence check (not bit-exact because of sampling stochasticity, but spectrogram correlation > 0.99).

2. **Two-stage cascade is the simplifying factor**: do not try to share weights or fuse stages. Stage 1 emits integers; that's a clean serialisable boundary. Cache `prompt_semantic` + `refer_g` per reference voice — they're cheap once computed (a few KB).

3. **HuBERT is the heaviest non-LLM component**: 95 M params, 12 transformer layers. Reuse our existing Transformer building block (same as used by BERT / CLIP encoders elsewhere in HartsyInference) — HuBERT is a *standard* Transformer encoder with no special modules. The only HuBERT-specific code is the 7-layer 1D conv feature extractor (strides `5,2,2,2,2,2,2`, kernels `10,3,3,3,3,2,2`, 512 channels each, GELU). This is straightforward Conv1D.

4. **VQ quantiser**: implement as a single-codebook residual VQ. The codebook is `(1024, 192)`. Given an input `(B, T, 192)`, return `(B, T)` integer indices via nearest-codebook L2 search. **Deterministic** — must match reference exactly. Use FP32 for the distance computation even if surrounding ops are FP16 (avoid tie-break drift).

5. **GPT KV cache**: the dominant cost is the 24-layer transformer × ~125-1500 tokens. Implement proper KV cache (preallocated tensor of shape `(batch, n_layers, 2, n_heads, max_len, head_dim)`) so each step is O(seq_len) attention not O(seq_len²). This is the same KV-cache pattern as our dotLLM cross-reference — consider extracting a shared `KvCache` type into `HartsyInference.Core`.

6. **Sampling**: top-k + repetition penalty + temperature. Match Python semantics:
   - Repetition penalty is applied multiplicatively to logits of previously generated tokens **before** softmax (PyTorch: `logits[batch, prev_token] /= rep_pen if logits[batch, prev_token] > 0 else logits[batch, prev_token] *= rep_pen`).
   - Top-k truncation, then softmax, then `torch.multinomial(probs, 1)`. For determinism, seed an Xorshift / SplitMix PRNG inside our sampler.

7. **SoVITS flow `reverse=True`**: at inference, the residual coupling layers run in reverse (`z_p → z`). Our implementation must take a `reverse` flag and swap the affine direction (`y = x*scale + bias` forward vs `x = (y - bias)/scale` reverse). The VITS reference is the canonical pattern.

8. **HiFi-GAN decoder**: already covered in [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md). For GPT-SoVITS specifically: 5 upsample stages, rates `[10, 8, 2, 2, 2]`, kernels `[16, 16, 8, 2, 2]`, initial channel 512, ResBlock1 with kernels `[3, 7, 11]` and dilations `[[1,3,5], [1,3,5], [1,3,5]]`. Speaker embed is added via a `Conv1d(gin_channels, channels, 1)` per upsample stage (the standard VITS pattern).

9. **Reference encoder `ref_enc`**: Conv2D stack on the linear spectrogram. Implement directly — small (~2 M params).

10. **Stochastic Duration Predictor**: present in the checkpoint but **not used at inference** in the GPT-SoVITS path (the GPT already determines token count). We can skip implementing SDP entirely for the inference path; load and ignore the weights. **Verify** this with a test by zeroing SDP weights and confirming output is unchanged.

11. **BERT-large**: at 1.3 GB this is the single largest model. Strategies:
    - Quantise to INT8 or Q4 at packaging.
    - Offer a "no-BERT" build for EN/JA/KO only (BERT input is zeros anyway for these langs).
    - Load lazily — only instantiate when a Chinese target is requested.
    - Stream weights from disk via mmap as we do for diffusion models.

12. **G2PW**: the upstream is an ONNX model. Options:
    - Re-implement in pure C# using our existing BERT loader (it's a small Chinese-BERT classifier on top of `bert-base-chinese` ~400 MB).
    - Replace with a dictionary-merge heuristic (`pypinyin` heteronym dict + jieba POS tags) at ~95% accuracy. Acceptable for most workflows.
    - **Recommended**: ship the dictionary heuristic in v1, add the pure-C# G2PW model in v2.

13. **Text normalisation per language**: the per-language `chinese.py`, `english.py`, `japanese.py`, `korean.py` files contain dozens of regex rules. Port them line-by-line. Validate with golden tests on a corpus of weird inputs (phone numbers, dates, currency, emoji).

14. **jieba in C#**: there are existing pure-C# ports of jieba (`jieba.NET`) — usable but check licence (MIT compatible). Alternatively, freeze the segmentation at packaging time for any known text and ship a lookup.

15. **pyopenjtalk in C#**: this is hard. OpenJTalk uses Mecab + NAIST-jdic which are large dictionaries with proprietary-ish licences. Options:
    - Ship a precomputed Japanese lexicon (~50 MB) covering top 100k tokens, fall back to per-character katakana for unknown.
    - Skip Japanese in v1; ship CN + EN first.

16. **Sample rate handling**: v2 is fixed at 32 kHz. Provide a built-in resampler to common rates (16/22.05/24/44.1/48 kHz) using a polyphase FIR. Don't expose 32 kHz as user-tunable — the model was trained at this rate and changing it breaks alignment.

17. **Streaming**: defer to v2 of our pipeline. Initial release: full-utterance synthesis only.

18. **Reference audio cache**: extract and cache `(prompt_semantic, refer_g, prompt_phone_ids, prompt_bert_feat)` per voice. These are tiny (~few KB) and avoid re-running HuBERT every utterance.

19. **Cross-reference with SwarmUI extension**: the SwarmUI HartsyInference extension (from MEMORY.md) will need to expose a TTS backend with per-voice cache and language selector. Plan the public API surface (`SoVitsPipeline.Synthesize(text, voice, language)`) before implementing internals, so the SwarmUI wiring is uncontroversial.

20. **Validation harness**: build a test that takes (text, ref_audio, ref_text) → produces waveform, then compares against a Python-generated reference waveform via mel-spectrogram L2 distance. Target: < 0.02 normalised MSE for v2 base voices.

## Reference Implementations

- [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) — Official Python/PyTorch reference. Mostly Chinese documentation; English wiki pages exist.
- [lj1995/GPT-SoVITS](https://huggingface.co/lj1995/GPT-SoVITS) — Official model weights (v1/v2/v2Pro/v3/v4 all here).
- [kevinwang676/GPT-SoVITS-v4](https://huggingface.co/kevinwang676/GPT-SoVITS-v4) — Community mirror with cleaner v4 file layout.
- [kevinwang676/GPT-SoVITS-v-3](https://huggingface.co/kevinwang676/GPT-SoVITS-v-3) — Community v3 mirror.
- [X-T-E-R/GPT-SoVITS-Inference](https://github.com/X-T-E-R/GPT-SoVITS-Inference) — Inference-only fork, simpler entrypoint (good source for reading the inference flow without the training code).
- [GPT-SoVITS-Infer (PyPI)](https://pypi.org/project/GPT-SoVITS-Infer/) — pip-installable inference wrapper.
- [GitYCC/g2pW](https://github.com/GitYCC/g2pW) — Reference G2PW polyphone disambiguator (INTERSPEECH 2022).
- [pyopenjtalk](https://pypi.org/project/pyopenjtalk/) — Japanese G2P (the standard).
- [g2p_en](https://pypi.org/project/g2p-en/) — English G2P / CMU dict + neural fallback.
- [jaywalnut310/vits](https://github.com/jaywalnut310/vits) — Original VITS reference (SynthesizerTrn comes from here).
- [hfl/chinese-roberta-wwm-ext-large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large) — BERT used.
- [TencentGameMate/chinese_speech_pretrain](https://github.com/TencentGameMate/chinese_speech_pretrain) — Origin of `chinese-hubert-base` weights.
- [nvidia/BigVGAN](https://github.com/NVIDIA/BigVGAN) — v3 vocoder.
- [DeepWiki: RVC-Boss/GPT-SoVITS](https://deepwiki.com/RVC-Boss/GPT-SoVITS) — Auto-generated architecture documentation (useful navigation).
- [Medium: GPT-SoVITS architecture (axinc-ai)](https://medium.com/axinc-ai/gpt-sovits-a-zero-shot-speech-synthesis-model-with-customizable-fine-tuning-e4c72cd75d87) — English write-up.
- [Medium: GPT-SoVITS inference walkthrough (alex)](https://medium.com/@alex1923221/gpt-sovits-audio-inference-process-analysis-bce1f8d3ec20) — English code walkthrough.
- [OpenVINO blog: enabling GPT-SoVITS](https://blog.openvino.ai/blog-posts/openvino-enable-digital-human-tts-gpt-sovits) — Inference graph decomposition (t2s_encoder / first_stage / stage_decoder).
- [v2 features wiki](https://github.com/RVC-Boss/GPT-SoVITS/wiki/GPT%E2%80%90SoVITS%E2%80%90v2%E2%80%90features-(%E6%96%B0%E7%89%B9%E6%80%A7)) | [v3/v4 features wiki](https://github.com/RVC-Boss/GPT-SoVITS/wiki/GPT%E2%80%90SoVITS%E2%80%90v3v4%E2%80%90features-(%E6%96%B0%E7%89%B9%E6%80%A7)) — Official version-feature pages.

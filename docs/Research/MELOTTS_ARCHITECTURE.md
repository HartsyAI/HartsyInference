# MeloTTS — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: HartsyInference.Audio (MeloTTS pipeline, also stage 1 of OpenVoice)

## Summary

MeloTTS (MyShell AI, 2024) is a high-quality multilingual TTS family based on **VITS** ([arXiv:2106.06103](https://arxiv.org/abs/2106.06103)) with the **Bert-VITS2** auxiliary-BERT extension. It is also released under MIT, ~70M parameters per language variant, and the **stage-1 acoustic model used by OpenVoice v2**. The architecture is end-to-end (text + speaker → waveform with no separate vocoder file): a phoneme TextEncoder (Transformer, 6 layers, hidden=192) fuses phoneme embeddings with tone, language, and BERT auxiliary features into a prior distribution; a Stochastic Duration Predictor + deterministic Duration Predictor jointly predict per-phoneme frame counts; a 4-layer normalizing Flow (residual coupling with WaveNet blocks) inverts a Gaussian sample into latents; and a **HiFi-GAN V1 generator** upsamples those latents to 44.1 kHz waveform. A per-language pretrained BERT (different model per language) is concatenated into the text encoder at inference; the BERT is run on the original orthographic text (not on phonemes), aligned to phoneme tokens via the language's G2P front-end.

Each language is shipped as a **separate ~208 MB checkpoint.pth** at HuggingFace under `myshell-ai/MeloTTS-<Language>`, with English-v3 ("EN-Newest") being the latest single-speaker English variant. The seven public language variants are: `MeloTTS-English` (5 accents: US, BR, IN, AU, Default), `MeloTTS-English-v2`, `MeloTTS-English-v3` (single-speaker "EN-Newest"), `MeloTTS-Chinese` (with ZH+EN code-switch), `MeloTTS-Spanish`, `MeloTTS-French`, `MeloTTS-Japanese`, `MeloTTS-Korean`.

This file covers the model architecture and inference path. The HiFi-GAN-V1 generator is documented in [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md). The per-language G2P front-ends are in [G2P_PHONEMIZATION.md](G2P_PHONEMIZATION.md). Mel/STFT preprocessing (used only for the BERT speaker condition path and during training) in [MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md). The auxiliary BERT encoders share infrastructure with [TEXT_ENCODERS.md](TEXT_ENCODERS.md).

Sources: [myshell-ai/MeloTTS](https://github.com/myshell-ai/MeloTTS) (MIT), [myshell-ai HuggingFace org](https://huggingface.co/myshell-ai), [VITS paper (arXiv:2106.06103)](https://arxiv.org/abs/2106.06103), [VITS2 (arXiv:2307.16430)](https://arxiv.org/abs/2307.16430), [Bert-VITS2 GitHub](https://github.com/fishaudio/Bert-VITS2), [jaywalnut310/vits reference impl](https://github.com/jaywalnut310/vits).

## Detailed Findings

### Architectural Lineage

MeloTTS is a fork/derivative of **Bert-VITS2** (which is itself a fork of `daniilrobnikov/vits2`, which forks `jaywalnut310/vits`). The README explicitly credits "TTS [Coqui], VITS, VITS2 and Bert-VITS2". Concretely this means:

- **From VITS (Kim et al. 2021)**: the variational end-to-end formulation, the posterior encoder (training-only), the residual-coupling normalizing flow, the stochastic duration predictor, and the HiFi-GAN-style generator stitched into a single model trained jointly.
- **From VITS2 (Kong et al. 2023)**: transformer-flow option (`use_transformer_flow`), noise-scaled monotonic alignment search (`use_noise_scaled_mas: true`), duration discriminator (training only), speaker-conditioned text encoder (`use_spk_conditioned_encoder: true`).
- **From Bert-VITS2**: the auxiliary BERT branch fused into the text encoder, tone + language ID embeddings on every phoneme token, and the multilingual phoneme symbol table.

Because the posterior encoder, duration discriminator, and all discriminator-side modules are training-only, the **inference graph** is much smaller than the on-disk checkpoint suggests. Roughly two-thirds of the parameters at inference live in the HiFi-GAN generator.

### Class Map (`melo/models.py`, `melo/modules.py`, `melo/attentions.py`)

| Component | Class | Purpose | Inference? |
|-----------|-------|---------|------------|
| Top-level model | `SynthesizerTrn` | Orchestrates everything | yes |
| Phoneme + BERT encoder | `TextEncoder` | Phoneme → prior `(m_p, logs_p)` | yes |
| Posterior encoder | `PosteriorEncoder` | Mel → `(m_q, logs_q)` | **no** (training-only) |
| Normalizing flow | `ResidualCouplingBlock` or `TransformerCouplingBlock` | Latent ↔ prior space | yes |
| Stochastic duration | `StochasticDurationPredictor` | Phoneme → log-duration via flow | yes |
| Deterministic duration | `DurationPredictor` | Phoneme → log-duration via Conv1d | yes |
| Generator (vocoder) | `Generator` (HiFi-GAN V1) | Latent → 44.1 kHz waveform | yes |
| Speaker embedding | `nn.Embedding(n_speakers, gin_channels)` | Speaker ID → 256-dim vector | yes |
| Reference encoder | `ReferenceEncoder` | Mel → 256-dim style vector | only if `n_speakers ≤ 0` |
| Multi-period discriminator | `MultiPeriodDiscriminator` | Adversarial loss | **no** |
| Duration discriminator | `DurationDiscriminator` | Duration adversarial loss | **no** |

### TextEncoder — phoneme + tone + language + BERT fusion

Located in `models.py::TextEncoder`. The input is **four parallel sequences of length `T`** (phoneme count) plus two BERT tensors:

| Input | Shape | Source |
|-------|-------|--------|
| `x` (phoneme ids) | `(B, T)` int64 | per-language G2P front-end |
| `tone` | `(B, T)` int64 | per-phoneme tone id (0 if untuned) |
| `language` | `(B, T)` int64 | per-phoneme language id (0..num_languages-1) |
| `bert` (main BERT) | `(B, 1024, T)` float | hidden state of the language's BERT, scattered to phoneme positions |
| `ja_bert` | `(B, 768, T)` float | optional Japanese BERT branch (always passed, zeros if unused) |

The forward is:

```
x_emb     = embedding(x)          # (B, T, hidden)         hidden=192
tone_emb  = tone_embedding(tone)  # (B, T, hidden)
lang_emb  = lang_embedding(lang)  # (B, T, hidden)
bert_emb  = Conv1d(1024, 192, k=1)(bert)        # transpose handled in code
ja_emb    = Conv1d( 768, 192, k=1)(ja_bert)
h = (x_emb + tone_emb + lang_emb) * sqrt(hidden) + bert_emb + ja_emb
h = transformer_encoder(h, mask)                 # 6 layers
stats   = Conv1d(hidden, 2*inter_channels, 1)(h) # → (B, 2*192, T)
m_p, logs_p = chunk(stats, 2, dim=1)
```

The transformer (in `attentions.py::Encoder`) is a standard FFT block: pre-LN, multi-head self-attention with **relative position embeddings** (window=4, same as VITS), Conv1d FFN (kernel=3, filter_channels=768), GELU. `n_heads=2`, `n_layers=6`, `p_dropout=0.1`.

Key fact: **the BERT vector for each phoneme is the BERT hidden state of the *grapheme word that produced the phoneme***, broadcast to every phoneme of that word. The mapping is done in the per-language `g2p()` function — see "Phoneme tokenization" below.

### PosteriorEncoder (training-only, can be omitted at inference)

WaveNet stack over mel-spectrogram: `pre = Conv1d(spec=1025, 192, k=1)`; `enc = WN(192, k=5, dilation=1, n_layers=16, gin_channels=256)`; `proj = Conv1d(192, 2*192, k=1)` → splits to `(m_q, logs_q)`. The WaveNet (`modules.py::WN`) is the standard non-causal dilated WaveNet residual stack used throughout VITS: each layer has a dilated Conv1d, gated activation `tanh ⊙ sigmoid`, a residual Conv1d, and a skip Conv1d; speaker embedding `g` is added at every layer. **Do not implement — never invoked at inference.**

### Flow (`ResidualCouplingBlock` — default) or `TransformerCouplingBlock`

The default Mel-only configs use the residual coupling flow with `n_flow_layer=4`:

```
for i in range(4):
    z = ResidualCouplingLayer(channels=192, hidden=192, kernel=5,
                              dilation_rate=1, n_layers=4,
                              gin_channels=256, mean_only=True)(z)
    z = Flip()(z)
```

Each `ResidualCouplingLayer` splits the 192 latent channels in half: the first half is passed through a `WN(96 → 96, k=5, n_layers=4)` to produce the shift parameter (mean-only coupling, no scale, for invertibility/stability), which is added to the second half. `Flip` swaps the two halves so the next layer transforms the other half. Speaker embedding `g` is injected into every `WN`.

The transformer-flow variant (`TransformerCouplingBlock`, used if `use_transformer_flow=True` and `n_layers_trans_flow=3` per the configs) replaces each WN with a 3-layer FFT block; functionally equivalent role.

**Inference direction**: `reverse=True`, runs the flow backwards (sample from prior `z_p ~ N(m_p, exp(logs_p))` then `z = flow⁻¹(z_p)`).

### StochasticDurationPredictor (SDP)

Bert-VITS2 keeps **both** the stochastic and the deterministic duration predictor from the VITS paper and combines them at inference via `sdp_ratio`:

```
logw = SDP(x, reverse=True, noise_scale=w) * sdp_ratio
     +  DP(x)                              * (1 - sdp_ratio)
```

Default at inference is `sdp_ratio=0.2`, `noise_scale_w=0.8`, `noise_scale=0.667`, `length_scale=1.0`.

**SDP internals** (`models.py::StochasticDurationPredictor`, identical to VITS):

```
pre   = Conv1d(in=192, h=192, k=1)
convs = DDSConv(h=192, k=3, n_layers=3, p_dropout=0.5)
proj  = Conv1d(192, 192, k=1)
cond  = Conv1d(gin=256, 192, k=1)        # speaker conditioning

# main flow path (post-flows in the diagram below are training-only)
flows: [ElementwiseAffine(2),
        ConvFlow(2, 192, kernel=3, n_layers=3), Flip(),
        ConvFlow(2, 192, kernel=3, n_layers=3), Flip(),
        ConvFlow(2, 192, kernel=3, n_layers=3), Flip(),
        ConvFlow(2, 192, kernel=3, n_layers=3), Flip()]
log_flow = Log()
```

At inference (`reverse=True`): draw `z ~ N(0,1) * noise_scale_w` of shape `(B, 2, T)`, pass through the flow stack in reverse, take the first channel as `logw`. The Log flow is applied to map back to log-duration. **`DDSConv` is a depthwise-separable Conv stack with GELU + LayerNorm + dropout**; `ConvFlow` is a piecewise-rational-quadratic spline coupling layer (Neural Spline Flow, see `modules.py::ConvFlow`).

**DP internals** (`models.py::DurationPredictor`): two `Conv1d + LayerNorm + ReLU + Dropout(0.5)` blocks (kernel=3) + `Conv1d(192, 1, k=1)` projection. Speaker embedding added pre-conv. Outputs `logw` directly.

The SDP gives natural-sounding rhythm variability, the DP gives a more conservative/stable baseline; the `sdp_ratio=0.2` default biases toward the deterministic predictor.

### Generator (HiFi-GAN V1, exactly)

The post-flow latent `z` of shape `(B, 192, T_frame)` is upsampled by a **stock HiFi-GAN V1 generator** with one extra upsampling stage to handle the 512-sample hop (vs 256 in standard HiFi-GAN-V1):

- `conv_pre = Conv1d(192, 512, k=7, pad=3)`
- 5 upsample stages with rates `[8, 8, 2, 2, 2]` (product = 512 = hop_length)
- Upsample kernel sizes `[16, 16, 8, 2, 2]`
- After each upsample: **3-branch MRF** with `resblock_kernel_sizes=[3,7,11]`, dilations `[[1,3,5],[1,3,5],[1,3,5]]`, type `ResBlock1`
- `conv_post = Conv1d(ch_after_last=16, 1, k=7, pad=3)` + tanh
- Speaker conditioning: every upsample stage gets `+ Conv1d(gin=256, channels, k=1)(g)` (residual injection)

Channel progression: `512 → 256 → 128 → 64 → 32 → 16` (halved at each of the 5 stages). This is exactly the HiFi-GAN-V1 schedule documented in [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md) section "Standard HiFiGAN Generator Architecture", just with one extra stage at the end (`×2` upsample, `512 → 256` ch becomes `512 → 256 → 128 → 64 → 32 → 16`).

**Total upsampling** = `8 × 8 × 2 × 2 × 2 = 512`. With `hop_length=512` mel frames, one latent frame at the flow output becomes 512 PCM samples → 44,100 Hz output.

### Speaker control

`gin_channels=256` throughout. The configs use **learned speaker embeddings** (`use_spk_conditioned_encoder=true`, `n_speakers=256` slots reserved even when only a few are populated):

```
emb_g = nn.Embedding(n_speakers=256, gin_channels=256)
g = emb_g(sid).unsqueeze(-1)   # (B, 256, 1)
```

The same `g` is injected into:
1. **TextEncoder** (via `use_spk_conditioned_encoder=True`)
2. Every **flow** coupling layer (`ResidualCouplingLayer.gin_channels=256`)
3. Every **upsample stage** of the generator
4. Both duration predictors

**Populated speaker IDs per language** (from `data.spk2id` in each config.json):

| Variant | `spk2id` mapping | n_speakers slot count |
|---------|------------------|------------------------|
| MeloTTS-English | `EN-US:0, EN-BR:1, EN_INDIA:2, EN-AU:3, EN-Default:4` | 256 |
| MeloTTS-English-v2 | same 5 accents | 256 |
| MeloTTS-English-v3 | `EN-Newest:0` (single speaker) | 1 |
| MeloTTS-Chinese | `ZH:1` | 256 |
| MeloTTS-Spanish | `ES:0` (`SP` alias maps to the same) | 256 |
| MeloTTS-French | `FR:0` | 256 |
| MeloTTS-Japanese | `JP:0` | 256 |
| MeloTTS-Korean | `KR:0` | 256 |

Note that English-v3 (the "EN-Newest" / "ENewest" line) is **a different lineage** from the multi-accent `MeloTTS-English` — same architecture, retrained on a single newer voice. The other variants ship one voice per checkpoint despite the 256-slot embedding table.

The `ReferenceEncoder` (mel → 256-dim style) is only constructed when `n_speakers ≤ 0`, which none of the public checkpoints use. Safe to skip at inference for all shipped models.

### Phoneme tokenization (per language)

Each language has its own front-end in `melo/text/<lang>.py` exporting `text_normalize`, `g2p`, and `get_bert_feature`. The phoneme vocabulary is **shared across all languages**: the `symbols` array in config.json (180+ for English, 124 for Chinese, etc.) is a superset of every language's phoneme set plus punctuation; phoneme ids index directly into the symbol embedding table.

| Language | G2P front-end | Phoneme inventory | Tones | Notes |
|----------|---------------|-------------------|-------|-------|
| English | `text/english.py` — CMUDict lookup + eng_to_ipa fallback | ARPABET-derived; stress as `0/1/2` digits in tone channel | 0–4 | uses NLTK `cmudict`, falls back to `eng_to_ipa` |
| Chinese | `text/chinese.py` — jieba + pypinyin | initial + final pinyin tokens | 0–6 (5 tones + neutral) | tone digit goes into `tone` channel |
| Chinese-EN mix | `text/chinese_mix.py` | union of ZH + EN | union of both | for code-switched ZH input with EN words |
| Spanish | `text/spanish.py` — rule-based + lexicon | IPA | 0 | no tones |
| French | `text/french.py` — rule-based + lexicon | IPA | 0 | no tones |
| Japanese | `text/japanese.py` — pyopenjtalk → katakana → phoneme | OpenJTalk phoneme set + pitch accents | 0–1 (high/low pitch) | requires UniDic dictionary at runtime |
| Korean | `text/korean.py` — Hangul-jamo decomposition + g2pkk | jamo IPA | 0 | rule-based via `g2pkk` |

All G2P front-ends also produce:
- `phones`: list of phoneme strings → mapped to ids via the config's `symbols` table.
- `tones`: list of tone integers (same length as phones).
- `word2ph`: list of (phoneme_count_for_word_i) — used to expand BERT hidden states from word-level to phoneme-level (each word's BERT vector is repeated `word2ph[i]` times).

With `add_blank=true` (always set), a blank token (id=0) is inserted between every phoneme, doubling the sequence length + 1. The model is trained with this convention; inference must match.

**Pure-C# implementation strategy is per-language**, and the bulk of the work goes into [G2P_PHONEMIZATION.md](G2P_PHONEMIZATION.md). Highlights of what each language needs:

- **English**: CMUDict (~134k entries, public domain) + ARPABET→phoneme-symbol table + heteronym disambiguation. Plus a fallback model for OOV — see G2P notes.
- **Chinese**: a Chinese segmenter (jieba.NET exists, MIT) + pinyin dictionary (`pypinyin` data is permissively licensed) + pinyin→phoneme mapping + tone digit extraction.
- **Spanish/French**: lexicon (e.g. derived from `gruut`, MIT) + a small rule engine for OOV. Spanish phonotactics are very regular; French has more exceptions.
- **Japanese**: hardest. Either ship a port of OpenJTalk (huge — MeCab + UniDic in pure C# is a major project) or skip Japanese in v1 and document.
- **Korean**: easiest — `g2pkk` is pure Python rules over Hangul jamo decomposition; portable in a few hundred lines of C#.

### BERT auxiliary encoder (per language)

MeloTTS uses a **different pretrained BERT per language**, loaded via HuggingFace `AutoModelForMaskedLM`:

| Language | BERT model | HuggingFace path | Hidden | License notes |
|----------|------------|------------------|--------|---------------|
| English (all variants) | BERT-base uncased | `bert-base-uncased` | 768 | Apache 2.0 |
| English (alternate path) | `transfo-xl` features? — NO, always `bert-base-uncased` for the main branch | — | — | — |
| Chinese (ZH, ZH_MIX_EN) | Chinese RoBERTa large WWM | `hfl/chinese-roberta-wwm-ext-large` | **1024** | Apache 2.0 |
| Japanese | Tohoku BERT-base v3 | `tohoku-nlp/bert-base-japanese-v3` | 768 | CC-BY-SA 4.0 |
| French | dbmdz French Europeana | `dbmdz/bert-base-french-europeana-cased` | 768 | MIT |
| Spanish (ES/SP) | BETO uncased | `dccuchile/bert-base-spanish-wwm-uncased` | 768 | CC-BY 4.0 |
| Korean | Kor-BERT base | `kykim/bert-kor-base` | 768 | Apache 2.0 |

The TextEncoder always projects via `Conv1d(1024, 192, k=1)` on the main `bert` slot, and `Conv1d(768, 192, k=1)` on the `ja_bert` slot. This means:

- **Chinese always uses the 1024-channel slot** (hidden=1024 ChineseRoBERTa).
- **Everything else (EN/JA/FR/ES/KR) is hidden=768**, but the model still expects 1024 channels on the main BERT input. The implementation zero-pads or projects the 768-dim BERT up to 1024 (zero-pads the last 256 channels) before feeding into the Conv1d(1024→192). For Japanese, the BERT is passed on the `ja_bert` 768-channel slot instead, and the main `bert` is zeros.

This is fiddly: the **two-slot design** (`bert` 1024-ch + `ja_bert` 768-ch) is a hardcoded artifact of Bert-VITS2 supporting both Chinese RoBERTa-large and Japanese tohoku-BERT simultaneously. MeloTTS keeps the layout. At inference, the Python code passes:

| Language | `bert` slot (1024-ch) | `ja_bert` slot (768-ch) |
|----------|------------------------|----------------------------|
| ZH, ZH_MIX_EN | Chinese RoBERTa-large hidden | zeros |
| EN | bert-base-uncased hidden, **padded to 1024 with zeros** | zeros |
| JP | zeros | tohoku-BERT-base-v3 hidden |
| FR | French BERT hidden, padded to 1024 | zeros |
| ES | Spanish BERT hidden, padded to 1024 | zeros |
| KR | Korean BERT hidden, padded to 1024 | zeros |

(Pad direction must match the reference — inspect each `text/<lang>_bert.py` to confirm whether zeros go on the trailing or leading channels. Default in Bert-VITS2 is trailing.)

BERT input is the **orthographic text** tokenized with the BERT's own tokenizer. The hidden states from a chosen layer (usually penultimate, `output_hidden_states=True; states[-3]` in some reference forks; check per-file) are mean-pooled per word, then repeated to match phoneme count via `word2ph`. Output shape: `(B, hidden, T_phoneme)`.

### Inference pipeline (Forward Pass)

This is the path the C# implementation must mirror. From `melo/api.py::TTS.tts_to_file` and `melo/models.py::SynthesizerTrn.infer`:

```
1. SENTENCE SPLIT
   Input: "Hello world. How are you?"
   a. Language-specific sentence splitter (e.g. English uses regex+nltk;
      Chinese uses punctuation rules).
   b. For EN and ZH_MIX_EN: camelCase → spaces (regex sub).
   Output: ["Hello world.", "How are you?"]

2. PER-SENTENCE PREPROCESSING (utils.get_text_for_tts_infer)
   For each sentence:
   a. text_normalize(text)          — language-specific text normalization
                                      (numbers → words, abbreviations, etc.)
   b. phones, tones, word2ph = g2p(text)
   c. phones = ["_"] + phones + ["_"]  (BOS/EOS blank)
      tones  = [0]   + tones  + [0]
      word2ph= [1]   + word2ph+ [1]
   d. if add_blank: interleave blank token between every phoneme
                    (phones, tones become length 2N+1)
      and word2ph multiplied by 2
   e. lang_ids = [language_id] * len(phones)
   f. bert_hidden = get_bert_feature(text, word2ph)   → (1024 or 768, T_phoneme)
   g. ja_bert     = get_bert_feature_jp(...) or zeros
   h. Pack into tensors:
      x      : (1, T)  int64 — phoneme ids
      x_len  : (1,)    int64
      tones  : (1, T)  int64
      lang   : (1, T)  int64
      bert   : (1, 1024, T)
      ja_bert: (1, 768, T)

3. SPEAKER SELECTION
   sid = LongTensor([spk2id[speaker_name]])
   g   = emb_g(sid).unsqueeze(-1)     # (1, 256, 1)

4. TEXT ENCODING
   m_p, logs_p, x_mask = enc_p(x, x_len, tones, lang, bert, ja_bert, g)
   # m_p, logs_p: (1, 192, T)
   # x_mask:     (1,  1,  T)

5. DURATION PREDICTION  (sdp_ratio mixes SDP and DP)
   logw_sdp = sdp(x_encoded, x_mask, g=g, reverse=True,
                  noise_scale=noise_scale_w=0.8)
   logw_dp  = dp(x_encoded, x_mask, g=g)
   logw     = logw_sdp * sdp_ratio + logw_dp * (1 - sdp_ratio)
   w        = exp(logw) * x_mask * length_scale
   w_ceil   = ceil(w)                       # (1, 1, T) integers
   T_audio  = sum(w_ceil) over T
   y_mask   = sequence_mask(T_audio).unsqueeze(1)  # (1, 1, T_audio)

6. ALIGNMENT MATRIX & LENGTH REGULATION
   attn_mask = x_mask * y_mask.transpose(2,3)            # (1, 1, T, T_audio)
   attn      = commons.generate_path(w_ceil, attn_mask)  # one-hot expand
                                                         # shape (1, 1, T_audio, T)
   m_p_expanded   = matmul(attn.squeeze(1), m_p.transpose(1,2)).transpose(1,2)
   logs_p_expanded= matmul(attn.squeeze(1), logs_p.transpose(1,2)).transpose(1,2)
   # both: (1, 192, T_audio)

7. PRIOR SAMPLE
   z_p = m_p + randn_like(m_p) * exp(logs_p) * noise_scale   # noise_scale=0.667

8. FLOW INVERSE
   z   = flow(z_p, y_mask, g=g, reverse=True)               # (1, 192, T_audio)

9. GENERATOR (HiFi-GAN V1)
   audio = dec(z[:, :, :max_len], g=g)                      # (1, 1, T_audio*512)

10. POSTPROCESS
    audio = audio.squeeze().cpu().numpy()
    sf.write(out_path, audio, samplerate=44100)
```

Default inference scalars in `api.py`:

| Param | Default | Role |
|-------|---------|------|
| `speed` | 1.0 | `length_scale = 1/speed` |
| `sdp_ratio` | 0.2 | mix of stochastic vs deterministic duration |
| `noise_scale` | 0.667 | prior sample temperature |
| `noise_scale_w` | 0.8 | SDP latent noise scale |
| `quiet` | False | progress bar suppression |

If multiple sentences: concatenate per-sentence audio with a configurable silence (~0.05–0.10 s of zeros) between them.

### Output sample rate

**All public MeloTTS variants are 44,100 Hz**, `hop_length=512`, `filter_length=2048` (n_fft), 1025 spec bins. The earlier rumour that some variants are 22.05 kHz is incorrect for the public releases — every config.json on the org has `"sampling_rate": 44100`.

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

## Algorithm Steps

### Full Inference Pipeline (per sentence)

```
1. TEXT NORMALIZATION (per-language, melo/text/<lang>.py::text_normalize)
   - Number expansion ("123" -> "one hundred twenty three")
   - Abbreviation expansion
   - Punctuation normalization
   - Case handling (lowercasing for ZH/EN as appropriate)
   - Language-specific: ZH does jieba segmentation here; JP does mecab; etc.

2. G2P (per-language, melo/text/<lang>.py::g2p)
   Output: phones (list[str]), tones (list[int]), word2ph (list[int])

3. PHONEME ID MAPPING
   - Map each phone string to int via config.symbols
   - Prepend/append "_" (blank, id=0)
   - Interleave blank id=0 between every pair (add_blank=true)
   - Replicate the same transformations for tones (pad with 0) and word2ph

4. BERT FEATURE EXTRACTION (per-language BERT model)
   - Run BERT on the original normalized text string
   - Extract hidden states (typically penultimate layer mean-pooled per word)
   - Expand from word-level to phoneme-level via word2ph
   - Place in correct slot:
       Chinese  -> bert (1024 ch)
       Japanese -> ja_bert (768 ch), bert = zeros
       Others   -> bert with last 256 channels zero-padded
   - For non-active slot: tensor of zeros of correct shape

5. PACK INPUTS
   x      = LongTensor([phone_ids])         # (1, T)
   x_len  = LongTensor([T])                  # (1,)
   tones  = LongTensor([tone_ids])           # (1, T)
   lang   = LongTensor([lang_ids])           # (1, T)
   bert   = FloatTensor(bert_features)       # (1, 1024, T)
   ja_bert= FloatTensor(ja_bert_features)    # (1, 768,  T)
   sid    = LongTensor([speaker_id])         # (1,)

6. SPEAKER EMBED
   g = emb_g(sid).unsqueeze(-1)              # (1, 256, 1)

7. TEXT ENCODE
   x_enc = embedding(x) * sqrt(192)
         + tone_emb(tones)
         + lang_emb(lang)
         + Conv1d(1024,192,1)(bert)
         + Conv1d( 768,192,1)(ja_bert)
   x_enc = transformer_encoder(x_enc, mask, g)   # 6 layers, 2 heads, FFN=768
   stats = Conv1d(192, 384, 1)(x_enc)
   m_p, logs_p = chunk(stats, 2, dim=1)          # (1, 192, T) each
   x_mask = sequence_mask(x_len)                  # (1, 1, T)

8. DURATION (mix SDP + DP)
   logw_sdp = sdp.forward(x_enc, x_mask, g=g, reverse=True,
                          noise_scale=noise_scale_w=0.8)
   logw_dp  = dp.forward(x_enc, x_mask, g=g)
   logw     = logw_sdp * 0.2 + logw_dp * 0.8
   w        = exp(logw) * x_mask * length_scale  # (1, 1, T)
   w_ceil   = ceil(w)
   T_audio  = sum(w_ceil)
   y_mask   = sequence_mask(T_audio).unsqueeze(1)

9. ALIGN / EXPAND
   attn = generate_path(w_ceil, attn_mask)        # (1, 1, T_audio, T)
   # equivalent to: for each phoneme i, repeat its (m_p[:,:,i], logs_p[:,:,i])
   # exactly w_ceil[0,0,i] times along the audio-frame axis
   m_p_exp    = matmul(attn.squeeze(1), m_p.transpose(1,2)).transpose(1,2)
   logs_p_exp = matmul(attn.squeeze(1), logs_p.transpose(1,2)).transpose(1,2)
   # both: (1, 192, T_audio)

10. SAMPLE PRIOR
    eps  = randn(shape_of(m_p_exp))               # deterministic via seeded RNG
    z_p  = m_p_exp + eps * exp(logs_p_exp) * noise_scale  # noise_scale=0.667

11. FLOW INVERSE
    z = flow.forward(z_p, y_mask, g=g, reverse=True)      # (1, 192, T_audio)
    # flow: 4 x [ResidualCouplingLayer(WN(192,k=5,nl=4,gin=256)) + Flip]
    # reverse: run layers in reverse order, each layer's reverse mode

12. GENERATE (HiFi-GAN V1)
    audio = dec.forward(z, g=g)                            # (1, 1, T_audio*512)
    # pre  : Conv1d(192,512,k=7)
    # for r,k in zip([8,8,2,2,2],[16,16,8,2,2]):
    #   x = LeakyReLU(x, 0.1)
    #   x = ConvTranspose1d(ch, ch//2, k, stride=r, pad=(k-r)//2)(x)
    #   x = x + cond_layer[i](g)                # speaker injection
    #   x = MRF(x)                              # avg of 3 ResBlock1 with k=[3,7,11]
    # post : LeakyReLU + Conv1d(16,1,k=7) + tanh
    audio = audio.squeeze().numpy()

13. (multi-sentence) CONCATENATE
    pieces.append(audio)
    pieces.append(silence_samples)   # default ~0.05 s of zeros
    final = concat(pieces[:-1])      # trim trailing silence
    write_wav(out_path, final, sr=44100)
```

### Duration mixing detail

`sdp_ratio=0.2` means the deterministic predictor dominates (0.8 weight). The SDP contributes 20% — enough for natural variability without unstable rhythms. Setting `sdp_ratio=1.0` gives full SDP (most natural, most varied); `sdp_ratio=0.0` gives full DP (most stable, slightly robotic). The reference defaults to 0.2 because most users want stability.

### `generate_path` detail

The `commons.py::generate_path(duration, mask)` function takes a `(B,1,T_in)` integer duration tensor and a `(B,1,T_out,T_in)` attention mask, and returns a `(B,1,T_out,T_in)` one-hot tensor where each row `t` (output frame) is 1 exactly at the input phoneme it aligns to. Implementation is a cumulative-sum trick: `cum = cumsum(duration, dim=-1); path = (cum > arange(T_out)[None,None,:,None]) & (cum_prev <= arange(T_out))`. For C# we implement as a `Repeat` directly — see implementation note 8 above.

## Open Questions

- [ ] Exact pretrained BERT layer used (penultimate? mean of last 4?) — must be confirmed by reading each `melo/text/<lang>_bert.py` file when implementing. Wrong layer = wrong BERT features = degraded but not catastrophic output.
- [ ] Whether English (and other non-ZH, non-JP) BERTs zero-pad on the leading or trailing channels of the 1024-ch slot. Default Bert-VITS2 convention is trailing-zeros; verify per file.
- [ ] Parameter count breakdown — the 67M estimate above is from module-by-module dim multiplication; should be confirmed with a `model.parameters().numel()` count against the reference once we have weight loading working.
- [ ] Whether `MeloTTS-English-v2` and v3 use different transformer-flow vs residual-coupling settings. v3's config has `n_layers_trans_flow: 3` set, but the model's `use_transformer_flow` flag is not visible from the config snippet — confirm from `models.py` defaults.
- [ ] Whether MeloTTS-Chinese's `num_tones=11` includes the neutral tone (mandarin has 4 lexical tones + neutral, but pinyin tone digits go 0-5 — config has 11, suggesting it covers some additional pitch-accent markers).
- [ ] Is there an "EN-Newest" voice in MeloTTS-English (multi-accent)? The install docs list 5 accents; the API code references EN-Newest. They may be the v3 model loaded under the multi-accent ID space — unclear. Inspect `api.py::TTS.__init__`'s checkpoint download URL switch.
- [ ] Whether we should ship all variants in v1 or stage them. Recommendation: ship English-v3 (smallest dep — only `bert-base-uncased`), then Chinese (largest BERT but biggest user demand), then Spanish/French (small BERTs + simpler G2P), then Korean, then Japanese last (OpenJTalk port is a major undertaking).

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

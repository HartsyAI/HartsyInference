# GPT-SoVITS — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: HartsyInference.Audio (GPT-SoVITS pipeline)

> **Stub.** The narrative walkthrough, restated pseudocode and resolved open questions were
> removed on 2026-08-06 — this model is built and verified, so the C# is the source of truth for
> *how it works*. What remains is what the code cannot tell you: upstream provenance, reference
> constants to diff a suspect port against, where implementations disagree, and bring-up traps.
> Full history is in git. Parity evidence: `docs/Checklists/PARITY_VERIFICATION.md`.

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

5. **GPT KV cache**: the dominant cost is the 24-layer transformer × ~125-1500 tokens. Implement proper KV cache (preallocated tensor of shape `(batch, n_layers, 2, n_heads, max_len, head_dim)`) so each step is O(seq_len) attention not O(seq_len²). This is the same KV-cache pattern the native `HartsyInference.LLM` package uses; consider extracting a shared `KvCache` type into `HartsyInference.Core`.

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

---

## C# build status (2026-06-20)

Built under `src/HartsyInference.Audio/Models/Hubert/` + `Models/GptSoVits/`. **Corrections applied** vs the
original doc above: the T2S transformer is **post-LayerNorm with no final LN** (`norm_first=False`); cn-HuBERT
taps **`last_hidden_state`** (not an intermediate layer); SoVITS `enc_q` is **n_layers=16**.

- [x] **`Hubert`** — 7-conv extractor (GroupNorm after conv 0, GELU) + feature projection + grouped pos-conv
  embed (k128/g16) + 12 post-LN transformer layers → `last_hidden_state [1,768,T]`. Reuses `IBackend` conv/
  norm/SDPA + `DiaHeads`. Synthetic-forward verified. **Shared with RVC.**
- [x] **`Text2Semantic`** (T2S) — post-LN GPT (512/24L/16h/FFN2048/ReLU), sinusoidal `alpha` positions, biased
  fused-QKV, text-bidirectional/audio-causal mask, top-k + rep-penalty(1.35) AR over semantic vocab 1025
  (EOS 1024). Reuses `NucleusSampler`. Synthetic-forward verified.
- [x] **`SoVitsSynthesizer`** — semantic VQ dequant (codebook `[1024,768]`, ×2 nearest) → ssl_proj → prior →
  sample z_p → **reused g-conditioned `VitsFlow` + `VitsHiFiGan`** (32 kHz). Synthetic-forward verified.
- [ ] **Staged:** full SoVITS `enc_p` (dual attention encoders + **MRTE** cross-attn replacing the structural
  conv prior), the `ref_enc` MelStyleEncoder (produces `ge` — currently caller-supplied), Chinese-RoBERTa BERT
  frontend + phoneme tokenizer, and exact `quantizer.vq.*` / ckpt-`"weight"` key reconciliation.

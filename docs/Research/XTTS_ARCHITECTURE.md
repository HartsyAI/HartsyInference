# XTTS-v2 — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: HartsyInference.Audio (XTTS-v2 pipeline)

## Summary

XTTS-v2 (Coqui, Sept 2023) is a multilingual zero-shot voice-cloning TTS that clones a target speaker's voice from as little as 6 seconds of reference audio, in any of 17 languages, with cross-language transfer (e.g. clone an English speaker into German). It is the most widely deployed open TTS as of 2026 despite Coqui shutting down in Jan 2024 — both the original [`coqui-ai/TTS`](https://github.com/coqui-ai/TTS) repo (archived) and the maintained Idiap fork [`idiap/coqui-ai-TTS`](https://github.com/idiap/coqui-ai-TTS) ship the same checkpoint, distributed via the HuggingFace repo [`coqui/XTTS-v2`](https://huggingface.co/coqui/XTTS-v2) under the Coqui Public Model License (CPML, non-commercial).

Architecturally XTTS-v2 is a four-component pipeline. A small **conditioning encoder** turns a 6+ second reference clip into a fixed-length speaker latent (and a separate 512-dim "speaker embedding" from a pretrained H/ASP speaker-verification net). A **GPT-2-style autoregressive transformer** (~443M params, 30 layers, d_model=1024, 16 heads) takes BPE text tokens (~6.6k vocab, per-language prefix tokens like `[en]`) plus the speaker latent and autoregressively predicts a stream of discrete **mel-codec tokens** drawn from a 1024-entry VQ-VAE codebook trained on 80-bin mel spectrograms at 22.05 kHz. A small **GPT→latent decoder** (a 6-layer Perceiver-style block called `gpt_inference_head`) converts the predicted mel-codec token sequence and speaker conditioning into a continuous latent stream. Finally a **HiFiGAN-based waveform decoder** (with speaker-embedding conditioning injected into its residual blocks) upsamples those latents directly to 24 kHz waveform — XTTS-v2 does NOT produce an intermediate mel spectrogram at inference; the HiFiGAN consumes the GPT latent stream directly. Total released model is ~1.86 GB FP32, ~931 MB FP16.

The model paper (["XTTS: a Massively Multilingual Zero-Shot Text-to-Speech Model", arXiv:2406.04904](https://arxiv.org/abs/2406.04904)) and the model card document the design lineage from Tortoise-TTS (the GPT+mel-token+vocoder pattern is Betker's) with three key changes: (1) cross-lingual training with language conditioning, (2) replacement of Tortoise's expensive diffusion+UnivNet stack with a single HiFiGAN that consumes GPT latents directly, and (3) chunked streaming. This file covers the architecture; the vocoder back-end is documented in [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md), the BPE/SentencePiece tokenizer machinery in [TOKENIZERS.md](TOKENIZERS.md), and mel preprocessing in [MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md).

Sources: [arXiv:2406.04904](https://arxiv.org/abs/2406.04904), [coqui-ai/TTS (archived)](https://github.com/coqui-ai/TTS), [idiap/coqui-ai-TTS fork](https://github.com/idiap/coqui-ai-TTS), [coqui/XTTS-v2 HF model](https://huggingface.co/coqui/XTTS-v2), [Coqui XTTS docs](https://docs.coqui.ai/en/latest/models/xtts.html), [Tortoise-TTS](https://github.com/neonbjb/tortoise-tts) (architectural ancestor).

## Detailed Findings

### Variants and History

| Version | Released | Languages | Notes |
|---------|----------|-----------|-------|
| **XTTS-v1** | Sept 2023 | 13 (en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn) | Initial release. Smaller GPT, mel-spectrogram intermediate, separate vocoder. Largely superseded within weeks. |
| **XTTS-v1.1** | Sept 2023 | 14 (+ ja) | Stability fixes, Japanese support, BPE tokenizer overhaul. |
| **XTTS-v2.0.0** | Nov 2023 | 16 (+ hu, ko) | Streaming inference, ~10% latency reduction, GPT latents fed directly to HiFiGAN (no mel intermediate). |
| **XTTS-v2.0.1** | Nov 2023 | 16 | Speaker-conditioning bug fix. |
| **XTTS-v2.0.2** | Dec 2023 | 17 (+ hi) | Hindi added, training improvements. **This is the canonical release** (`coqui/XTTS-v2` HF revision `v2.0.2`). |
| **XTTS-v2.0.3** | Dec 2023 | 17 | Last official release before shutdown. Pronunciation and prosody fixes. |
| Community "v3" efforts | 2024-2026 | varies | No official v3. Community work focuses on (a) re-finetunes on cleaner data (e.g. `daswer123/xtts-finetune-webui`), (b) ONNX/CT2 exports, (c) the **Idiap fork** which is a maintenance fork, not an architecture change. **There is no v3 checkpoint that changes the architecture.** |
| **Forks of interest** | | | [`idiap/coqui-ai-TTS`](https://github.com/idiap/coqui-ai-TTS) (active maintenance), [`daswer123/xtts-api-server`](https://github.com/daswer123/xtts-api-server) (API wrapper, popular), [`erew123/alltalk_tts`](https://github.com/erew123/alltalk_tts) (gradio + finetuning), [`OpenVoiceOS/ovos-tts-plugin-xtts`](https://github.com/OpenVoiceOS/ovos-tts-plugin-xtts) (deployment). |

**Officially supported languages in v2.0.2 (17 total)**: English (`en`), Spanish (`es`), French (`fr`), German (`de`), Italian (`it`), Portuguese (`pt`), Polish (`pl`), Turkish (`tr`), Russian (`ru`), Dutch (`nl`), Czech (`cs`), Arabic (`ar`), Chinese (`zh-cn`), Japanese (`ja`), Hungarian (`hu`), Korean (`ko`), Hindi (`hi`).

**Parameter count**: ~443M trainable parameters total. Breakdown (from `model_args` in `config.json` and HF safetensors index):

| Component | Params (approx.) | Role |
|-----------|------------------|------|
| GPT-2 backbone | ~349M | Autoregressive mel-token predictor |
| GPT conditioning Perceiver | ~12M | Reference-audio → fixed-length cond latent for GPT |
| Text embeddings | ~6.8M | 6681 vocab × 1024 dim |
| Mel-token embeddings | ~1.1M | 1026 vocab × 1024 dim |
| Mel-VQ codebook + projection | ~3M | 1024 × 80 (codebook) + linear heads |
| HiFiGAN decoder | ~70M | GPT latent → 24 kHz waveform |
| H/ASP speaker encoder (bundled but frozen) | ~7M | Reference audio → 512-dim verification embedding for HiFiGAN |
| **Total** | **~443M** | (1.86 GB FP32, the "~440M" commonly cited in the paper) |

### Overall Pipeline

```
                   Reference audio (6+ s, 22.05 kHz mono)
                            |
              +-------------+-------------+
              |                           |
              v                           v
   Mel spectrogram (80 bins)     H/ASP speaker encoder
              |                           |
              v                           v
   GPT conditioning Perceiver    Speaker embedding (512-d)
              |                           |
              v                           |
   Conditioning latent (32 tokens × 1024) |
              |                           |
              +-----+ Text + lang token   |
                    |     |               |
                    v     v               |
              +----------------+          |
              |  GPT-2 (30L)   |          |
              |   AR decoder   |          |
              +----------------+          |
                    |                     |
                    v                     |
       Mel-codec tokens (1024-vocab)      |
                    |                     |
                    v                     |
       GPT inference head (latents)       |
                    |                     |
                    v                     |
              +-------------+              |
              |  HiFiGAN    |<-------------+
              |  decoder    |   conditioning via residual blocks
              +-------------+
                    |
                    v
       24 kHz waveform (float32)
```

### Tokenizer (Text)

XTTS-v2 uses a **single shared BPE tokenizer** for all 17 languages, NOT per-language tokenizers concatenated as commonly described — that was the v1 approach. The v2 tokenizer is a single HuggingFace `tokenizers` `Tokenizer` (BPE model) stored in `vocab.json` (~360 KB, the file is the full `Tokenizer.to_str()` JSON, not just vocab+merges as the OpenAI GPT-2 split format).

- **Vocab size**: 6,681 tokens (model_args.gpt_number_text_tokens = 6681 in the config).
- **Model type**: `BPE` with `pre_tokenizer = Whitespace`, `decoder = ByteLevel`, `normalizer = NFC`.
- **Special tokens at sequence start**: A language token of the form `[<lang>]` is prepended. The vocab contains: `[en] [es] [fr] [de] [it] [pt] [pl] [tr] [ru] [nl] [cs] [ar] [zh-cn] [ja] [hu] [ko] [hi]`. Plus `[START]` (BOS, id 261), `[STOP]` (EOS, id 0), and `[SPACE]`.
- **Text normalization layer (in Python wrapper, not in the tokenizer)**: number expansion (e.g. "2024" → "two thousand twenty-four"), currency expansion, abbreviation expansion. Per-language. The Python class is `VoiceBpeTokenizer` in `TTS/tts/layers/xtts/tokenizer.py`. For non-Latin scripts the normalizer also does romanization for some languages (e.g. `pypinyin` for Chinese, `cutlet` for Japanese) **before BPE encoding** — so the BPE sees Pinyin/romaji, not Han characters. **This is the dirtiest part of the pipeline to port** (see "C# Implementation Notes" below).
- **Per-language token-length limits** enforced in the wrapper (e.g. English ~250 chars per chunk, Japanese ~100). The model truncates at `max_text_tokens = 402`.
- **Sequence layout (text side)**: `[lang_tok] BPE(text...) [START]`. The GPT then generates mel tokens until it emits the mel-side EOS.

### GPT-2 Backbone

The GPT is a standard GPT-2 decoder-only causal Transformer, implemented in `TTS/tts/layers/xtts/gpt.py` as the `GPT` class wrapping a HuggingFace `GPT2Model`. Config from the released `config.json` `model_args`:

| Parameter | Value | Source field |
|-----------|-------|--------------|
| `gpt_max_audio_tokens` | 605 | max audio tokens per generation |
| `gpt_max_text_tokens` | 402 | max text tokens per input |
| `gpt_max_prompt_tokens` | 70 | max conditioning prompt length |
| `gpt_layers` | 30 | n_layer |
| `gpt_n_model_channels` | 1024 | n_embd / hidden size |
| `gpt_n_heads` | 16 | n_head |
| `gpt_number_text_tokens` | 6681 | text vocab size |
| `gpt_start_text_token` | 261 | text [START] id |
| `gpt_stop_text_token` | 0 | text [STOP] id |
| `gpt_num_audio_tokens` | 1026 | mel-codec vocab (1024 codes + START + STOP) |
| `gpt_start_audio_token` | 1024 | mel [START] |
| `gpt_stop_audio_token` | 1025 | mel [STOP] |
| `gpt_code_stride_len` | 1024 | frames per mel token at training time |
| `gpt_use_masking_gt_prompt_approach` | true | training detail |
| `gpt_use_perceiver_resampler` | true | use Perceiver for cond instead of mean-pool |
| `gpt_num_audio_tokens_perceiver` | 32 | output length of Perceiver |

Standard GPT-2 details:
- Pre-norm LayerNorm (`epsilon=1e-5`), GELU activation, learned positional embeddings of shape `(max_seq_len, 1024)` where `max_seq_len = max_prompt + max_text + max_audio ≈ 70 + 402 + 605 = 1077`.
- Each layer: `LN -> MultiHeadSelfAttention(causal, 16 heads × 64 dim, bias=True) -> residual -> LN -> MLP(1024 -> 4096 -> 1024) -> residual`.
- Final `LayerNorm` then **two separate prediction heads** sharing the trunk: `text_head: Linear(1024, 6681)` for text-prediction loss during training, and `mel_head: Linear(1024, 1026)` for the mel-codec tokens used at inference. **At inference only `mel_head` is used.**
- Token-type / sequence-position layout fed to the GPT (see `gpt.py:GPT.forward` and `get_logits`):

```
[ cond_latent (32) | text_emb(text_tokens) + text_pos | mel_emb(mel_tokens) + mel_pos ]
        |                       |                                  |
   from Perceiver       text_embeddings + text_positional   mel_embeddings + mel_positional
```

Crucially there are **separate positional encodings** for the conditioning, text, and mel-token regions, and **separate embedding tables** for text vs mel tokens. The "shared trunk" is just the 30-layer transformer + final LN.

### Mel-VQ Codebook (DVAE)

XTTS-v2 does NOT predict raw waveform tokens (it is not EnCodec/SoundStream based). Instead it predicts tokens from a small **Discrete VAE over 80-bin mel spectrograms**, trained jointly with the GPT. Code: `TTS/tts/layers/tortoise/dvae.py`, class `DiscreteVAE`.

- **Codebook size**: 1024 entries (config: `num_tokens = 1024`).
- **Code dim**: 512 (config: `codebook_dim = 512`).
- **Input**: 80-bin mel spectrogram at 22.05 kHz with hop=256, window=1024, n_fft=1024 — see "Mel Parameters" below.
- **Time compression ratio**: ~1024 audio samples per mel token at 22.05 kHz (the `gpt_code_stride_len = 1024` field). So 1 second of speech ≈ 22 mel tokens.
- **Encoder**: stack of 1-D conv blocks with stride-2 downsamplers, total downsampling factor 4 along the mel-time axis (80 mel-frames per second → 20 tokens per second after VQ); residual blocks with GroupNorm; final projection to 512-d, then nearest-codebook lookup with EMA codebook updates (during training only — frozen at inference).
- **Decoder (DVAE side)**: symmetric upsampling stack to reconstruct mel — but **the DVAE decoder is not used at inference**. It exists in the checkpoint (under the `dvae` key in `model.pth`) for two reasons: (a) it was used to tokenize the training corpus, and (b) the codebook embedding itself is read out for the GPT to consume. At inference only the codebook embedding matrix (shape `[1024, 512]`) is needed.

**Inference flow with the codebook**: the GPT predicts `mel_token_id ∈ [0, 1023]` (plus 1024=START, 1025=STOP). These IDs are **embedded via the GPT's own mel-token embedding `mel_embedding: Embedding(1026, 1024)`** — NOT via the DVAE codebook directly. The DVAE codebook is consumed only by the `gpt_inference_head`/HiFiGAN path indirectly: the GPT's final hidden states (1024-dim) are returned directly as the latents, and the mel tokens themselves are used only for early stopping (STOP token detection) and KV-cache management. **This is the v2 simplification over v1**: v1 decoded mel tokens → mel spectrogram → vocoder; v2 sends GPT latents straight to HiFiGAN.

### Conditioning Encoder (Speaker Latent for the GPT)

The reference audio is encoded twice, by two independent components, producing **two different speaker representations**. Both are bundled in the checkpoint.

**(a) GPT conditioning latent — Perceiver resampler** (`TTS/tts/layers/xtts/perceiver_encoder.py`):

- Input: mel spectrogram of the reference clip (80 bins, 22.05 kHz, see "Mel Parameters").
- A small Conv1D-then-Transformer stack (`gpt_conditioning_encoder`) produces a variable-length sequence of 1024-dim vectors.
- A **Perceiver IO resampler** with 32 learned latent queries cross-attends to the variable-length sequence, producing a fixed `(32, 1024)` "conditioning latent". This is what the GPT consumes at positions 0–31 of its input sequence.
- The reference can be 6–30 s (longer is better but capped). Multiple reference clips can be concatenated and averaged at the latent level (`get_conditioning_latents()` accepts a list).

**(b) HiFiGAN speaker embedding — H/ASP (ECAPA-TDNN style)** (`TTS/tts/layers/xtts/hifigan_decoder.py:ResNetSpeakerEncoder`):

- A pretrained speaker-verification network (the "H/ASP" model from VoxCeleb training). Frozen during XTTS training.
- Input: raw waveform 16 kHz mono (resampled from the reference).
- Output: **512-dim** L2-normalized speaker embedding (a `(1, 512)` vector). This is the conventional speaker-encoder output you find in modern speaker-verification systems.
- This embedding is injected into the HiFiGAN residual blocks via FiLM-style conditioning (see HiFiGAN section).

So at inference, `get_conditioning_latents(audio_path)` returns a `(gpt_cond_latent: (32, 1024), speaker_embedding: (512,))` pair. Both are cached and reused across multiple `inference()` calls with the same reference voice. This is the entire "voice clone" — XTTS does NOT need finetuning to add a new speaker; it is purely zero-shot via this latent extraction.

### HiFiGAN Waveform Decoder

XTTS-v2's HiFiGAN is a modified version of the standard HiFiGAN V1 generator (Kong et al., 2020 — see [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md) for the baseline architecture) with two XTTS-specific changes:

1. **Input is GPT latents (1024-dim, ~22 Hz), not mel spectrogram (80 bins, ~86 Hz).** A pre-conv `Conv1d(1024 → 512, kernel=7)` projects the GPT latents into the HiFiGAN hidden dimension.
2. **Speaker conditioning via FiLM in every residual block.** The 512-dim speaker embedding is projected to per-block `(scale, shift)` and modulates each ResBlock1's activations. This is what makes the same GPT latent stream produce a different voice for different speakers.

**Upsampling**: total upsampling factor must match the GPT latent rate → 24 kHz audio rate. GPT latents are at `sample_rate / gpt_code_stride_len = 22050 / 1024 ≈ 21.53 Hz`, but the model is configured to output at 24 kHz. Actual `upsample_rates = [8, 8, 2, 2]` (factor 256), `upsample_kernel_sizes = [16, 16, 4, 4]`, `resblock_kernel_sizes = [3, 7, 11]`, `resblock_dilation_sizes = [[1,3,5], [1,3,5], [1,3,5]]` — standard HiFiGAN V1. The mismatch between 21.53 Hz latent rate × 256 ≈ 5512 Hz and 24000 Hz target is resolved by a final resampling layer (a learned `ConvTranspose1d`) — see `hifigan_decoder.py:HifiDecoder`.

**Output sample rate**: 24,000 Hz, mono, float32 in approximately `[-1, 1]`.

**FP16 caveat**: HiFiGAN's `weight_norm` layers and the FiLM modulations are numerically sensitive; the official inference path keeps the HiFiGAN in FP32 even when the GPT is in FP16. We should follow the same convention.

### Inference Pipeline (Step by Step)

The full forward at inference, per `TTS/tts/models/xtts.py:Xtts.inference()`:

```
1. PREPROCESS REFERENCE AUDIO (once per speaker)
   a. Load reference wav -> resample to 22050 Hz (mono float32).
   b. Compute mel spectrogram (80 bins, n_fft=1024, hop=256, win=1024, fmin=0, fmax=8000).
   c. Conv stack -> Perceiver(32 queries) -> gpt_cond_latent shape (32, 1024).
   d. Independently: resample to 16 kHz -> H/ASP speaker encoder -> speaker_embedding (512,) L2-normed.
   Cache the (gpt_cond_latent, speaker_embedding) pair.

2. TEXT PREPROCESSING (per utterance)
   a. Per-language normalizer (numbers, currency, abbreviations).
   b. For zh: pypinyin romanization. For ja: cutlet romanization. For ko: jamo decomposition.
   c. BPE encode -> token IDs.
   d. Prepend language token id (e.g. [en] = 6679 or wherever it lives in vocab).
   e. Append [START] (id 261).
   f. Truncate to gpt_max_text_tokens=402.

3. GPT AUTOREGRESSIVE DECODE
   Input layout (no KV cache, t=0):
     embeds = concat(
       gpt_cond_latent,                              # (32, 1024)
       text_embedding(text_tokens) + text_pos_emb,   # (T_text, 1024)
       mel_embedding([gpt_start_audio_token=1024])   # (1, 1024)  start of mel
       + mel_pos_emb[0]
     )
   Then iteratively:
     for t in 0 .. gpt_max_audio_tokens=605:
       hidden = GPT2(embeds, kv_cache).last_hidden_state[-1]   # (1024,)
       latents.append(hidden)
       logits = mel_head(hidden)                                # (1026,)
       apply repetition_penalty, length_penalty, top_k, top_p
       next_tok = sample(logits) (or argmax if temperature -> 0)
       if next_tok == gpt_stop_audio_token (1025): break
       embeds = mel_embedding(next_tok) + mel_pos_emb[t+1]
   Output: latents tensor (T_mel, 1024) where T_mel <= 605.

4. WAVEFORM DECODE
   a. latents = latents.transpose(0,1).unsqueeze(0)   # (1, 1024, T_mel)
   b. waveform = hifigan_decoder(latents, speaker_embedding)
      - conv_pre: 1024 -> 512
      - 4 upsampling stages × {transposed conv + MRF × 3 with FiLM(speaker_emb)}
      - conv_post -> tanh
      - final resample to exactly 24000 Hz
   Output: (1, 1, N_samples) float32, ~[-1, 1].

5. POSTPROCESS
   a. Trim leading/trailing silence (optional, in CLI wrapper).
   b. Write to WAV at 24000 Hz.
```

**Default sampling parameters** (from `XttsArgs` / the `model.inference()` signature):
- `temperature = 0.75`
- `length_penalty = 1.0` (applied to log-probs, not to logits directly — actually multiplies the EOS log-prob to discourage early stopping)
- `repetition_penalty = 10.0` (note: unusually high vs LLMs because the mel-token space is small and prone to loops)
- `top_k = 50`
- `top_p = 0.85`
- `num_beams = 1` (greedy/sampling; beam search is supported but rarely used)
- `do_sample = True`
- `enable_text_splitting = True` (Python-side chunking by sentence/punctuation, then concat audio with a small crossfade)

### Streaming

Streaming is implemented in `Xtts.inference_stream()` and is the marquee v2 feature. The strategy is **chunked GPT output piped into chunked HiFiGAN decoding**, with overlap-add at chunk boundaries to hide HiFiGAN edge artifacts.

**Default streaming knobs** (from the `inference_stream` signature):
- `stream_chunk_size = 20` mel tokens per chunk (default; many wrappers lower this to 4-8 for first-token latency).
- `overlap_wav_chunks = 1024` samples of crossfade between adjacent decoded audio chunks.
- `enable_text_splitting = True` — text is also pre-split by sentence so the GPT can emit `[STOP]` between sentences and the stream restarts (this hides intra-sentence buffering).

**Algorithm**:
```
1. Start GPT autoregressive loop as in non-streaming.
2. After every `stream_chunk_size` mel tokens generated:
   a. Slice the latest (stream_chunk_size + overlap_pad) GPT latents.
   b. Run HiFiGAN on that slice (re-runs the convs over a small window — cheap, no incremental conv state is maintained).
   c. Take the central (stream_chunk_size * 1024 / 22050 * 24000) samples; crossfade `overlap_wav_chunks` samples with the previous chunk.
   d. Yield the audio chunk (an async/iterator yield in Python).
3. Continue GPT until [STOP] or max_audio_tokens.
4. Yield one final tail chunk.
```

**First-token latency** depends almost entirely on `stream_chunk_size`. Reported numbers on a single RTX 4090:
- `stream_chunk_size=20`: ~350-450 ms time to first audio chunk.
- `stream_chunk_size=8`: ~150-200 ms TTFA.
- `stream_chunk_size=4`: ~80-120 ms TTFA but more HiFiGAN edge artifacts (the crossfade region becomes a larger fraction of the emitted audio).
- **Sub-200 ms TTFA is achievable** on consumer GPUs with `stream_chunk_size=4-8`, which is what makes XTTS-v2 viable for conversational agents.

Real-time factor (RTF) for streaming on RTX 4090 FP16: ~0.15–0.25 (4–7x real-time after the first chunk).

**Important**: HiFiGAN is re-evaluated from scratch on each slice — there is no incremental conv state maintained in the official implementation. This is wasteful but simple and the overlap-add prevents discontinuities. A pure-C# port can do better with a true streaming-conv implementation (carry the last `kernel - 1` samples of each conv layer's state).

### Cross-Language Voice Cloning

A key XTTS-v2 capability: clone a speaker who only speaks English in their reference audio, then generate German (or any of the 17 supported languages) in that voice.

**How it works**:
1. The speaker representations (Perceiver `gpt_cond_latent` and the H/ASP `speaker_embedding`) are **language-agnostic** — the H/ASP encoder is a speaker-verification model that was trained to extract identity-only features and discard linguistic content, and the GPT conditioning Perceiver receives mel features that, after training on a multilingual corpus, the model has learned to project into a similar language-neutral latent space.
2. The **language token** prepended to the BPE text sequence is what tells the GPT which phonetic distribution to generate. The GPT was trained to produce mel tokens matching the language token regardless of the speaker latent's source language.
3. Therefore: at inference, you can mix any (reference voice, language token, text) triple.

**Quality caveats**:
- Cross-language quality is best for source/target language pairs that share phonemes (e.g. EN→DE is excellent, EN→JA is decent, AR→ZH is rough).
- The speaker's accent in the target language is dictated by the GPT's training distribution — XTTS will produce a native-sounding German with the speaker's vocal timbre, NOT a German-accented version of their English voice. This is usually what users want but is sometimes surprising.
- Languages with non-Latin scripts (zh, ja, ko, ar, hi) go through the romanization preprocessor before BPE; cross-language cloning into these works fine as long as the user passes the script in the form the normalizer expects (Han chars are auto-pinyinized; kanji+kana is auto-romajized).

### Reference Audio Requirements

Documented and empirical guidance:
- **Length**: 6 seconds minimum (the Perceiver needs enough mel frames to extract meaningful cond), 10-30 seconds optimal, longer than 30 s is truncated/sub-sampled.
- **Channels**: mono (stereo is auto-mixed-down).
- **Sample rate**: any (the loader resamples to 22.05 kHz for the Perceiver path and 16 kHz for the H/ASP path).
- **Cleanliness**: single speaker, low background noise, no reverb, no music. Strongly affects clone quality — XTTS will faithfully reproduce noise/reverb/echo from the reference into the output. Standard practice: pre-denoise with RNNoise or Demucs vocal isolation, or use a clean studio-recorded sample.
- **Multiple clips**: you can pass a list of reference paths; the latents from each are averaged. Use 2-5 clips of the same speaker for best identity stability.

### Mel Parameters (Reference Mel for the Perceiver and DVAE)

| Param | Value | Notes |
|-------|-------|-------|
| Sample rate | 22050 Hz | Reference-side rate (output is 24 kHz, separate path) |
| n_fft | 1024 | |
| hop_length | 256 | |
| win_length | 1024 | |
| n_mels | 80 | |
| mel_fmin | 0 | |
| mel_fmax | 8000 | |
| Power | 2 (magnitude²) | Standard mel-power spectrogram |
| Log | natural log of `mel + 1e-5` | Not 10·log10 — natural log |
| Normalization | none at the mel layer; per-clip mean-std is NOT applied | The DVAE was trained on raw log-mel |

These mel parameters MUST be reproduced exactly. See [MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md) for the reference STFT/mel-filterbank implementation. The mel filterbank is computed via librosa's standard `mel(sr=22050, n_fft=1024, n_mels=80, fmin=0, fmax=8000, htk=False)` recipe (Slaney normalization, not HTK).

### HuggingFace Model Files

The canonical revision is `coqui/XTTS-v2` (HF), pinned to `v2.0.2`. Full file listing:

| File | Size | Purpose |
|------|------|---------|
| `config.json` | ~4.6 KB | Full model config + `model_args` (all GPT/HiFiGAN/Perceiver hyperparameters) + supported_languages list + audio config (sample_rate=24000, etc.). |
| `model.pth` | ~1.86 GB | PyTorch pickle, the **entire model as one dict**. Top-level keys: `gpt` (GPT-2 + embeddings + heads), `hifigan_decoder` (HiFiGAN + speaker conv heads), `dvae` (the DVAE encoder+codebook+decoder; only the codebook is used at inference). |
| `vocab.json` | ~361 KB | Full HuggingFace `Tokenizer.to_str()` JSON dump (BPE model + merges + pre/post processors), NOT the OpenAI-style split (vocab.json + merges.txt). |
| `tokenizer.json` | (alias) | Same content as `vocab.json` under the standard HF filename in some revisions. |
| `speakers_xtts.pth` | ~7.7 MB | Precomputed `gpt_cond_latent` (32×1024) + `speaker_embedding` (512) pairs for **~58 named "studio speakers"** (e.g. `Claribel Dervla`, `Daisy Studious`, etc.). Lets users get high-quality voices without supplying their own reference audio. PyTorch pickle: dict of {speaker_name: {"gpt_cond_latent": tensor, "speaker_embedding": tensor}}. |
| `dvae.pth` | ~210 MB | **Legacy** — duplicate of the DVAE weights also embedded in `model.pth`. Some older inference paths load this separately. Pure-C# port can ignore. |
| `mel_stats.pth` | ~0.5 KB | Mean/std stats for mel normalization. **Used during DVAE training only**, not used at inference for v2. Safe to ignore. |
| `hash.md5` | small | Integrity manifest. |
| `LICENSE.txt` | ~3 KB | CPML (Coqui Public Model License, **non-commercial**). |
| `README.md` | varies | Model card with usage examples. |

**Notes on the .pth format**:
- All `.pth` files are PyTorch pickles. Loading requires either `torch.load(weights_only=False)` (insecure — arbitrary code execution risk) or a safe re-pickling pass. **For HartsyInference: convert to safetensors offline at packaging time.** The conversion is one-shot: load the dict, walk each tensor, write `safetensors` with the same key names. See [SAFETENSORS_FORMAT.md](SAFETENSORS_FORMAT.md). After conversion the package becomes `xtts_v2.safetensors` (~1.86 GB), plus `tokenizer.json`, `speakers_xtts.safetensors`, and `config.json`.
- Community safetensors conversions exist (search HF for `xtts-v2-safetensors`) but are NOT official. We should do our own conversion to be sure of the key names and to strip the DVAE decoder weights (unused at inference, saves ~100 MB).

### Memory and Performance

| Metric | Value | Conditions |
|--------|-------|-----------|
| Disk (FP32, all weights) | 1.86 GB | Full `model.pth`. |
| Disk (FP32, inference-only, no DVAE decoder) | ~1.75 GB | After stripping unused DVAE keys. |
| Disk (FP16) | ~931 MB | Half-precision conversion. |
| Disk (INT8 quant, GPT only) | ~600 MB | Community quantizations. Quality drop is noticeable but acceptable. |
| **VRAM (FP16, inference, no KV cache)** | ~1.4 GB | Weights only. |
| **VRAM (FP16, inference, full KV cache for 605 mel tokens)** | ~1.8 GB | Weights + KV cache (30 layers × 16 heads × 64 dim × 2 × ~1077 tokens × 2 bytes ≈ 130 MB). |
| **VRAM (FP32)** | ~2.5 GB | Weights only. ~3 GB with KV cache. |
| **CPU RAM (FP32)** | ~3 GB | Includes Python overhead, mel computation buffers, tokenizer. |
| **RTF non-streaming (RTX 4090, FP16)** | ~0.12 | 8x real-time. |
| **RTF streaming (RTX 4090, FP16, chunk=20)** | ~0.18 | Slight overhead for chunking. |
| **RTF non-streaming (RTX 3060 12 GB, FP16)** | ~0.4 | 2.5x real-time. |
| **RTF non-streaming (CPU only, 12-core, FP32)** | ~3-5 | Sub-real-time. CPU inference is impractical for live use. |
| **TTFA streaming (RTX 4090, chunk=8)** | ~150-200 ms | Time to first audio chunk after `inference_stream()` call. |
| **TTFA non-streaming (RTX 4090, 5-sec utterance)** | ~600-800 ms | Wait for the full GPT decode. |

The GPT is the bottleneck for non-streaming (autoregressive, sequence-length-bound). The HiFiGAN is the bottleneck for first-token latency in streaming (one HiFiGAN forward per chunk). The Perceiver and H/ASP encoder are cheap and run once per speaker.

## Key Numbers / Constants

| Constant | Value | Notes |
|----------|-------|-------|
| GPT hidden size | 1024 | n_embd |
| GPT layers | 30 | n_layer |
| GPT heads | 16 | n_head, head_dim=64 |
| GPT max text tokens | 402 | Per utterance |
| GPT max audio tokens | 605 | Per utterance, cap on generation length |
| GPT max conditioning tokens | 70 | Capped, but Perceiver always emits exactly 32 |
| Text vocab size | 6681 | BPE |
| Mel-codec vocab size | 1026 | 1024 codes + START(1024) + STOP(1025) |
| Text [START] / [STOP] | 261 / 0 | |
| Mel [START] / [STOP] | 1024 / 1025 | |
| Perceiver query count | 32 | Output cond latent length |
| Speaker embedding dim | 512 | H/ASP output |
| DVAE codebook size | 1024 × 512 | Used only for training/tokenization, not inference |
| Reference mel sample rate | 22050 Hz | |
| Reference mel n_fft / hop / win | 1024 / 256 / 1024 | |
| Reference mel n_mels | 80 | |
| Reference mel fmin / fmax | 0 / 8000 | |
| GPT code stride length | 1024 samples @ 22050 Hz | ~21.5 mel tokens / sec |
| Output sample rate | 24000 Hz | Mono, float32 |
| HiFiGAN upsample rates | [8, 8, 2, 2] | Factor 256 |
| HiFiGAN resblock kernels | [3, 7, 11] | V1 config |
| Supported languages | 17 | en es fr de it pt pl tr ru nl cs ar zh-cn ja hu ko hi |
| Total parameters | ~443M | ~1.86 GB FP32 |
| License | CPML | Non-commercial |

## Data Layouts / Formats

### Text Token Sequence

```
[lang_tok_id, bpe_id, bpe_id, ..., bpe_id, 261]
^^^           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  ^^^
language      BPE of (normalized, romanized)  [START]
                          text                = signals GPT to begin mel generation
```

### GPT Inference Input Embedding Layout

```
positions:   0..31      32..(32+T_text-1)            (32+T_text)..end
             ^^^^       ^^^^^^^^^^^^^^^^^^           ^^^^^^^^^^^^^^^^^^^^
             cond       text_emb + text_pos_emb      mel_emb + mel_pos_emb
             latent     (T_text from text tokens)    (autoregressively appended)
             (32×1024)
```

Note: text and mel use **separate positional embeddings**, both learned, separate tables. The cond latent has no positional embedding added (the Perceiver bakes in its own ordering via learned queries).

### Speakers File

```
speakers_xtts.pth = {
  "Claribel Dervla": {
    "gpt_cond_latent": Tensor(1, 32, 1024) float32,
    "speaker_embedding": Tensor(1, 512, 1) float32,
  },
  "Daisy Studious": {...},
  ... ~58 speakers total
}
```

### Audio Output

```
Shape: (N_samples,) float32, range approximately [-1, 1]
Sample rate: 24000 Hz
Format: raw PCM, written to WAV with scipy.io.wavfile or soundfile
```

### Weight File Top-Level Keys (in `model.pth`)

```
{
  "model": {
    "gpt.text_embedding.weight":            (6681, 1024)
    "gpt.text_pos_embedding.emb.weight":    (402, 1024)
    "gpt.mel_embedding.weight":             (1026, 1024)
    "gpt.mel_pos_embedding.emb.weight":     (608, 1024)
    "gpt.gpt.h.{0..29}.{ln_1,attn,ln_2,mlp}...":   30 GPT-2 layers
    "gpt.final_norm.weight" / ".bias":      (1024,)
    "gpt.text_head.weight" / ".bias":       (6681, 1024) / (6681,)
    "gpt.mel_head.weight" / ".bias":        (1026, 1024) / (1026,)
    "gpt.conditioning_encoder...":          mel -> latent conv stack
    "gpt.conditioning_perceiver...":        Perceiver IO (32 latents)
    "hifigan_decoder.waveform_decoder...":  HiFiGAN convs + ResBlocks (with speaker FiLM)
    "hifigan_decoder.speaker_encoder...":   H/ASP ResNet (frozen, included in checkpoint)
    "dvae...":                              Discrete VAE (encoder+codebook+decoder).
                                            ONLY the codebook is used at inference,
                                            and ONLY indirectly (the GPT's mel_embedding
                                            already encodes the codebook info).
                                            => entire dvae.* subtree is unused at inference.
  },
  "step": int,
  "config": dict,  # mirror of config.json
}
```

## Algorithm Steps

### Full End-to-End Inference (Non-Streaming)

```
Input: reference_audio_path, text, language

# Once per speaker
1. ref_wav_22k = load_resample(reference_audio_path, 22050, mono)
   ref_mel = log(MelSpectrogram_80(ref_wav_22k, hop=256, win=1024, n_fft=1024, fmin=0, fmax=8000) + 1e-5)
2. cond_features = gpt_conditioning_encoder(ref_mel)
   gpt_cond_latent = perceiver(cond_features, queries=32)        # (32, 1024)
3. ref_wav_16k = load_resample(reference_audio_path, 16000, mono)
   speaker_embedding = L2Norm(h_asp_speaker_encoder(ref_wav_16k))  # (512,)

# Per utterance
4. text_norm = normalize_text(text, language)
   if language in {zh-cn, ja, ko, hi, ar}: text_norm = romanize(text_norm, language)
   text_ids = [lang_tok(language), *bpe_encode(text_norm), 261]
   truncate text_ids to 402 tokens.

5. text_embeds = text_embedding[text_ids] + text_pos_embedding[0..len-1]
   cond_embeds = gpt_cond_latent                                  # (32, 1024)
   gpt_input = concat(cond_embeds, text_embeds)                   # (32+T_text, 1024)

6. # Autoregressive decode
   mel_tokens = [1024]   # gpt_start_audio_token
   latents = []
   kv_cache = None
   for t in 0 .. 604:
     if t == 0:
       inp = concat(gpt_input, mel_embedding[1024] + mel_pos_embedding[0])
       hidden, kv_cache = gpt2_forward(inp, attention_mask=causal)
       last_hidden = hidden[-1]                                   # (1024,)
     else:
       inp = mel_embedding[mel_tokens[-1]] + mel_pos_embedding[t]
       last_hidden, kv_cache = gpt2_forward_incremental(inp, kv_cache)
     latents.append(last_hidden)
     logits = mel_head(last_hidden)                               # (1026,)
     apply repetition_penalty, top_k, top_p, temperature
     next_tok = sample(logits)
     if next_tok == 1025: break
     mel_tokens.append(next_tok)
   latents = stack(latents)                                       # (T_mel, 1024)

7. # Vocoder
   latents = latents.unsqueeze(0).transpose(1,2)                  # (1, 1024, T_mel)
   waveform = hifigan_decoder(latents, speaker_embedding)         # (1, 1, N_samples)
   waveform = waveform.squeeze()                                  # (N_samples,)

Output: 24 kHz waveform.
```

### Streaming Variant

```
After step 6 starts, every `stream_chunk_size` (default 20) new tokens:
  chunk_latents = latents[-(stream_chunk_size + pad):]
  chunk_wav = hifigan_decoder(chunk_latents, speaker_embedding)
  central = chunk_wav[edge_trim : -edge_trim]
  crossfade_overlap_wav_chunks (default 1024) samples with previous emitted chunk.
  yield central.

After step 6 terminates, flush any remaining latents through the vocoder and yield the final chunk.
```

## C# Implementation Notes for HartsyInference

1. **GPT-2 backbone is in HartsyInference.LLM territory.** A 30-layer pre-norm causal Transformer with learned positional embeddings is exactly what the native `HartsyInference.LLM` package implements for small open LLMs. We should expose a configurable GPT-2 module in `HartsyInference.LLM` (or factor it to a shared low-level package) and instantiate it from XTTS with `n_layer=30, n_embd=1024, n_head=16, head_dim=64, bias=True, ffn_dim=4096, max_seq_len≈1077`. Reuse the `HartsyInference.LLM` RoPE-free / learned-pos-embedding path. Reuse the `HartsyInference.LLM` KV-cache infrastructure verbatim — the only XTTS-specific concern is that we have **two prediction heads** (text_head, mel_head) sharing the trunk, only mel_head is needed at inference, and the input embedding is the concatenation of three sub-sequences with **two different positional embedding tables** (text_pos_embedding, mel_pos_embedding) that must be indexed independently. See [DOTLLM_ARCHITECTURE.md](DOTLLM_ARCHITECTURE.md) for the GPT-2 layer primitives we already have.

2. **Mel-VQ codebook is not needed at inference.** This is the most important simplification to bake in. The GPT's `mel_embedding: Embedding(1026, 1024)` already contains everything downstream consumers need; we never need to look up the DVAE codebook directly at runtime. **Strip the entire `dvae.*` subtree from the safetensors package** at conversion time. Saves ~100 MB.

3. **Conditioning encoder = Conv1d stack + Perceiver IO.** The Perceiver is small (~12M params) but needs care: it has 32 learned latent queries, cross-attention from queries → ref_mel features, and ~2-4 transformer layers on the queries. We need to implement:
   - `Conv1d` (already in HartsyInference.Core for HiFiGAN), GroupNorm, GELU.
   - A small Perceiver IO block: one `nn.MultiheadAttention(query=latents, kv=features)` per layer, with pre-LayerNorm and a feed-forward block. ~200 LOC of new code. Document parameter naming in the safetensors mapping table during port.
   - Mel spectrogram with the exact parameters listed above (22050, n_fft=1024, hop=256, win=1024, n_mels=80, fmin=0, fmax=8000, Slaney mel norm, natural log + 1e-5). See [MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md). This is shared with Kokoro mel preprocessing (different params, same machinery).

4. **H/ASP speaker encoder is a small ECAPA-TDNN-style 1D ResNet.** ~7M params, raw 16 kHz waveform → 512-d L2-normed embedding. Architecture:
   - Pre-emphasis (optional, the official path omits it).
   - 40-bin mel filterbank input (NOT 80-bin — H/ASP uses 40 bins at 16 kHz internally; the wav is first turned into a 40-bin mel inside the encoder).
   - 1D ResNet with SE blocks: input 40 → conv1d(stride 1) → 3 ResNet stages with channel widths 32, 64, 128, 256 → SE blocks → ASP (attentive statistics pooling) → linear → 512-d → L2 norm.
   - Implement as a separate small module under `HartsyInference.Audio.SpeakerEncoder` since other TTS models may want to reuse it. **Frozen at training, frozen at inference.**

5. **HiFiGAN decoder reuses our HiFiGAN code with two modifications.** See [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md). The two XTTS-specific changes:
   - Pre-conv accepts 1024-dim input (GPT latents) instead of 80-dim mel.
   - Each ResBlock1 has a FiLM conditioning path: `Linear(512, 2 * channels)` per block produces `(scale, shift)` from `speaker_embedding`, applied as `x = x * (1 + scale) + shift` between the two dilated convs. Add this as an optional flag `useFilmConditioning` on the existing HiFiGAN ResBlock1 class.
   - Final upsample factor is 256, but the final layer is a learned `ConvTranspose1d` to bridge 22050/1024 * 256 = 5512 Hz → 24000 Hz, NOT a simple linear interpolation. Don't shortcut this with `Resample` — load the actual trained weights.

6. **Streaming = KV cache + IAsyncEnumerable.** See [STREAMING_AUDIO_INFERENCE.md](STREAMING_AUDIO_INFERENCE.md) for the project-wide async-iterator pattern. For XTTS specifically:
   - GPT side: KV cache is per-layer `(K, V)` tensors growing along the sequence axis. The first call processes `32 + T_text + 1` tokens; subsequent calls process exactly 1 token. Allocate the KV cache upfront sized for the max sequence (32 + 402 + 605 = 1039 KV positions; preallocate native float16 buffers of `(30 layers, 2, 16 heads, 1039, 64)` = ~130 MB for one inference, reusable across utterances).
   - HiFiGAN side: the official Python re-runs HiFiGAN on each chunk. A better C# implementation maintains per-conv-layer "tail state" (last `kernel-1` samples of each Conv1d) so each chunk runs purely incrementally. This is ~2x faster but more complex; v1 of our port can mirror the Python approach.
   - Crossfade between chunks: linear ramp over `overlap_wav_chunks=1024` samples, applied in-place on the emitted float32 buffer before yield.

7. **Tokenizer = HuggingFace BPE.** The `vocab.json` file is a full `tokenizers.Tokenizer.to_str()` JSON dump — not the OpenAI vocab+merges split. We need a parser for this format. See [TOKENIZERS.md](TOKENIZERS.md). Implementation plan:
   - Parse the JSON to extract: `model.vocab` (token → id map), `model.merges` (BPE merge list), pre-tokenizer (Whitespace = split on `\s+`), normalizer (NFC Unicode normalization), decoder (ByteLevel).
   - Implement standard byte-level BPE encode: NFC → split-on-whitespace → for each word, byte-encode → apply BPE merges → token IDs.
   - **Language token lookup**: at encode time, look up `f"[{lang}]"` in the vocab table and prepend.
   - **Romanization preprocessors**: this is the hard part. The Python wrapper uses external libraries:
     - `pypinyin` for Chinese (Han → Pinyin with tones)
     - `cutlet` for Japanese (Han+kana → romaji)
     - For Korean, Arabic, Hindi: the wrapper does light normalization but mostly relies on the BPE seeing the native script.
     - **Pure-C# port strategy**: for Chinese, ship a precomputed Han→Pinyin lookup table (~20K most common characters, ~500 KB) with simple tone-mark application; for Japanese, ship a Han+kana→romaji table or a small finite-state morphological analyzer. These are one-time ports; not blockers for English/European-language support.
   - **For first ship**: support 14 of 17 languages (drop zh, ja for v1 of the C# port). Add zh + ja in a follow-up with the romanization tables.

8. **License**: CPML is **non-commercial**. HartsyInference itself can ship the model loader code (BSD/MIT-style), but users must accept CPML before downloading weights. Mirror Coqui's approach: weights are not bundled in the NuGet package; a `HartsyInference.Audio.Xtts.DownloadModel(licenseAccepted: true)` helper fetches from HF on first use. Surface the license text and require explicit acceptance.

9. **Validation reference**: the [Idiap fork](https://github.com/idiap/coqui-ai-TTS) is the recommended reference since it is actively maintained. Pin a specific commit, write a Python script that emits intermediate tensors (mel of reference, gpt_cond_latent, speaker_embedding, first 10 mel-token logits, final waveform first 1000 samples), and validate the C# port against those at every component boundary.

## Open Questions

- [ ] **Exact safetensors key naming conventions** after conversion from `model.pth`. The `.pth` keys use dots-and-numbers (`gpt.gpt.h.0.attn.c_attn.weight`); we'll need a deterministic key-mapping table for the C# loader. Build this during the offline pickle→safetensors conversion script.
- [ ] **Is the H/ASP speaker encoder's mel front-end included in the checkpoint, or do we need to compute the 40-bin mel ourselves and feed it?** Reading `hifigan_decoder.py:ResNetSpeakerEncoder.forward` shows the mel is computed inside the encoder (good), but verify the mel params (likely 40 bins, 16000 Hz, hop=160, win=400) and that the filterbank is also baked into the weights.
- [ ] **Repetition penalty implementation**: the default `repetition_penalty=10.0` is unusually aggressive. Confirm it is applied as the standard HuggingFace `RepetitionPenaltyLogitsProcessor` (divide-or-multiply by penalty depending on logit sign) and not as a custom XTTS variant.
- [ ] **Whether `enable_text_splitting=True` should be default-on in our API.** The Python default is True; users almost always want sentence-level chunking for natural pauses. But this adds complexity (sentence segmentation, per-language rules). Punt to a `XttsInferenceOptions.AutoSplitSentences = true` flag.
- [ ] **Cross-lingual tone preservation for tonal languages (zh, vi)**. The Pinyin romanization includes tones; verify the BPE actually carries the tone marks through. (Vietnamese is not in the 17-language list but community finetunes have added it.)
- [ ] **INT8/FP16 quality drop quantification**. Community claims FP16 is indistinguishable; INT8 GPT is "noticeable on long utterances". Validate with our own MOS-like ABX testing before recommending a default precision.
- [ ] **Mel padding for very short references (<6 s)**. The Perceiver expects enough mel frames to attend to; how does the official code handle 3-second references? (Most likely: pads with silence and downweights, but verify.)
- [ ] **Whether to ship `speakers_xtts.pth`** (the 58 studio voices) by default. ~7.7 MB, very high utility — strongly suggest yes, ship alongside the model weights and expose them as `XttsBuiltinSpeakers.Claribel`, etc.

## Cross-References

- [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md) — HiFiGAN baseline architecture; XTTS uses it with FiLM speaker conditioning and 1024-dim latent input.
- [MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md) — Mel-filterbank and STFT recipe used for the reference encoding.
- [TOKENIZERS.md](TOKENIZERS.md) — BPE / SentencePiece formats and HuggingFace `tokenizer.json` parsing.
- [WHISPER_ARCHITECTURE.md](WHISPER_ARCHITECTURE.md) — sibling speech model; shares mel preprocessing and BPE conventions.
- [KOKORO_ARCHITECTURE.md](KOKORO_ARCHITECTURE.md) — alternative TTS pipeline (style-based, non-cloning); contrast with XTTS's zero-shot cloning approach.
- [DOTLLM_ARCHITECTURE.md](DOTLLM_ARCHITECTURE.md) — GPT-2 transformer layer primitives (XTTS reuses these).
- [SAFETENSORS_FORMAT.md](SAFETENSORS_FORMAT.md) — target format for the converted weights.
- `STREAMING_AUDIO_INFERENCE.md` — async-iterator pattern for streamed audio output (referenced; see project-wide streaming spec).

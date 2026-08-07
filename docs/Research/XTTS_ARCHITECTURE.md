# XTTS-v2 — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: HartsyInference.Audio (XTTS-v2 pipeline)

> **Stub.** The narrative walkthrough, restated pseudocode and resolved open questions were
> removed on 2026-08-06 — this model is built and verified, so the C# is the source of truth for
> *how it works*. What remains is what the code cannot tell you: upstream provenance, reference
> constants to diff a suspect port against, where implementations disagree, and bring-up traps.
> Full history is in git. Parity evidence: `docs/Checklists/PARITY_VERIFICATION.md`.

## Summary

XTTS-v2 (Coqui, Sept 2023) is a multilingual zero-shot voice-cloning TTS that clones a target speaker's voice from as little as 6 seconds of reference audio, in any of 17 languages, with cross-language transfer (e.g. clone an English speaker into German). It is the most widely deployed open TTS as of 2026 despite Coqui shutting down in Jan 2024 — both the original [`coqui-ai/TTS`](https://github.com/coqui-ai/TTS) repo (archived) and the maintained Idiap fork [`idiap/coqui-ai-TTS`](https://github.com/idiap/coqui-ai-TTS) ship the same checkpoint, distributed via the HuggingFace repo [`coqui/XTTS-v2`](https://huggingface.co/coqui/XTTS-v2) under the Coqui Public Model License (CPML, non-commercial).

Architecturally XTTS-v2 is a four-component pipeline. A small **conditioning encoder** turns a 6+ second reference clip into a fixed-length speaker latent (and a separate 512-dim "speaker embedding" from a pretrained H/ASP speaker-verification net). A **GPT-2-style autoregressive transformer** (~443M params, 30 layers, d_model=1024, 16 heads) takes BPE text tokens (~6.6k vocab, per-language prefix tokens like `[en]`) plus the speaker latent and autoregressively predicts a stream of discrete **mel-codec tokens** drawn from a 1024-entry VQ-VAE codebook trained on 80-bin mel spectrograms at 22.05 kHz. A small **GPT→latent decoder** (a 6-layer Perceiver-style block called `gpt_inference_head`) converts the predicted mel-codec token sequence and speaker conditioning into a continuous latent stream. Finally a **HiFiGAN-based waveform decoder** (with speaker-embedding conditioning injected into its residual blocks) upsamples those latents directly to 24 kHz waveform — XTTS-v2 does NOT produce an intermediate mel spectrogram at inference; the HiFiGAN consumes the GPT latent stream directly. Total released model is ~1.86 GB FP32, ~931 MB FP16.

The model paper (["XTTS: a Massively Multilingual Zero-Shot Text-to-Speech Model", arXiv:2406.04904](https://arxiv.org/abs/2406.04904)) and the model card document the design lineage from Tortoise-TTS (the GPT+mel-token+vocoder pattern is Betker's) with three key changes: (1) cross-lingual training with language conditioning, (2) replacement of Tortoise's expensive diffusion+UnivNet stack with a single HiFiGAN that consumes GPT latents directly, and (3) chunked streaming. This file covers the architecture; the vocoder back-end is documented in [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md), the BPE/SentencePiece tokenizer machinery in [TOKENIZERS.md](TOKENIZERS.md), and mel preprocessing in [MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md).

Sources: [arXiv:2406.04904](https://arxiv.org/abs/2406.04904), [coqui-ai/TTS (archived)](https://github.com/coqui-ai/TTS), [idiap/coqui-ai-TTS fork](https://github.com/idiap/coqui-ai-TTS), [coqui/XTTS-v2 HF model](https://huggingface.co/coqui/XTTS-v2), [Coqui XTTS docs](https://docs.coqui.ai/en/latest/models/xtts.html), [Tortoise-TTS](https://github.com/neonbjb/tortoise-tts) (architectural ancestor).

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

## C# Implementation Notes for HartsyInference

1. **GPT-2 backbone is in HartsyInference.LLM territory.** A 30-layer pre-norm causal Transformer with learned positional embeddings is exactly what the native `HartsyInference.LLM` package implements for small open LLMs. We should expose a configurable GPT-2 module in `HartsyInference.LLM` (or factor it to a shared low-level package) and instantiate it from XTTS with `n_layer=30, n_embd=1024, n_head=16, head_dim=64, bias=True, ffn_dim=4096, max_seq_len≈1077`. Reuse the `HartsyInference.LLM` RoPE-free / learned-pos-embedding path. Reuse the `HartsyInference.LLM` KV-cache infrastructure verbatim — the only XTTS-specific concern is that we have **two prediction heads** (text_head, mel_head) sharing the trunk, only mel_head is needed at inference, and the input embedding is the concatenation of three sub-sequences with **two different positional embedding tables** (text_pos_embedding, mel_pos_embedding) that must be indexed independently. The GPT-2 layer primitives already exist in `HartsyInference.LLM`.

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

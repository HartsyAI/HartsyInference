# NVIDIA Canary — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: SharpInference.Audio (Canary pipeline)

## Summary

NVIDIA Canary is a family of encoder-decoder multitask speech models — automatic speech recognition (ASR) and automatic speech translation (AST) — built by NVIDIA NeMo. Architecturally each model pairs a **FastConformer encoder** (same family used by Parakeet — see [PARAKEET_ARCHITECTURE.md](PARAKEET_ARCHITECTURE.md)) with an **autoregressive Transformer decoder** (very similar to Whisper's decoder — see [WHISPER_ARCHITECTURE.md](WHISPER_ARCHITECTURE.md)). The encoder consumes 16 kHz log-mel features (128 mel bins) and downsamples by 8x via depthwise-striding subsampling; the decoder is prompted with a sequence of special tokens (start, source-lang, target-lang, task, PnC toggle, timestamp toggle, …) Whisper-style and generates output tokens token-by-token with cross-attention into the encoder.

The "Flash" variants reduce the decoder from 24 layers to 4 (similar to Whisper-large-v3-turbo and Distil-Whisper) and trade a small amount of quality for ~3-4x end-to-end speedup; the v2 release re-architected the model around a unified BPE SentencePiece tokenizer (16,384 tokens) and scaled language coverage from 4 → 25 European languages.

Audio preprocessing is identical to Parakeet (NeMo FilterbankFeatures) — see [MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md). The pipeline reuses our FastConformer encoder (Parakeet) and a Whisper-like cross-attention Transformer decoder.

Sources: [Canary-1B HF](https://huggingface.co/nvidia/canary-1b), [Canary-1B-Flash HF](https://huggingface.co/nvidia/canary-1b-flash), [Canary-180M-Flash HF](https://huggingface.co/nvidia/canary-180m-flash), [Canary-1B-v2 HF](https://huggingface.co/nvidia/canary-1b-v2), [Less is More: Canary paper (arXiv 2406.19674)](https://arxiv.org/abs/2406.19674), [Canary-v2 + Parakeet-v3 paper (arXiv 2509.14128)](https://arxiv.org/abs/2509.14128), [Training and Inference Efficiency paper (arXiv 2503.05931)](https://arxiv.org/abs/2503.05931), [NeMo fast-conformer_aed.yaml](https://github.com/NVIDIA/NeMo/blob/main/examples/asr/conf/speech_multitask/fast-conformer_aed.yaml), [NVIDIA Canary blog](https://developer.nvidia.com/blog/new-standard-for-speech-recognition-and-translation-from-the-nvidia-nemo-canary-model/), [NeMo Canary tutorial](https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/Canary_Multitask_Speech_Model.ipynb)

## Detailed Findings

### 1. Variants

All Canary variants are encoder-decoder FastConformer-AED ("Attention Encoder-Decoder") models. Differences are encoder/decoder depth, tokenizer, language coverage, training data, and the addition of optional tags (timestamps, ITN, diarization). All run at 16 kHz mono, 128 mel bins, max ~40 s per utterance (longer audio is chunked with overlap).

| Model | Params | Enc layers | Dec layers | d_model (enc/dec) | Vocab | Tokenizer | Languages | Tasks | File (.nemo) | Release |
|-------|-------:|-----------:|-----------:|------------------:|------:|-----------|----------:|-------|-------------:|---------|
| `nvidia/canary-1b` | 1.0B | 24 (FC-XL) | 24 (Transformer) | 1024 / 1024 | 1024×4 + specials | Concatenated SentencePiece (per-language, 1024 each) | 4 (en, de, es, fr) | ASR + AST (En↔3) + PnC toggle | 4.07 GB | Apr 2024 |
| `nvidia/canary-1b-flash` | 883M | 32 (FC-L) | **4** (Transformer) | 1024 / 1024 | 1024×4 + specials | Concatenated SentencePiece | 4 (en, de, es, fr) | ASR + AST + PnC + **timestamps** | 3.54 GB | Mar 2025 |
| `nvidia/canary-180m-flash` | 182M | 17 (FC) | **4** (Transformer) | 512 / 512 (est.) | 1024×4 + specials | Concatenated SentencePiece | 4 (en, de, es, fr) | ASR + AST + PnC + timestamps | 737 MB | Mar 2025 |
| `nvidia/canary-1b-v2` | 978M | 32 (FC-L) | **8** (Transformer) | 1024 / 1024 | **16,384** unified | Unified SentencePiece BPE | **25** European | ASR + AST (X↔En, all dirs) + timestamps (NFA) | 6.36 GB | Aug 2025 |
| `nvidia/canary-1b-asr` (NIM) | 978M | 32 | 8 | 1024 / 1024 | 16,384 unified | Unified SentencePiece BPE | 25 | ASR only (subset of v2 weights) | — | Oct 2025 |
| `nvidia/canary-qwen-2.5b` | 2.5B | 32 (FC-L, from canary-1b-flash) | Qwen3-1.7B decoder + LoRA + linear proj | 1024 enc / 2048 LLM | Qwen3 vocab | Qwen3 tokenizer | en only | ASR with PnC, LLM-style outputs | — | Jul 2025 |

Notes:
- **FC-XL** (used by Canary-1B v1) = FastConformer "XL" preset: 24 layers, d_model 1024, 8 heads, FFN 4096, conv kernel 9, subsampling 8x.
- **FC-L** (used by all "Flash" + v2 1B variants) = same d_model/heads but a *deeper* encoder (32 layers) and a *thinner* decoder. The "Flash" idea: keep the heavy encoder for accuracy, shrink the decoder (which is run autoregressively) for throughput. Same principle as Whisper-large-v3-turbo.
- **Canary-180M-Flash** is the most aggressive shrink: 17 encoder layers + 4 decoder layers. NVIDIA has not published the exact d_model for 180M but config patterns in NeMo suggest 512 with 8 heads and FFN 2048 (verify from extracted `.nemo` `model_config.yaml`).
- **Canary-1B-v2** is a re-architecture, not a fine-tune: it switches to a **single unified BPE** tokenizer (16,384 tokens — larger and more efficient than the v1 concatenated 4×1024 setup) and doubles the decoder depth from 4 (Flash) back to 8 to handle 25 languages.
- **Canary-Qwen-2.5B** is a Speech-Augmented LLM (SALM): the canary-1b-flash encoder is frozen, audio embeddings are projected with a linear layer into Qwen3-1.7B's embedding space, then LoRA-adapted Qwen3 generates output. Out of scope for the standard Canary pipeline — needs LLM integration via dotLLM.

#### Performance (Open ASR Leaderboard, WER lower is better)

Canary-1B (v1) — Mozilla Common Voice 16.1:
| Lang | WER |
|------|-----|
| en | 7.97% |
| de | 4.61% |
| es | 3.99% |
| fr | 6.53% |

Canary-1B-Flash — English (LibriSpeech / SPGI / MCV-16):
| Dataset | WER |
|---------|----:|
| LibriSpeech Clean | 1.48% |
| LibriSpeech Other | 2.87% |
| SPGISpeech | 1.95% |
| MCV-16 EN | 6.99% |

Canary-1B-Flash — multilingual (MCV-16.1):
| Lang | WER |
|------|----:|
| de | 4.09% |
| es | 3.62% |
| fr | 6.15% |

Canary-1B-v2 — multilingual rollups:
| Bench | WER |
|------|----:|
| FLEURS (25 langs) | 8.40% |
| CoVoST (13 langs) | 8.85% |
| MLS (6 langs) | 7.27% |
| LibriSpeech Clean | 2.18% |
| LibriSpeech Other | 3.56% |

#### Translation BLEU / COMET (FLEURS)

Canary-1B (v1) and Canary-1B-Flash (similar tier):
| Direction | BLEU | COMET |
|-----------|-----:|------:|
| En→De | 32.15 / 32.27 | 0.81 |
| En→Fr | 40.76 / 41.22 | 0.82 |
| En→Es | 22.66 | — |
| De→En | 33.98 / 35.5 | 0.85 |
| Fr→En | 30.95 / 33.42 | 0.85 |
| Es→En | 21.80 | — |

Canary-1B-v2 (24-language rollups):
| Direction | BLEU | COMET |
|-----------|-----:|------:|
| X→En FLEURS (24 langs) | 29.08 | 79.30 |
| En→X FLEURS (24 langs) | 29.40 | 84.56 |
| X→En CoVoST (13 langs) | 40.48 | 77.48 |
| En→X CoVoST (5 langs) | 32.33 | 80.29 |

#### Inference speed (RTFx = inverse RTF; higher is faster)

| Model | RTFx A100 | RTFx H100 |
|-------|----------:|----------:|
| Canary-1B (v1) | ~150 | — |
| Canary-1B-Flash | **1046** | **1669** |
| Canary-180M-Flash | **1233** | **2041** |
| Canary-1B-v2 | 749 | — |

Flash variants are >5x faster than Canary-1B v1 because most inference time in encoder-decoder ASR is autoregressive decoding, and decoder layers go from 24 → 4.

### 2. Architecture

#### 2.1 Encoder — FastConformer

Identical architecture family to Parakeet; see [PARAKEET_ARCHITECTURE.md](PARAKEET_ARCHITECTURE.md) for the FastConformer block diagram (subsampling → N × {feed-forward macaron-net, multi-head self-attention with relative positional encoding, depthwise-separable convolution, feed-forward macaron-net, layer-norm}). Canary's encoder differs from Parakeet only in **depth** and **lack of a CTC/RNNT head** (no auxiliary loss; the encoder is purely a feature extractor whose output is consumed by the Transformer decoder via cross-attention).

Canary-1B (v1) — FastConformer XL:
- `n_layers`: 24
- `d_model`: 1024
- `n_heads`: 8 (head dim = 128)
- `ff_expansion_factor`: 4 → FFN inner = 4096
- `conv_kernel_size`: 9
- `subsampling`: `dw_striding`, factor 8x → frame rate = 12.5 Hz (one feature per 80 ms)
- `subsampling_conv_channels`: 256
- `self_attention_model`: `rel_pos` (Transformer-XL-style relative positional bias)
- `dropout`: 0.1

Canary-1B-Flash / Canary-1B-v2 — FastConformer L (deeper, same width):
- `n_layers`: 32
- Other dims identical to XL.

Canary-180M-Flash:
- `n_layers`: 17
- Width is reduced (likely d_model 512, FFN 2048, 8 heads), but verify from the unpacked `.nemo` config — NVIDIA's README doesn't publish it.

The encoder emits a sequence of `[T/8, d_model]` features that the decoder cross-attends to. Same `forward(x, x_lens) → (enc_out, enc_out_lens)` API as Parakeet.

#### 2.2 Decoder — Autoregressive Transformer

A standard pre-LN Transformer decoder, very close to Whisper's. Per-layer block: self-attention (causal) → cross-attention (decoder Q × encoder K/V) → FFN.

| Variant | Layers | Hidden | Heads | FFN inner | Max seq |
|---------|------:|------:|-----:|----------:|--------:|
| Canary-1B (v1) | 24 | 1024 | 8 | 4096 | 512 |
| Canary-1B-Flash | 4 | 1024 | 8 | 4096 | 512 |
| Canary-180M-Flash | 4 | 512 (likely) | 8 | 2048 (likely) | 512 |
| Canary-1B-v2 | 8 | 1024 | 8 | 4096 | 512 |

NeMo config keys (`fast-conformer_aed.yaml`):
```yaml
transf_decoder:
  num_layers: 24
  hidden_size: 1024
  inner_size: 4096
  num_attention_heads: 8
  ffn_dropout: 0.1
  attn_score_dropout: 0.1
  attn_layer_dropout: 0.1
  mask_future: False         # cross-attn is unmasked
  pre_ln: true               # pre-LayerNorm, like GPT
  max_sequence_length: 512
```

Key differences from Whisper's decoder:
- **Pre-LN** (Whisper is also pre-LN — same).
- **Learned absolute positional embeddings** on the decoder side (Whisper: same), with `max_sequence_length` = 512 (Whisper: 448).
- **Tied embedding / output projection**: yes (verify in checkpoint — typical NeMo pattern).
- **No prefix tokens for "no-speech" probability**: Canary has `<|nospeech|>` but uses it differently from Whisper's separate no-speech prob output.

Practical implication: a SharpInference `TransformerDecoder` class built for Whisper can be reused with minor parameter changes (different vocab size, different max seq, different special-token IDs). The forward graph (self-attn → cross-attn → FFN, pre-LN) and the KV cache shape are identical.

#### 2.3 Tokenizer

**Canary v1 family (canary-1b, canary-1b-flash, canary-180m-flash) — Concatenated SentencePiece** (called "AggregateTokenizer" or `type=agg` in NeMo, but this is **not** the unified-aggregate from academic literature — it's a *concatenation* of per-language tokenizers with shared special tokens):

- One independent SentencePiece BPE model per language. Each is 1024 tokens.
- Per-language vocabularies are stacked with offsets:
  - `[0 … 1023]`: English BPE
  - `[1024 … 2047]`: German BPE
  - `[2048 … 3071]`: Spanish BPE
  - `[3072 … 4095]`: French BPE
- Plus a small block of shared **special tokens** (see §3): `<|startoftranscript|>`, `<|endoftranscript|>`, `<|transcribe|>`, `<|translate|>`, `<|nospeech|>`, `<|en|>`, `<|de|>`, `<|es|>`, `<|fr|>`, `<|pnc|>`, `<|nopnc|>`, etc.

Total vocab is on the order of 5,120 (4×1024 + ~24 specials). The Flash variants add timestamp special tokens (`<|timestamps|>`, `<|notimestamps|>`).

When encoding, NeMo picks the SentencePiece model for the active language and emits IDs in that language's range. When decoding the model output, the token ID range determines which sub-tokenizer to decode with.

**For our C# implementation** the cleanest approach is:
1. Extract all 4 SentencePiece `.model` files from the `.nemo` tar.
2. At runtime, build an IdRange → SubTokenizer dispatcher.
3. Encoding: route by current language, add the offset.
4. Decoding: subtract offset, route to per-language SentencePiece decoder.

We need a pure-C# SentencePiece reader (no native `sentencepiece` library). This is shared with several other components — see [TOKENIZERS.md](TOKENIZERS.md).

**Canary v2 — Unified SentencePiece BPE**:
- Single SentencePiece model with **16,384** tokens.
- Trained jointly over all 25 languages' transcripts.
- Plus **1,162 special tokens** specific to Canary2 prompts (task tokens, language IDs for 25 langs, timestamp / PnC / ITN / diarization / emotion / foreign-word flags, placeholder slots for future tasks).
- The paper experimented with 4K / 8K / 16K vocab sizes; larger consistently won on both ASR and AST.
- Tokenizer `.model` file is named like `<hash>_tokenizer.model` inside the `.nemo` tar (e.g., `cc5d48e83aad4be48aa9fa264b727c4b_tokenizer.model`).

For v2 our pipeline is simpler: a single SentencePiece BPE model + a known fixed list of special tokens. No per-language dispatching.

### 3. Prompt Format

The decoder is **prompted** in a way deliberately parallel to Whisper. The training objective is: given an encoded audio sequence and a special-token prefix, autoregressively generate the answer (transcription or translation) followed by `<|endoftranscript|>`. The prefix encodes the task.

#### 3.1 Canary v1 / Flash prompt (`prompt_format: "canary"`)

The decoder is initialised with this sequence before generation:

```
<|startoftranscript|> <|src_lang|> <|task|> <|tgt_lang|> <|pnc|or|nopnc|> [ <|timestamps|or|notimestamps|> ] <generated text…> <|endoftranscript|>
```

Token-by-token roles:
- `<|startoftranscript|>` (BOS) — always first.
- `<|src_lang|>` — one of `<|en|>`, `<|de|>`, `<|es|>`, `<|fr|>` — the language of the audio.
- `<|task|>` — `<|transcribe|>` (ASR) or `<|translate|>` (AST).
- `<|tgt_lang|>` — language of the output text. For ASR `tgt == src`. For AST, `tgt ≠ src`.
- `<|pnc|>` / `<|nopnc|>` — emit punctuation & capitalization, or strip them.
- `<|timestamps|>` / `<|notimestamps|>` — only Flash variants. Toggle timestamp-token emission interleaved with words. `<|notimestamps|>` is the default.
- After this prefix, the model autoregressively generates text tokens until it emits `<|endoftranscript|>` (EOS).

There is also an `<|nospeech|>` token that the model may emit immediately after the prefix to indicate the audio contains no speech (used at training time for VAD-style examples).

Example for "English audio → English transcript, with PnC, no timestamps":
```
<|startoftranscript|> <|en|> <|transcribe|> <|en|> <|pnc|> <|notimestamps|>
```

Example for "French audio → English translation, with PnC, with timestamps":
```
<|startoftranscript|> <|fr|> <|translate|> <|en|> <|pnc|> <|timestamps|>
```

#### 3.2 Canary v2 prompt (`prompt_format: "canary2"`)

Extended with optional decoder-context, more toggles, and an explicit slot template:

```
<|startofcontext|> <decoder context tokens…> <|startoftranscript|> <|emo:undefined|> <|src_lang|> <|tgt_lang|> <|pnc|or|nopnc|> <|itn|or|noitn|> <|timestamp|or|notimestamp|> <|diarize|or|nodiarize|> <|foreign|or|noforeign|> <generated text…> <|endoftranscript|>
```

Default dialog slot values (when caller doesn't override):
- `emotion`: `<|emo:undefined|>`
- `pnc`: enabled (training transcripts all have PnC)
- `itn`: `<|noitn|>` (no inverse text normalization)
- `timestamp`: `<|notimestamp|>`
- `diarize`: `<|nodiarize|>`
- `foreign`: `<|noforeign|>` (whether to wrap non-source-language words specially)

The `<|startofcontext|>` block (`CANARY2_BOCTX`) can contain a textual prompt that biases decoding (e.g., a domain glossary or speaker name). It is optional; if absent, only `<|startoftranscript|>` (`CANARY_BOS`) appears.

#### 3.3 Differences from Whisper's prompt

Whisper's prefix is: `<|startoftranscript|> <|lang|> <|task|> [<|notimestamps|>]`. Canary v1 essentially adds the **target-language** token (Whisper can only translate-to-English) and the **PnC toggle**. Canary v2 adds even more dimension (ITN, diarization, foreign-word handling, decoder context, emotion).

For C#, model both as a `CanaryPrompt` builder struct that takes options (`SrcLang`, `TgtLang`, `Task`, `Pnc`, `Timestamps`, optional `ContextText` for v2) and emits the token-ID prefix array. Each model variant ships its own prompt builder because the special-token IDs change.

### 4. Mel Preprocessing

Identical to Parakeet — NeMo `FilterbankFeatures`. See [MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md) for the canonical pipeline and [PARAKEET_ARCHITECTURE.md](PARAKEET_ARCHITECTURE.md) for the reference fixture.

From `fast-conformer_aed.yaml`:

| Param | Value |
|-------|-------|
| `sample_rate` | 16000 |
| `n_fft` | 512 |
| `features` (n_mels) | **128** |
| `window_size` | 0.025 s (= 400 samples) |
| `window_stride` | 0.010 s (= 160 samples → 100 fps before encoder subsampling) |
| `window` | `hann` |
| `normalize` | `per_feature` (per-mel-bin mean/var normalization over the utterance) |
| `dither` | 1e-5 |
| `pad_to` | 0 (no padding; lengths are tracked) |
| `log` | true (log mel) |
| `frame_splicing` | 1 |

After the 8x encoder subsampling, effective frame rate is 12.5 Hz (80 ms / encoder frame). Whisper, for comparison, runs at 50 Hz (20 ms / encoder frame); Canary's encoder is coarser but has the deeper FastConformer block to compensate.

**Important**: `per_feature` normalization needs the *full* feature sequence in memory to compute mean/std. For streaming, NeMo uses a "cache-aware" path with running stats — but the published Canary models are full-utterance only (use the chunked 40 s sliding window for long audio).

### 5. Decoding

#### 5.1 Algorithm

Autoregressive generation, exactly as Whisper:
1. Encode audio → `enc_out` of shape `[1, T/8, 1024]`.
2. Build prompt prefix tokens (see §3).
3. Feed prefix through decoder once with full causal mask, populating the **self-attention KV cache**.
4. For each cross-attention layer, compute encoder K/V projections **once** and cache them (the **cross-attention KV cache**) — they never change during generation.
5. Loop: feed last predicted token → decoder runs one step using cached self-attn KV + cached cross-attn K/V → output logits → sample/greedy → append → break on `<|endoftranscript|>` or max length.

#### 5.2 Beam search

Default published config: `beam_size = 5`, `length_penalty = 1.0`. Greedy (`beam_size = 1`) is the recommended setting for Flash inference and is what the published RTFx numbers assume — beam-5 is ~5x slower.

NeMo class: `BeamSearchSequenceGeneratorWithLanguageModel` (an external LM is *not* used by default in Canary; the class supports it but the published checkpoints don't ship an LM).

#### 5.3 KV caches

- **Self-attention cache**: per-layer `[B, n_heads, seq_so_far, head_dim]` for K and V. Grows by 1 step per token. Same shape behavior as Whisper.
- **Cross-attention cache**: per-layer `[B, n_heads, T/8, head_dim]` for K and V — computed once after encode and *never* updated. This is the single biggest perf win and **must** be implemented (otherwise the encoder K/V projections run every decoder step).

For C# (CUDA via PTX, pinned unmanaged memory), pre-allocate both caches at max length and grow `seq_so_far` by index rather than reallocating.

#### 5.4 Chunked / streaming long-form decoding

Max audio per forward = 40 s (= 4000 mel frames pre-subsampling, ≈ 500 encoder frames post-subsampling). For longer audio, NeMo provides **AlignAtt** chunked decoding (`Canary Chunked and Streaming Decoding`):
- Cross-attention scores are inspected on the second-to-last decoder layer (`xatt_layer = -2`).
- A token is committed only if its peak cross-attention is sufficiently far from the right edge of the encoded chunk (`cross_attention_threshold`, default 8 frames). Otherwise the chunk advances and the token is re-decoded.

This is more complex than chunked Whisper. For first-pass C# implementation: do **40 s sliding window with overlap** (1 s overlap; merge text via dedup) and skip AlignAtt initially. Add AlignAtt later if streaming/long-form quality matters.

### 6. Canary-1B-Flash — What Changes vs Canary-1B

Both are 1B-class encoder-decoder ASR/AST models, four languages, same prompt format ("canary"), same training data scale (85k hours). What changed:

| Aspect | Canary-1B (v1) | Canary-1B-Flash |
|--------|---------------|------------------|
| Encoder layers | 24 (FC-XL) | **32** (FC-L) |
| Decoder layers | 24 | **4** |
| Total params | 1.0B | **883M** |
| Timestamps | no | **yes** (toggle token) |
| RTFx (A100) | ~150 | **1046** (~7x faster) |
| Training | 150k steps | 200k steps |
| English WER (MCV-16) | 7.97% | 6.99% |
| License | CC-BY-NC-4.0 | **CC-BY-4.0** (commercial OK) |

**Mechanism**: Flash is **not** a direct distillation of Canary-1B-v1 with a smaller decoder. It is a **separate model trained from scratch** with a different architecture (deeper encoder, shallower decoder) and a longer schedule, and it adds timestamp tokens to the vocabulary. NVIDIA's paper "Training and Inference Efficiency of Encoder-Decoder Speech Models" (arXiv 2503.05931) covers the design space.

The pattern matches **Whisper-large-v3-turbo** and **Distil-Whisper**: cut decoder depth aggressively (Whisper-turbo: 32→4; Distil-Whisper: 32→2; Canary-Flash: 24→4) because autoregressive decoding dominates wall-clock time. Encoder cost is amortised across all output tokens, so making the encoder *deeper* (Canary-Flash 32 layers vs v1 24) actually improves accuracy at minimal speed cost.

**Implementation impact**: same decoder class as v1 with `num_layers = 4`. No special distillation handling.

### 7. Translation Pairs Supported

#### Canary v1 / Flash (4 languages):
- ASR: `en → en`, `de → de`, `es → es`, `fr → fr`
- AST En→X: `en → de`, `en → es`, `en → fr`
- AST X→En: `de → en`, `es → en`, `fr → en`
- **Not supported**: X→X (e.g., `de → fr` is not in training).

Total: 4 ASR + 6 AST = 10 directed task pairs.

#### Canary v2 (25 languages):
Languages: `bg, hr, cs, da, nl, en, et, fi, fr, de, el, hu, it, lv, lt, mt, pl, pt, ro, sk, sl, es, sv, ru, uk`.

- ASR: 25 self-pairs.
- AST En→X: 24 (English → each of the other 24).
- AST X→En: 24 (each of the other 24 → English).
- **Not supported**: X→X among non-English pairs.

Total: 25 ASR + 48 AST = 73 directed task pairs.

### 8. Timestamps

Two distinct mechanisms across the Canary family. Don't confuse them.

#### 8.1 Canary-Flash family — Whisper-style timestamp tokens

The vocabulary includes `<|timestamps|>` and `<|notimestamps|>` toggle tokens (in the prompt prefix), plus a block of **timestamp tokens** interleaved with text (similar to Whisper's `<|0.00|>` … `<|30.00|>` tokens at 20 ms or 80 ms resolution).

When `<|timestamps|>` is in the prefix, the decoder emits sequences like:
```
<|ts_0.00|> Hello <|ts_0.45|> <|ts_0.45|> world <|ts_0.80|> ...
```
Pairs of timestamp tokens bracket each word/segment. The pipeline reads them out to produce word-level and segment-level start/end times. Reported accuracy: F1 @ 200 ms collar = **95.5%** on LibriSpeech test-clean (Canary-1B-Flash).

This is "experimental" in NVIDIA's words — quality matches reasonable forced-aligner output but is not perfect on noisy or fast speech. The exact token granularity (number of timestamp tokens, time-per-step) is not in the README — extract from the `.nemo` vocab or NeMo source.

#### 8.2 Canary-1B-v2 — NeMo Forced Aligner (NFA) + auxiliary CTC

v2 takes a different approach: instead of token-based timestamps, it runs the **NeMo Forced Aligner** with an **auxiliary CTC head** to produce segment-level timestamps. The main decoder generates plain text; CTC alignment is computed in a second pass over the same encoder features. This is more reliable but requires the CTC head weights (shipped in the `.nemo`).

The CTC head is a small linear projection over the encoder output. Implementation: roughly 100 lines of straight Viterbi over `log_softmax(encoder_out @ ctc_weight)`.

#### 8.3 Canary-1B (v1) — no timestamps

Doesn't support timestamps at all. If a user needs them with v1, run a separate aligner.

### 9. Memory and Performance

#### VRAM (full FP16/BF16 inference, batch=1, 40 s audio)

| Model | Weights | Cross-attn KV (40s) | Self-attn KV (~200 tok) | Total |
|-------|--------:|--------------------:|------------------------:|------:|
| Canary-1B (v1) | ~2.0 GB | ~24 MB (24 layers × 500 frames × 1024 × 2 × 2B) | ~3 MB | **~2.1 GB** |
| Canary-1B-Flash | ~1.8 GB | ~4 MB (4 layers) | ~0.5 MB | **~1.9 GB** |
| Canary-180M-Flash | ~370 MB | ~1 MB | ~0.2 MB | **~0.4 GB** |
| Canary-1B-v2 | ~2.0 GB | ~8 MB (8 layers) | ~1 MB | **~2.1 GB** |

Add ~50-100 MB for the mel/encoder intermediate activations. Comfortable on a 4 GB GPU; trivial on 8+ GB.

INT8 / FP8 quantization is supported by NeMo via TensorRT-LLM but no published recipe for pure ONNX/Safetensors. For our SharpInference loader we should plan FP16 first, GGUF Q8_0/Q4_K_M later.

#### Compute / latency

Encoder cost is dominant for short audio; for long audio (40 s) on the Flash variants, decoder cost is 30-40% of total because there are only 4 decoder layers. RTFx > 1000 on A100 for Flash means **transcribing 1 hour of audio takes ~3.5 seconds of compute**.

For C# / CUDA: the encoder FastConformer maps to ~10 PTX kernels per layer (FFN, attention, conv) — same as Parakeet. The decoder maps to ~8 PTX kernels per layer (self-attn, cross-attn, FFN) — same as Whisper. **No new kernels needed** if both Parakeet and Whisper are already implemented.

### 10. C# Implementation Notes

#### Reuse strategy
- **Encoder**: 100% reuse the FastConformer encoder built for Parakeet (`SharpInference.Audio.FastConformer`). Only parameters differ (layer count and possibly d_model for 180M). Construct with config struct.
- **Decoder**: 95% reuse the Transformer decoder built for Whisper (`SharpInference.Audio.TransformerDecoder`). Same forward graph (self-attn → cross-attn → FFN, pre-LN). Differences: vocab size, max_seq, special-token IDs, and pos-embedding table size.
- **Mel preprocessor**: 100% reuse the NeMo `FilterbankFeatures` implementation built for Parakeet — same config (128 mels, 512 n_fft, 25 ms / 10 ms, `per_feature` norm).
- **Beam search**: 100% reuse Whisper's beam search (greedy + beam-5 with length penalty).
- **KV caches**: same shapes and lifetime as Whisper. Reuse Whisper's `KvCache` allocator.

#### New things to build
1. **CanaryPrompt builder** — emits the prefix-token sequence for v1 ("canary") and v2 ("canary2") formats. Handles src/tgt language, task, PnC, timestamps, and v2-only options (ITN, diarize, emotion, decoder context).
2. **Concatenated tokenizer router** (v1 / Flash) — IDs are offset by language. Encode and decode dispatch on the active language. Internally hosts 4 pure-C# SentencePiece BPE models.
3. **Special-token table** — needs to be generated per-model from the unpacked `.nemo` because NVIDIA changed token IDs between checkpoints. Plan: parse the `vocab.json` / `tokenizer.model` from the `.nemo` archive at load time and build a name → ID map (`Bos`, `Eos`, `Transcribe`, `Translate`, `EnLang`, `DeLang`, …). Hardcoding IDs is fragile.
4. **Timestamp-token reader** (Flash) — post-process decoded token stream into `WordTimestamp { Text, Start, End }` by pairing adjacent timestamp tokens. Document the timestamp resolution after extracting from one of the models.
5. **NFA forced aligner** (v2) — optional second-pass timestamping using the auxiliary CTC head from the encoder. Defer to phase 2.
6. **.nemo archive loader** — `.nemo` files are gzipped tars containing: `model_config.yaml`, `model_weights.ckpt` (PyTorch pickle) or `.safetensors`, and one or more SentencePiece `.model` files. Implementation steps:
   - tar extract in-memory (`System.Formats.Tar`).
   - Parse `model_config.yaml` (simple YAML reader — already available for other configs).
   - Read `.ckpt` (PyTorch pickle) or `.safetensors` weights — prefer safetensors path if present; otherwise need a minimal pickle reader.
   - Extract `<hash>_tokenizer.model` (or per-language tokenizer .model files for v1).
   - HuggingFace also publishes `.nemo` only (not raw safetensors) — so the pickle path is unavoidable for full coverage, **or** we ship a conversion script (Python) that one-shots `.nemo` → safetensors and ship the converted weights. Recommend the latter for our first release; ship a converter, expect safetensors as the canonical input format.

#### Special-token vocabulary template (v1 / Flash — verify against extracted model)

Based on NeMo source (`canary` prompt format) and the v1 paper, expect these specials (exact IDs depend on checkpoint; treat as names):

| Name | Notes |
|------|-------|
| `<pad>` | Padding (often ID 0) |
| `<unk>` | Unknown |
| `<|startoftranscript|>` | BOS for the prompt |
| `<|endoftranscript|>` | EOS |
| `<|startofcontext|>` | v2 only — start of decoder-context block |
| `<|nospeech|>` | Emitted to indicate no speech |
| `<|transcribe|>` | Task = ASR |
| `<|translate|>` | Task = AST |
| `<|en|>`, `<|de|>`, `<|es|>`, `<|fr|>` | v1 langs |
| `<|bg|>`, `<|hr|>`, `<|cs|>`, `<|da|>`, `<|nl|>`, `<|et|>`, `<|fi|>`, `<|el|>`, `<|hu|>`, `<|it|>`, `<|lv|>`, `<|lt|>`, `<|mt|>`, `<|pl|>`, `<|pt|>`, `<|ro|>`, `<|sk|>`, `<|sl|>`, `<|sv|>`, `<|ru|>`, `<|uk|>` | v2 additional langs |
| `<|pnc|>`, `<|nopnc|>` | Punct+caps toggle |
| `<|timestamps|>`, `<|notimestamps|>` | Flash + v2 |
| `<|itn|>`, `<|noitn|>` | v2 only — inverse text norm |
| `<|diarize|>`, `<|nodiarize|>` | v2 only |
| `<|foreign|>`, `<|noforeign|>` | v2 only |
| `<|emo:undefined|>`, `<|emo:happy|>`, … | v2 only — emotion slots (mostly placeholder/undefined in published checkpoints) |
| `<|ts_0.00|>`, `<|ts_0.02|>`, … (or `_0.08|>` at 80 ms grid) | Flash + v2 timestamp grid |

Canary v2 has **1,162 special tokens total** (most are timestamp grid + placeholder slots for future expansion). Don't try to hardcode the table — generate it from the model at load time.

#### File format priority

| Format | Encoder | Decoder | Tokenizer | Use |
|--------|---------|---------|-----------|-----|
| `.nemo` (NeMo tar) | PyTorch pickle | PyTorch pickle | SentencePiece `.model` files | Source of truth; HF distribution format |
| Safetensors (converted) | Safetensors | Safetensors | SentencePiece `.model` | **Preferred** for SharpInference; ship a Python converter |
| GGUF | Quantized | Quantized | Token data block | Future, for quantized inference |

#### Implementation order (when building this in SharpInference.Audio)
1. Implement Whisper (encoder + decoder + greedy/beam). Validates the Transformer decoder + KV cache.
2. Implement Parakeet (FastConformer + CTC). Validates FastConformer encoder + mel preprocessor.
3. Implement Canary-1B-Flash (FastConformer encoder from Parakeet + Transformer decoder from Whisper + new prompt + concatenated tokenizer). This is the fastest payoff — most flexible (4 languages, ASR + AST + timestamps) and fastest variant.
4. Add Canary-1B-v2 (different vocab/tokenizer + 8-layer decoder + 25 languages + NFA aligner).
5. Add Canary-180M-Flash (same code as 1B-Flash, smaller hparams).

#### Validation targets (vs Python NeMo reference)
- LibriSpeech test-clean WER: 1.48% (Canary-1B-Flash), 1.87% (180M-Flash), 2.18% (v2).
- FLEURS En→Fr BLEU: ~41 (Canary-1B-Flash).
- Tolerance: WER within ±0.05% absolute, BLEU within ±0.3 absolute. Anything wider indicates a numeric/preprocessing/tokenizer bug.

## Open Questions

- **Canary-180M-Flash exact d_model / FFN / heads** — not published; extract from `.nemo`.
- **Exact timestamp-token grid** — 20 ms (Whisper) vs 80 ms (matches encoder frame rate)? Suspect 80 ms — but verify from extracted vocab.
- **Tied vs untied output projection** in the decoder — assume tied (standard NeMo); confirm from `model_weights.ckpt`.
- **Whether v2's auxiliary CTC head can be used for VAD** — likely yes, similar to Parakeet CTC. Would let us skip the encoder pass when no speech is detected.
- **Pickle reader** vs Python conversion script — decide which approach we want for first release. Conversion script is simpler short-term but adds a Python dependency for users.

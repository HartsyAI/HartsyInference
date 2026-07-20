# IndexTTS-2 — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: HartsyInference.Audio (IndexTTS pipeline)

## Summary

IndexTTS is a family of zero-shot voice-cloning text-to-speech systems developed by the **Bilibili Index Team**. It started as a heavy rework of Tortoise/XTTS (single-codebook codec + GPT-style autoregressive LM + neural vocoder) and evolved into a fully cascaded three-stage system. The two open-weight versions in scope for HartsyInference are:

- **IndexTTS-1.5** (May 2025) — single-codebook DVAE codec at 25 Hz, GPT-2-style 24-layer / 1280-dim / 20-head decoder generating mel codes, Conformer-Perceiver speaker conditioner, BigVGAN-v2 vocoder synthesising 24 kHz waveform directly from GPT output. Chinese + English. ~3.66 GB on disk.
- **IndexTTS-2** (Sept 2025, [arXiv:2506.21619](https://arxiv.org/abs/2506.21619)) — three modules trained separately: (T2S) the same GPT decoder enlarged to support an additional emotion-conditioning Perceiver, (S2M) a non-autoregressive **flow-matching DiT** that turns semantic codec tokens + speaker embedding + GPT latents into an 80-band mel at 22 050 Hz, and (vocoder) NVIDIA's pretrained `bigvgan_v2_22khz_80band_256x`. A separate fine-tuned **Qwen-3 0.6B** ("qwen0.6bemo4-merge") provides text-to-emotion-distribution control. Chinese + English (training corpus 55 000 h covers Chinese, English, Japanese; output quality is calibrated for ZH+EN). ~5.9 GB on disk.

The hallmark new capabilities of IndexTTS-2 are (a) **explicit duration control** via a token-count input that fixes the AR generation length without distorting prosody, (b) **disentangled emotion vs. speaker identity** via a Gradient-Reversal-Layer trick during T2S training, and (c) **natural-language emotion prompts** distilled from DeepSeek-R1 into a tiny LoRA-fine-tuned Qwen-3 0.6B. The system tops Chinese TTS benchmarks (WER on test-zh beats F5-TTS by ~0.5 pp and MaskGCT by ~2 pp) while matching F5-TTS on English.

For HartsyInference, IndexTTS-2 is a complex pipeline that **reuses several blocks we already have or are planning** — a causal GPT-2 style decoder (identical in shape to the native `HartsyInference.LLM` transformer), a DiT block stack (already implemented for Flux/SD3 in `HartsyInference.Diffusion`), a flow-matching Euler scheduler (already in `HartsyInference.Diffusion`, see [FLOW_MATCHING_AUDIO.md](FLOW_MATCHING_AUDIO.md)), a BigVGAN-v2 vocoder ([HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md)), and a Conformer encoder for the speaker-prompt perceiver. The genuinely new work is: a custom **semantic codec** (Vocos-style decoder over 8192 codebook entries, see [AUDIO_CODECS.md](AUDIO_CODECS.md)), a **wav2vec2-BERT** front end for semantic-token extraction during conditioning, the **Conformer-Perceiver** prompt module, and the BPE+pinyin **Chinese tokenizer**.

Sources: [IndexTTS-2 paper (arXiv 2506.21619)](https://arxiv.org/abs/2506.21619), [IndexTTS paper (arXiv 2502.05512)](https://arxiv.org/abs/2502.05512), [HF IndexTeam/IndexTTS-2](https://huggingface.co/IndexTeam/IndexTTS-2), [HF IndexTeam/IndexTTS-1.5](https://huggingface.co/IndexTeam/IndexTTS-1.5), [index-tts repo](https://github.com/index-tts/index-tts), [DeepWiki: index-tts](https://deepwiki.com/index-tts/index-tts), [IndexTTS-2 demo page](https://index-tts.github.io/index-tts2.github.io/), [BigVGAN-v2 22kHz 80band 256x](https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x), [Qwen3 0.6B-emo4-merge config](https://huggingface.co/IndexTeam/IndexTTS-2/tree/main/qwen0.6bemo4-merge).

## Detailed Findings

### 1. Variants

| Variant | Date | Params (active) | Output langs | Output SR | HF repo | Size on disk |
|---|---|---|---|---|---|---|
| **IndexTTS-1** (original) | Feb 2025 | ~750 M (GPT 700 M + DVAE 50 M + BigVGAN-v2 112 M) | ZH + EN | 24 kHz | [IndexTeam/Index-TTS](https://huggingface.co/IndexTeam/Index-TTS) | ~3.5 GB |
| **IndexTTS-1.5** | May 2025 | Same shape as v1; quality / English improvements via more training data | ZH + EN | 24 kHz | [IndexTeam/IndexTTS-1.5](https://huggingface.co/IndexTeam/IndexTTS-1.5) | 3.66 GB |
| **IndexTTS-2** | Sept 2025 | T2S GPT (~870 M) + emo Perceiver (~10 M) + S2M flow-DiT (~300 M) + BigVGAN-v2 (112 M) + Qwen-3 0.6B emo (~600 M) ≈ 1.9 B total | ZH + EN (training data also ja) | 22.05 kHz mel → vocoder | [IndexTeam/IndexTTS-2](https://huggingface.co/IndexTeam/IndexTTS-2) | 5.9 GB |
| IndexTTS-2.5 (technical report only, [arXiv 2601.03888](https://arxiv.org/abs/2601.03888)) | Dec 2025 | Codec compressed 50 Hz → 25 Hz; S2M U-DiT replaced with Zipformer; **2.28× faster RTF**; adds Japanese + Spanish in the official model | ZH + EN + ja + es | 22.05 kHz | not yet open-weights at time of writing | — |

The 1.5 GPT checkpoint is 1.17 GB (fp16/fp32 mix), the 2.0 GPT checkpoint is 3.48 GB because the embedding tables, emotion perceiver, and longer max-mel-tokens (1815 vs 800) all grew it. Parameter counts above are estimated from the config (see §2) and known checkpoint sizes; the paper does not publish a single "X parameters" headline.

### 2. Architecture

#### 2.1 Text Tokenizer (shared across all variants)

A single **SentencePiece BPE** tokenizer (`bpe.model`, 476 kB on both repos) covers Chinese characters + English wordpieces + pinyin tones + punctuation in **12 000 tokens** total. Layout per [DeepWiki: Tokenization](https://deepwiki.com/index-tts/index-tts/4.3-text-processing-and-normalization):

- ~8 400 individual CJK characters (one Chinese character ≈ one token).
- ~1 721 pinyin syllables with tone digits (token IDs 8474–10201 contain `pinyin1`/`pinyin2`/.../`pinyin5`).
- English/Latin BPE wordpieces.
- Punctuation tokens with explicit fixed IDs.
- Special tokens: `start_text_token = 0`, `stop_text_token = 1`. (Mel-side: `start_mel_token = 8192`, `stop_mel_token = 8193`.)

A `TextTokenizer` Python wrapper does CJK-aware pre/post-processing: `tokenize_by_CJK_char()` inserts spaces around CJK characters so BPE does not merge them with surrounding Latin text, and `de_tokenized_by_CJK_char()` reverses it. The pre-tokenizer also normalises punctuation, expands Arabic digits, and (optionally) converts user-supplied pinyin annotations into the pinyin-token range so the user can override default pronunciation. A separate `pinyin.vocab` file in `checkpoints/` lists all 1 729 valid pinyin+tone combinations.

In v2 the same `bpe.model` is reused unchanged.

#### 2.2 GPT Decoder (T2S module in v2)

A decoder-only causal Transformer, GPT-2 style, with separate embedding tables for text tokens and mel/semantic tokens. Identical hyperparameters in 1.5 and 2.0 (per `config.yaml`):

| Field | Value |
|---|---|
| `model_dim` | 1280 |
| `layers` | 24 |
| `heads` | 20 (head_dim = 64) |
| `number_text_tokens` | 12 000 |
| `number_mel_codes` | 8194 (= 8192 codebook + start + stop) |
| `max_text_tokens` | 600 |
| `max_mel_tokens` | 800 (v1.5) / **1815** (v2) |
| `mel_length_compression` | 1024 (i.e. one mel code spans 1024 raw 24 kHz samples ≈ 42.7 ms → 23.4 Hz code rate; the paper rounds to 25 Hz) |

Attention uses standard sinusoidal/RoPE-free relative positions (the original Tortoise GPT2 implementation; v2 keeps the same backbone). Sequence layout per training:

```
[start_text]  text_tokens...  [stop_text]  [speaker_cond_prefix...]
[start_mel]   mel_tokens...   [stop_mel]
```

The `speaker_cond_prefix` is a fixed-length sequence of conditioning vectors prepended after the text — see §2.3. In v2 there is also an `emotion_cond_prefix` from the emotion Perceiver and, optionally, a **duration token** that tells the model exactly how many mel tokens to emit.

**v2 additions inside the same backbone:**

- **GPT-latent tap.** The final transformer layer's hidden states are exported and fed downstream into the S2M flow matcher as an auxiliary conditioning stream (`gpt_dim: 1280`). This is what the paper calls "GPT Latent Enhancement" — it stabilises pronunciation during high-emotion synthesis.
- **Emotion Perceiver prefix.** A second, smaller Perceiver (output_size 512, 4 attention heads, 4 blocks; see `gpt.emo_condition_module` in the config) emits an emotion-style prefix that sits alongside the speaker prefix.
- **Gradient Reversal Layer (GRL).** During training a GRL with a speaker classifier is attached at the emotion prefix to remove speaker identity from the emotion path; symmetrically the speaker prefix has a GRL with an emotion classifier to remove emotion. Inference does not see the GRL.

#### 2.3 Speaker Conditioning — Conformer-Perceiver

`condition_type: "conformer_perceiver"` in `config.yaml`. The encoder is a 6-block Conformer (`output_size 512`, `attention_heads 8`, `linear_units 2048`, `input_layer "conv2d2"` → 2× subsample) running on the reference-audio mel spectrogram, followed by a Perceiver-Resampler (`perceiver_mult 2`) that distils a variable-length reference into a fixed-length sequence of 512-dim slots. The Perceiver output is projected to `model_dim=1280` and inserted as the speaker-conditioning prefix.

Per the IndexTTS-1 paper, this design beats both Tortoise's single d-vector and VALL-E's full-prompt approach: variable-length reference is supported (no length cap), and timbre similarity / output stability are noticeably better than CosyVoice's speaker embeddings. The Conformer block layout is the standard ESPnet recipe: conv-subsample → positional embedding → N × (FFN/2 + MHSA + Conv + FFN/2 + LN) with Macaron-style half-FFNs and a depthwise convolution module.

In v2 the same Conformer-Perceiver is reused; the emotion Perceiver is its own narrower copy operating on the **emotion-reference** mel (which can be a different audio clip than the speaker reference, enabling timbre/emotion disentanglement at inference time).

#### 2.4 Audio Codec — DVAE (v1.5) vs Semantic Codec + S2Mel (v2)

The codec story is **completely different** between v1.5 and v2 and is the single biggest implementation jump.

**v1.5 — DVAE (Discrete VAE) over mel-spectrogram.** Tortoise-style. `dvae.pth` is 243 MB. Config (from `vqvae:` block of v1.5 `config.yaml`):

| Field | Value |
|---|---|
| Input | 100-band mel at 24 kHz, hop=256 (≈ 93.75 Hz frame rate) |
| `num_tokens` (codebook size) | 8 192 |
| `hidden_dim` | 512 |
| `codebook_dim` | 512 |
| `num_resnet_blocks` | 3 |
| `num_layers` | 2 (× downsample) |
| `kernel_size` | 3, positional_dims 1 |

The DVAE downsamples mel 4× along time → ~23.4 Hz code rate (matches the `mel_length_compression=1024` in the GPT — 1024 samples ≈ 4 mel frames). The GPT emits codebook indices in `[0, 8192)` plus the start/stop sentinels; the BigVGAN-v2 decoder takes the GPT's last-layer **latent** (1280-dim per code) — *not* the integer codes — together with the speaker embedding and produces 24 kHz waveform directly. (`gpt_dim: 1280` in the bigvgan section of the config; `feat_upsample: false`; `cond_d_vector_in_each_upsampling_layer: true`.)

**v2 — Semantic Codec + Flow-Matching S2Mel + BigVGAN-v2.** The codec is now the front-end *and* a separate back-end:

- **Semantic codec front-end** (sits between the reference audio and the GPT to extract semantic tokens used as the GPT's mel-token vocabulary surrogate). Config:

  | Field | Value |
  |---|---|
  | `codebook_size` | 8 192 |
  | `hidden_size` | 1024 |
  | `codebook_dim` | 8 |
  | `vocos_dim` | 384 |
  | `vocos_intermediate_dim` | 2048 |
  | `vocos_num_layers` | 12 |

  The architecture is **Vocos-style** (a stack of ConvNeXt blocks producing magnitude/phase that an inverse STFT turns into audio), with the quantiser operating in an 8-dim subspace per the modern FSQ/low-dim-RVQ trend. Used together with a **wav2vec2-BERT** semantic-feature extractor — `wav2vec2bert_stats.pt` (9.3 kB) ships the mean/std normalisation statistics for the extractor's continuous features; the actual w2v-BERT weights are downloaded from HF on first use. The semantic-codec encoder is itself trained to **discretise the w2v-BERT features** at 25 Hz so the GPT can predict them as tokens.

- **S2Mel — Semantic-to-Mel flow-matching DiT** (`s2mel.pth`, 1.2 GB). Non-autoregressive. Generates an 80-band 22 050 Hz mel spectrogram conditioned on the semantic-token sequence, the speaker embedding (style_encoder dim 192), and the GPT-latent stream. Config (`s2mel.DiT` block):

  | Field | Value |
  |---|---|
  | `hidden_dim` | 512 |
  | `num_heads` | 8 |
  | `depth` | 13 |
  | `in_channels` | 80 (= mel bands) |
  | `content_dim` | 512 |
  | `content_codebook_size` | 1024 (residual content quantiser) |
  | `block_size` | 8192 |
  | `class_dropout_prob` | 0.1 (CFG training) |
  | `final_layer_type` | `wavenet` (8-layer WaveNet head, kernel 5, hidden 512) |
  | `long_skip_connection`, `uvit_skip_connection` | true (UNet-style skips through the transformer) |
  | `style_condition`, `is_causal`, `time_as_token`, `style_as_token` | true, false, false, false |

  The trajectory is rectified-flow / CFM (see [FLOW_MATCHING_AUDIO.md](FLOW_MATCHING_AUDIO.md)) integrated with Euler at **25 steps** by default. A `length_regulator` (channels 512, sampling_ratios [1,1,1,1], no VQ) upsamples the semantic-token sequence to the mel frame rate before the DiT runs.

- **BigVGAN-v2 vocoder** ([nvidia/bigvgan_v2_22khz_80band_256x](https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x)). 112 M params, MIT licence, 80-band mel → 22.05 kHz waveform, 256× upsampling, Snake-Beta activation, multi-scale sub-band CQT discriminator (training only). Loaded by reference — IndexTTS-2 does **not** ship this weight; it pulls it from the NVIDIA HF repo. See [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md) for the C# implementation plan.

#### 2.5 Text-to-Emotion (T2E) Module — Qwen-3 0.6B + LoRA-Merge

The T2E module lets a user type *"sound surprised but trying to stay calm"* and have the system produce an 8-way emotion vector consumed by the emotion Perceiver. Pipeline:

1. **Teacher distillation.** DeepSeek-R1 is used offline to label a corpus of natural-language emotion descriptions with 8-way probability vectors over `[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]`.
2. **Student fine-tune.** Qwen-3 0.6B is LoRA-fine-tuned to predict the same distribution, then the LoRA weights are merged → `qwen0.6bemo4-merge/`. From the shipped `config.json`:

   | Field | Value |
   |---|---|
   | `architectures` | `Qwen3ForCausalLM` |
   | `hidden_size` | 1024 |
   | `num_hidden_layers` | 28 |
   | `num_attention_heads` | 16 |
   | `num_key_value_heads` | 8 (GQA, 2:1 ratio) |
   | `head_dim` | 128 |
   | `intermediate_size` | 3072 |
   | `vocab_size` | 151 936 |
   | `max_position_embeddings` | 32 768 |
   | `rope_theta` | 1 000 000 |
   | `tie_word_embeddings` | true |
   | `torch_dtype` | bfloat16 |
   | model.safetensors size | 1.19 GB |

3. **Output mapping.** The Qwen-3 output produces an 8-way distribution which is **multiplied** with two precomputed feature matrices shipped in the repo:
   - `feat1.pt` (57 kB) — `spk_matrix`, the speaker-side projection.
   - `feat2.pt` (375 kB) — `emo_matrix`, the emotion-side projection.

   `config.yaml` declares `emo_num: [3, 17, 2, 8, 4, 5, 10, 24]` — these are the per-emotion sample counts that built the matrices, in the same order as the 8 emotions above (calm 24, surprised 10, melancholic 5, disgusted 4, afraid 8, sad 2, angry 17, happy 3 — yes, very imbalanced).

The user has **three** ways to drive emotion at inference time:
- Provide an **emotion-reference audio** (separate from the speaker reference) → goes through the emotion Perceiver directly.
- Provide an **explicit 8-dim emotion vector** → multiplied with `emo_matrix` to become the Perceiver-style prefix.
- Provide a **natural-language description** → Qwen-3 0.6B-emo predicts the 8-dim vector, then same as above.

The GRL inside the T2S GPT guarantees that whichever path is taken, **speaker identity stays glued to the speaker reference** and is not contaminated by the emotion path (and vice-versa).

#### 2.6 Duration Control

The T2S GPT can accept an optional **target token count** at the prompt boundary. Two generation modes:
- **Free-running.** No count supplied → GPT emits until `stop_mel_token` exactly like a normal autoregressive LM. Prosody comes from the reference.
- **Duration-constrained.** A count `N` is supplied → a "duration encoding" mechanism (positional bias + a small extra embedding table added to the mel-token stream) lets the model emit exactly `N` semantic tokens with smooth ending. The paper claims this is the first AR TTS to combine free-mode generation and exact-duration generation in a single model without quality loss.

Because the semantic codec rate is fixed at 25 Hz, `N` tokens = `N * 40 ms` of audio. To synthesize "exactly 8.000 seconds", pass `N = 200`.

### 3. Inference Pipeline (IndexTTS-2)

End-to-end forward pass for a single utterance:

1. **Text preprocess.** Normalise text (number expansion, punctuation), CJK-aware whitespace insertion, **optional pinyin annotation injection** (the user may pass `[pin1 yin1]` markers to force pronunciation), then SentencePiece-BPE encode against the 12 000-token vocab → `text_ids` ∈ `[2..11999]` plus `start_text_token=0` / `stop_text_token=1`.
2. **Speaker reference encode.** Compute mel (100-band, 24 kHz, hop=256, n_fft=1024) of the speaker reference clip → Conformer (6 blocks) → Perceiver (2× resampling multiplier) → fixed-length 512-dim slots → linear project to 1280 → **speaker prefix** tokens.
3. **Emotion reference encode** *(optional path A)*. Same mel as above on the emotion reference clip → emotion Conformer-Perceiver (4 blocks, 4 heads, output 512) → linear to 1280 → **emotion prefix**.
   - *Path B (vector):* `emo_vec(8) @ emo_matrix(feat2.pt)` → emotion prefix directly.
   - *Path C (text):* Qwen-3 0.6B-emo runs on the natural-language description with the bundled `chat_template.jinja`, decodes an 8-way probability vector, then Path B.
4. **w2v-BERT semantic features on the speaker reference** (only for the prompt portion that will seed the GPT) → semantic-codec encoder → discrete semantic tokens (`prompt_semantic_tokens`). These are prepended to the GPT's mel-token stream as the conditioning prefix the GPT will continue from.
5. **GPT autoregressive generation.** Input sequence: `[start_text, text_ids, stop_text, speaker_prefix, emotion_prefix, (optional duration token), start_mel, prompt_semantic_tokens]`. Sample mel/semantic tokens autoregressively with KV cache until `stop_mel_token` or, in duration-constrained mode, until exactly `N` tokens are emitted. Save the **final-layer hidden state** for every emitted position (`gpt_latents`, shape `(N, 1280)`).
6. **S2Mel flow matching.** Compose conditioning: `[semantic_tokens, gpt_latents, speaker_style(192), emotion_style]`. Initialise `x_1 ~ N(0, I)` of shape `(80, frames_target)` where `frames_target` is derived from the semantic-token count via the length regulator. Run 25 Euler steps of the velocity-prediction DiT (depth=13, hidden=512, heads=8) under CFG (the `class_dropout_prob=0.1` in training enables CFG at inference). Pass the result through the 8-layer WaveNet final layer → `mel_pred` (80-band, 22 050 Hz).
7. **BigVGAN-v2 vocoder.** `mel_pred` → `bigvgan_v2_22khz_80band_256x` → 22 050 Hz mono PCM. Done.

The pipeline for **v1.5** is much simpler: text → BPE → GPT (using the DVAE codebook as the mel-token vocabulary) → GPT-latents → BigVGAN-v2 (custom-trained, 24 kHz, 100-band, ships in the repo). No semantic codec, no S2Mel flow matching, no emotion modules, no duration control.

### 4. Reference Audio Requirements

- **Length.** 5–10 s of clean speech is the documented sweet spot. The Conformer-Perceiver imposes no hard cap, so multi-clip references (concatenate several utterances of the same speaker) measurably improve similarity; 30 s of reference is typical for production clones.
- **Quality.** Studio-clean is best. Music/background noise contaminates both timbre (speaker prefix) and prosody (the `prompt_semantic_tokens` that seed the GPT).
- **Sample rate.** Internally resampled to 24 kHz mono for the Conformer's mel front end. Any input SR works.
- **Language.** Per the official model card the *reference* can be **any language** even though the *output* is restricted to ZH+EN; this enables cross-lingual cloning (Spanish reference → English output in the target speaker's voice).
- **Emotion reference (v2 only).** A separate clip if you want to dictate emotion explicitly; otherwise omit and the emotion vector defaults to neutral.

### 5. Streaming

**IndexTTS-2 base release does not support true streaming.** Two reasons:

- The S2Mel **flow-matching DiT is non-causal** (`is_causal: false` in the config) and operates on the full semantic-token sequence at once, so the first mel frame cannot be emitted until the entire GPT generation finishes and the 25-step ODE has converged on the whole utterance.
- The BigVGAN-v2 vocoder is a non-streaming 256× upsampler; it can be chunked with overlap-add but is not natively causal.

In practice the best **latency strategy** at inference time is **sentence-level pipelining**:
1. Split input text by sentence boundaries.
2. Run the full T2S → S2Mel → BigVGAN pipeline per sentence in serial; emit each sentence's PCM as it completes.
3. Optionally prepend the previous sentence's last ~0.5 s of prompt-semantic tokens to the next sentence to maintain prosodic continuity.

A community vLLM fork ([Ksuriuri/index-tts-vllm](https://github.com/Ksuriuri/index-tts-vllm)) accelerates the T2S step substantially but still cannot deliver true frame-by-frame streaming.

**IndexTTS-2.5** (technical report only, not yet open weights as of this writing) addresses streaming by (a) compressing the semantic codec from 50 Hz to 25 Hz (halving the AR length), and (b) replacing the U-DiT S2M backbone with a **causal Zipformer** that allows chunked generation. RTF goes from 0.310 → 0.136 (2.28×). When 2.5 weights drop, HartsyInference.Audio should adopt the Zipformer path for live use; until then plan only for **batched / sentence-level** generation with first-audio latency ≈ 0.3–1.5 s on a desktop GPU.

See [STREAMING_AUDIO_INFERENCE.md](STREAMING_AUDIO_INFERENCE.md) for HartsyInference's streaming-vs-batched abstraction (`IStreamingPipeline<TIn,TOut>`) — IndexTTS-2 will implement the non-streaming `IBatchedPipeline` surface and expose sentence-level `IAsyncEnumerable<AudioChunk>` via a sentence-splitting wrapper, not via in-model streaming.

### 6. Emotion and Style Control

Three independent surfaces, all routed through the **emotion Perceiver prefix** that augments the T2S GPT:

| Surface | How user supplies it | Goes through |
|---|---|---|
| **Audio prompt** | Path to a `.wav` displaying the target emotion (different speaker OK) | Mel → 4-block Conformer (output 512, 4 heads) → 2× Perceiver → 1280-dim prefix |
| **Explicit vector** | Length-8 float vector over `[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]`, typically summing to 1 | `vec @ feat2.pt (emo_matrix)` → prefix |
| **Natural-language description** | `"speak softly with a tinge of sadness"` plus the target text segment | Qwen-3 0.6B-emo (LoRA-merged) decodes 8-vec → `vec @ feat2.pt` → prefix |

The **disentanglement guarantee** is delivered structurally at training time: a Gradient Reversal Layer plus a small speaker classifier sits on top of the emotion-prefix output and forces it to be speaker-invariant; symmetrically a GRL+emotion-classifier sits on the speaker prefix. At inference both classifiers are dropped. The architectural consequence is that you can take **speaker A**'s timbre and apply **speaker B**'s emotional delivery (or a vector / text emotion) without bleed-through — this is the main quality claim of the paper.

The natural-language route is convenient but not necessary: if HartsyInference does not want to ship a 1.19 GB Qwen-3 0.6B alongside the TTS, the **vector** path is a clean, deterministic, dependency-free alternative that exposes the same expressive range.

### 7. HuggingFace Files

**IndexTTS-2** (`IndexTeam/IndexTTS-2`, total 5.9 GB):

| File | Size | Purpose |
|---|---|---|
| `config.yaml` | 2.88 kB | All hyperparameters (mel front-end, GPT, semantic_codec, s2mel, vocoder) |
| `bpe.model` | 476 kB | SentencePiece BPE, 12 000 vocab, ZH/EN/pinyin tokens |
| `gpt.pth` | **3.48 GB** | T2S Transformer (24L × 1280, 20 heads) + Conformer-Perceiver speaker conditioner + 4-block emotion Perceiver |
| `s2mel.pth` | **1.20 GB** | Flow-matching DiT (13L × 512, 8 heads) + length regulator + WaveNet final layer + style encoder + semantic codec (Vocos decoder) |
| `wav2vec2bert_stats.pt` | 9.34 kB | Per-channel mean/std normalisation for w2v-BERT features (model weights pulled from HF on demand) |
| `feat1.pt` | 57.2 kB | `spk_matrix` — projection from 8-dim emotion vector to a speaker-side bias |
| `feat2.pt` | 375 kB | `emo_matrix` — projection from 8-dim emotion vector to the emotion Perceiver input |
| `qwen0.6bemo4-merge/model.safetensors` | **1.19 GB** | Qwen-3 0.6B fine-tuned (LoRA-merged) to predict 8-way emotion distributions from text |
| `qwen0.6bemo4-merge/config.json` | 727 B | Qwen-3 config: 28 layers, 1024 hidden, 16 heads, 8 KV-heads, head_dim 128, intermediate 3072, vocab 151 936, RoPE θ=1e6, bf16 |
| `qwen0.6bemo4-merge/tokenizer.json` | 11.4 MB | Qwen-3 tokenizer (BPE with merges) |
| `qwen0.6bemo4-merge/vocab.json` | 2.78 MB | Vocab |
| `qwen0.6bemo4-merge/merges.txt` | 1.67 MB | BPE merges |
| `qwen0.6bemo4-merge/added_tokens.json` | 707 B | Special tokens added during fine-tune |
| `qwen0.6bemo4-merge/chat_template.jinja` | 550 B | Chat template for emotion-prompt formatting |
| `qwen0.6bemo4-merge/special_tokens_map.json` | 616 B | Standard special-token mapping |
| `qwen0.6bemo4-merge/tokenizer_config.json` | 5.43 kB | Tokenizer config |
| `qwen0.6bemo4-merge/generation_config.json` | 117 B | Default generation params |
| `qwen0.6bemo4-merge/Modelfile` | 360 B | Ollama Modelfile |
| `LICENSE.txt`, `LICENSE_ZH.txt`, `README.md`, `.gitattributes` | 22 kB | Metadata |

Not in the repo: `bigvgan_v2_22khz_80band_256x` (the vocoder weights live at [`nvidia/bigvgan_v2_22khz_80band_256x`](https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x), MIT licence, ~445 MB) and the w2v-BERT base weights (pulled from HF on first use; `facebook/w2v-bert-2.0`, ~580 M params).

**IndexTTS-1.5** (`IndexTeam/IndexTTS-1.5`, total 3.66 GB):

| File | Size | Purpose |
|---|---|---|
| `config.yaml` | 2.49 kB | Hyperparameters (mel, GPT, vqvae, bigvgan) |
| `bpe.model` | 476 kB | Same SentencePiece BPE as v2 |
| `unigram_12000.vocab` | 94.7 kB | Human-readable vocab dump (debug aid; not strictly needed at runtime) |
| `gpt.pth` | 1.17 GB | GPT (24L × 1280, 20 heads) + Conformer-Perceiver speaker conditioner |
| `dvae.pth` | 243 MB | DVAE codec (100-band mel → 8192-entry codebook at ~23 Hz) |
| `bigvgan_generator.pth` | 536 MB | Custom-trained BigVGAN-v2 24 kHz generator (100-band, 4×4×4×4×2×2 upsample = 1024× hop, gpt_dim=1280 conditioning, snake-beta activation, CQT discriminator on the training side) |
| `bigvgan_discriminator.pth` | 1.65 GB | Training-only; **not needed for inference** — can be omitted from the HartsyInference distribution |

### 8. Memory and Performance

**Disk + VRAM (fp16/bf16):**

| Component | Disk | VRAM-resident (fp16) |
|---|---|---|
| **IndexTTS-1.5 total inference** | 1.95 GB (drop the discriminator) | ~2.5 GB |
| **IndexTTS-2 total inference** | 5.9 GB + ~445 MB BigVGAN-v2 + ~1.1 GB w2v-BERT = ~7.4 GB | ~6–8 GB |
| └ T2S GPT-2 (24L×1280) | 3.48 GB → 1.7 GB fp16 | ~1.8 GB + KV cache (peaks ~1.5 GB for max-len) |
| └ S2Mel flow-DiT (13L×512) | 1.2 GB → 0.6 GB fp16 | ~0.7 GB + flow activations |
| └ Qwen-3 0.6B-emo | 1.19 GB bf16 already | ~0.7 GB (optional) |
| └ BigVGAN-v2 vocoder | 445 MB → 224 MB fp16 | ~0.4 GB |
| └ w2v-BERT 2.0 | 1.1 GB → 580 MB fp16 | ~0.7 GB (only during reference-encode) |

A 12 GB GPU (RTX 3060, RTX 4070) runs IndexTTS-2 comfortably; an 8 GB card requires offloading w2v-BERT and Qwen-3 to CPU between uses.

**Latency / RTF** (paper + community measurements, A100/H100 class GPU, batch 1):

| Stage | RTF (audio_secs / processing_secs ⁻¹) |
|---|---|
| T2S (GPT autoregressive)  | ~0.232 — dominant cost; scales with output length |
| S2Mel (25-step flow ODE) | ~0.060 |
| BigVGAN-v2 vocoder | ~0.018 |
| **Total** | **~0.31** (A100) |

On a RTX 3060 12 GB the community reports RTF ≈ 13× without optimisation (one second of audio takes ~13 s wall-clock); with FlashAttention + bf16 + KV-cache tricks this falls below 2×. **First-audio latency** for a single sentence on an A100 is roughly 0.3–0.5 s (sentence of ~3 s); on consumer hardware expect 1–5 s.

**Training scale** (for reference, not implementer-relevant): 55 000 hours of multilingual ZH/EN/JA, 8× A100 80 GB, AdamW lr=2e-4, three weeks of training.

### 9. C# Implementation Notes for HartsyInference

Mapping every IndexTTS-2 component onto HartsyInference packages:

#### 9.1 HartsyInference.Audio.IndexTTS (new package)

Top-level orchestrator class `IndexTtsPipeline` with constructor taking the model directory. Owns:

- `IndexTextTokenizer` — SentencePiece BPE loader (see [TOKENIZERS.md](TOKENIZERS.md)). Handle CJK whitespace injection and pinyin-bracket annotation expansion in pure C# (`Span<char>` based). The SentencePiece `.model` file is a flatbuffers blob; a hand-rolled loader is small (≈ 1 kLOC) and is already needed for several other models. Vocabulary size 12 000, special tokens hard-coded (`start_text=0`, `stop_text=1`).
- `SpeakerEncoder` — mel front-end (100-band, n_fft 1024, hop 256, 24 kHz; reuse [MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md)) + 6-block Conformer + Perceiver-Resampler. The Conformer is new; budget ~600 LOC of C# (subsample-conv2d2 → MHSA + relative-pos + depthwise-conv + Macaron FFNs). Perceiver-Resampler is ~150 LOC.
- `EmotionEncoder` — same shape as `SpeakerEncoder` but 4 blocks / 4 heads / output 512. Can share the Conformer block implementation.
- `OptionalQwenEmotionModel` — gated behind a build flag. Reuse the **native `HartsyInference.LLM` Qwen3 implementation directly**: drop the 1.19 GB `model.safetensors` into the engine's config-driven Qwen3 decoder with `hidden=1024, layers=28, heads=16, kv_heads=8, head_dim=128, vocab=151936, intermediate=3072, rope_theta=1e6, tie_word_embeddings=true`. Inference does a single greedy/argmax decode of the 8-emotion-token output. Document the chat template format from `chat_template.jinja` for the prompt. **Recommendation: do not bundle by default** — most users will pass either a reference clip or an explicit 8-vector; ship Qwen-3-emo as a separately downloadable add-on.
- `T2sGptDecoder` — causal Transformer, 24 × 1280 / 20 heads. Architecture is a **plain GPT-2** (sinusoidal positional embeddings, no RoPE, no GQA, full attention). Reuse a shared GPT decoder block if the engine already has one; otherwise this is ~400 LOC of standard MHSA + FFN. Must support **prefix conditioning** (the speaker/emotion prefixes are continuous 1280-dim vectors inserted *between* the text tokens and the mel-start token — they are not in the embedding table). Implement the prefix path via a `prepend_continuous(Tensor prefix)` API on the KV-cache builder. Sampling: top-p, top-k, repetition penalty (Tortoise/XTTS defaults). Stop at `stop_mel_token = 8193` or at `max_mel_tokens = 1815`. Must export the **final-layer hidden state per emitted position** for downstream S2Mel — make this an opt-in tap to avoid extra memory traffic when not needed.
- `SemanticCodec` — Vocos-style decoder (12 ConvNeXt blocks, dim 384, intermediate 2048) over an 8192-entry codebook with codebook_dim 8. The encoder side runs only on the reference clip during conditioning. Both halves are pure-C# CNNs; reuse [AUDIO_CODECS.md](AUDIO_CODECS.md)'s Vocos decoder description (the SNAC / WavTokenizer pattern is structurally identical).
- `W2vBert2Extractor` — a 580 M-param wav2vec2-BERT encoder used only to extract continuous features that the semantic codec then quantises. **Pre-quantise the reference clip's semantic tokens once** at session start so this network does not run per-token. Architecture: a CNN feature extractor + 24-block Conformer-style Transformer; substantial work (~1500 LOC). Mitigation: cache reference-derived semantic tokens to disk keyed by audio hash.
- `S2MelFlowMatcher` — the 13-block × 512-dim DiT. **Reuse the Flux/SD3 DiT block implementation in HartsyInference.Diffusion** with the following deltas: (a) `in_channels=80` instead of image latent channels, (b) two skip-connection patterns (`long_skip_connection` + `uvit_skip_connection`) — both are simple add-after-N-blocks patterns, (c) WaveNet final layer (8 dilated-conv blocks, kernel 5, hidden 512, gated tanh×sigmoid activation), (d) style conditioning concatenated to every block's input rather than via AdaLN — read the s2mel checkpoint to confirm. Scheduler: `FlowMatchEulerDiscreteScheduler` already exists; 25 steps default. CFG is supported (`class_dropout_prob=0.1` trained the unconditional path) — reuse the velocity-field CFG combiner from [FLOW_MATCHING_AUDIO.md](FLOW_MATCHING_AUDIO.md).
- `BigVGanV2Vocoder` — reuse the BigVGAN section of [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md). Snake-Beta activation is non-trivial CUDA work (we already need it for several other models). The 22 kHz / 80-band / 256× variant is the **NVIDIA pretrained** weight — separate small download manager class.

#### 9.2 Configuration loader

`config.yaml` is plain YAML. Use the existing YamlDotNet dependency (already in the solution per `NUGET_PACKAGE_DESIGN.md`). The config schema maps cleanly to a strongly-typed `IndexTtsConfig` record with nested `Gpt`, `SemanticCodec`, `S2Mel.Dit`, `S2Mel.Wavenet`, `Vocoder` records.

#### 9.3 Weight loading

Both `.pth` files are PyTorch pickle archives (zip-of-pickled-tensors). We already have a pickle reader in `HartsyInference.Common.Weights` (used for Kokoro and other PyTorch-checkpoint models). The state-dict keys follow standard PyTorch module-path convention — write a per-component key-prefix mapper (`gpt.*` → `T2sGptDecoder`, `conditioning_encoder.*` → `SpeakerEncoder`, `emo_encoder.*` → `EmotionEncoder`). The Qwen-3 sub-model is in `model.safetensors` (standard safetensors layout — reuse [SAFETENSORS_FORMAT.md](SAFETENSORS_FORMAT.md)). `feat1.pt` and `feat2.pt` are simple 2-D float matrices — load them as `Tensor<float>` constants on the device and use them via a fused matmul in the emotion path.

#### 9.4 Chinese tokenizer notes

The SentencePiece BPE alone handles CJK fine **provided** the pre-tokenizer inserts whitespace around CJK characters before BPE — otherwise the BPE will merge punctuation/spaces into the Chinese tokens incorrectly. Implement `TokenizeByCjkChar(ReadOnlySpan<char>)` returning a `string` with spaces inserted at every CJK / non-CJK boundary, and the inverse for detokenisation. The pinyin range `[8474..10201]` is documented in `pinyin.vocab` (94.7 kB ASCII file in the v1.5 repo and shipped in `checkpoints/` in the GitHub repo); load it as a `Dictionary<string,int>` for explicit pinyin-bracket annotation expansion. Tone marks: numbers 1–5 follow each syllable (e.g. `zhong1 guo2`).

#### 9.5 Validation tolerances

Per project rule "validate against references": compare on three fronts:
- **GPT token output** against the Python reference for a fixed seed + fixed input → expect *bit-identical* mel-token sequence (deterministic argmax / fixed-seed top-p).
- **Mel spectrogram** out of the S2Mel ODE → within `1e-3` mean-abs-diff after CFM-Euler at the same step count (flow matching is sensitive to step roundoff; bf16 vs fp32 will diverge after step 10).
- **Final waveform PSNR** ≥ 35 dB vs the Python reference for the same speaker+text input. BigVGAN-v2 has a known small non-determinism in its CUDA path so a small numerical gap is acceptable.

#### 9.6 Phased build order

1. **Phase 1 — IndexTTS-1.5 path only.** Skip semantic codec, S2Mel, w2v-BERT, Qwen-3, emotion. This gets a working ZH+EN voice clone with ~25% of the implementation work. Reuses everything we already need for Kokoro plus the GPT-2 + Conformer-Perceiver + DVAE + BigVGAN-v2 (24 kHz custom variant). Output: 24 kHz mono.
2. **Phase 2 — Add IndexTTS-2 pipeline.** Implement semantic codec (Vocos-style), wav2vec2-BERT extractor, S2Mel flow-matching DiT, the NVIDIA 22 kHz BigVGAN-v2. Output: 22.05 kHz mono with non-emotional, non-duration-controlled cloning. Quality should already beat IndexTTS-1.5.
3. **Phase 3 — Emotion control surfaces.** Wire up the emotion Perceiver, `feat1.pt`/`feat2.pt` matrix paths, and the **vector** and **audio-prompt** emotion APIs. Skip Qwen-3 for now.
4. **Phase 4 (optional add-on) — Qwen-3 0.6B-emo text-to-emotion.** Ship as a separately-downloadable extension package `HartsyInference.Audio.IndexTTS.EmotionText` to keep the base distribution small.
5. **Phase 5 (when IndexTTS-2.5 open-weights arrive).** Replace S2Mel with the streaming Zipformer variant, drop semantic-codec frame rate to 25 Hz, expose true `IAsyncEnumerable<AudioChunk>` streaming.

#### 9.7 What we deliberately do not implement

- **Training/fine-tuning paths.** Discriminators, GAN losses, GRL+classifier heads, EMA codebook update, KL warm-up — all training-only, omit from the inference distribution.
- **vLLM-style paged attention.** Out of scope for HartsyInference's first release; revisit when HartsyInference.API adds batched serving.
- **CUDA fused kernels for Snake-Beta.** The NVIDIA BigVGAN repo provides one for 1.5–3× speedup; write a PTX equivalent only after the pure-C# pipeline lands and we have a baseline RTF to compare against.

---

## Appendix A — Cross-reference summary for HartsyInference

| External dep | HartsyInference doc |
|---|---|
| Mel spectrogram (24 kHz / 100-band, 22.05 kHz / 80-band) | [MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md) |
| Semantic codec (Vocos-style decoder), DVAE quantiser | [AUDIO_CODECS.md](AUDIO_CODECS.md) |
| BigVGAN-v2 vocoder (custom 24 kHz and NVIDIA 22 kHz) | [HIFIGAN_VOCODER.md](HIFIGAN_VOCODER.md) |
| Flow-matching ODE solver, CFG, sway-sampling option | [FLOW_MATCHING_AUDIO.md](FLOW_MATCHING_AUDIO.md) |
| Streaming abstractions (`IStreamingPipeline`, `AudioRingBuffer`) | [STREAMING_AUDIO_INFERENCE.md](STREAMING_AUDIO_INFERENCE.md) |
| SentencePiece BPE loader, CJK tokenization | [TOKENIZERS.md](TOKENIZERS.md) |
| Safetensors loader (for Qwen-3 emo sub-model) | [SAFETENSORS_FORMAT.md](SAFETENSORS_FORMAT.md) |
| DiT block reuse from Flux/SD3 | [SD3_ARCHITECTURE.md](SD3_ARCHITECTURE.md), [FLUX_ARCHITECTURE.md](FLUX_ARCHITECTURE.md) |
| Qwen-3 0.6B (architecture identical to the native LLM path) | [DOTLLM_ARCHITECTURE.md](DOTLLM_ARCHITECTURE.md) |
| (Comparison) StyleTTS-2-class non-AR TTS | [KOKORO_ARCHITECTURE.md](KOKORO_ARCHITECTURE.md) |

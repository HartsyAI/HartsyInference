# Higgs Audio v2 — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: SharpInference.Audio (Higgs Audio pipeline)

## Summary

Higgs Audio v2 (Boson AI, released July 2025 under Apache 2.0) is an audio foundation model that turns a Llama-3.2-3B text LLM into a multimodal speech generator. The model is **autoregressive**: it predicts a sequence of discrete audio codec tokens (8 parallel codebooks per audio frame) interleaved with text in a ChatML-style conversation, then a separate **HiggsAudioV2Tokenizer** (a custom dual semantic + acoustic codec) decodes those tokens into a 24 kHz waveform. Three architectural pieces define v2: (1) a stock Llama-3.2-3B backbone (28 layers, 3072 hidden, GQA 24/8 heads, RoPE-llama3) extended with audio-stream BOS/EOS, audio placeholder, and an audio-delay token in the text vocab; (2) a **DualFFN audio adapter** — for audio positions only, a second parallel FFN block (2.2 B extra params) runs alongside the standard Llama FFN, giving the LM a dedicated acoustic expert at minimal compute overhead; (3) the **unified tokenizer**, a HuBERT-base "semantic" branch (16 kHz, 50 Hz) fused with a DAC-style RVQ "acoustic" branch (24 kHz, 25 Hz, 12 codebooks × 1024) — 2 kbps total, but the LM only consumes/produces 8 of those codebooks (`num_codebooks=8`, `codebook_size=1024` per the actual `config.json`).

The pipeline uniquely supports four modes from one checkpoint via the chat template alone — **single-speaker smart voice**, **multi-speaker dialogue** (`[SPEAKER0]`/`[SPEAKER1]` tags), **zero-shot voice cloning** (reference audio in assistant role), and **multi-speaker voice cloning** (per-speaker reference audio in scene role). Generation uses standard Llama sampling plus a custom **RAS (Repetition-Aware Sampling)** logits processor (`ras_win_len=7`, `ras_win_max_num_repeat=2`) to suppress repetitive audio loops. Codebooks are arranged in a MusicGen-style **delay pattern** so all 8 streams can be predicted in parallel per LM step.

Higgs v2.5 (Sep 2025) is a 1B-parameter condensation with the same tokenizer and chat template but stronger primary-language coverage (en/zh/ko/ja via GRPO) and explicit expressiveness control tags.

For SharpInference this maps cleanly to: **dotLLM patterns for the Llama-3.2-3B backbone + a new DualFFN MLP variant; a new audio codec implementation in SharpInference.Audio that combines DAC-style decoder ops (already documented in [AUDIO_CODECS.md](AUDIO_CODECS.md)) with a HuBERT-style semantic encoder; pure string templating for prompt construction; KV-cache + `IAsyncEnumerable<float[]>` for streaming.**

Sources: [boson-ai/higgs-audio (GitHub)](https://github.com/boson-ai/higgs-audio), [bosonai/higgs-audio-v2-generation-3B-base (HF)](https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base), [bosonai/higgs-audio-v2-tokenizer (HF)](https://huggingface.co/bosonai/higgs-audio-v2-tokenizer), [HiggsAudioV2 transformers docs](https://huggingface.co/docs/transformers/model_doc/higgs_audio_v2), [Boson AI v2 blog](https://www.boson.ai/blog/higgs-audio-v2), [Boson AI v2.5 blog](https://www.boson.ai/blog/higgs-audio-v2.5), [erogol model-check writeup](https://erogol.substack.com/p/model-check-higgs-audio-v2-unified).

## Detailed Findings

### 1. Variants

| Variant | Params | Base LLM | Release | HF Path | License |
|---|---|---|---|---|---|
| Higgs Audio v1 (Understanding) | n/a (not open-sourced for generation) | — | 2024 | (internal; used as teacher for v2 annotation) | — |
| **Higgs Audio v2 — Generation 3B base** | **3.6 B LLM + 2.2 B DualFFN ≈ 5.8 B total** | **Llama-3.2-3B** | **Jul 2025** | **`bosonai/higgs-audio-v2-generation-3B-base`** | **Apache-2.0** |
| Higgs Audio v2 Tokenizer | ~600 M acoustic (DAC) + ~95 M semantic (HuBERT-base) | (standalone codec) | Jul 2025 | `bosonai/higgs-audio-v2-tokenizer` | Apache-2.0 |
| Higgs Audio v2.5 | 1 B condensed | Llama-derived (smaller) | Sep 2025 | `bosonai/HiggsAudioV2.5` (limited release; also on Microsoft Foundry / DeepInfra / Eigen) | Apache-2.0 (per Boson blog) |
| Higgs Audio v3 STT | 2.68 B total (Whisper-Large-v3 enc + Qwen3 dec) | Qwen3 | 2026 | `bosonai/higgs-audio-v3-stt` | Apache-2.0 |

**Languages.** v2 was pretrained on AudioVerse (10 M hours) with English dominant, plus Chinese (Mandarin), Korean, German, Spanish; in practice the released checkpoint supports en/zh well and many others zero-shot. v2.5 formalises primary language support (en/zh/ko/ja via GRPO) and secondary (es/de/fr/it via zero-shot generalization).

**File sizes (v2-generation-3B-base).** Total ≈ 23 GB. Weights are duplicated as one consolidated and one 3-shard form:

| File | Size | Purpose |
|---|---|---|
| `model.safetensors` | 11.5 GB | Consolidated weights (BF16) |
| `model-00001-of-00003.safetensors` | 4.97 GB | Shard 1 |
| `model-00002-of-00003.safetensors` | 4.98 GB | Shard 2 |
| `model-00003-of-00003.safetensors` | 1.59 GB | Shard 3 |
| `model.safetensors.index.json` | 31.1 kB | Shard index |
| `config.json` | 1.1 kB | Model architecture config |
| `generation_config.json` | 351 B | Default sampling params |
| `processor_config.json` | 682 B | Processor config |
| `chat_template.jinja` | 3.05 kB | ChatML template with scene/audio handling |
| `tokenizer.json` | 17.2 MB | Llama-3 BPE vocab (128 256 tokens) |
| `tokenizer_config.json`, `special_tokens_map.json` | <1 kB each | Tokenizer metadata |
| `LICENSE` | 9.17 kB | Apache-2.0 |

**File sizes (v2-tokenizer).** Total ≈ 12.3 GB:

| File | Size | Purpose |
|---|---|---|
| `model.safetensors` | 11.5 GB | Combined acoustic+semantic weights |
| `model.pth` | 806 MB | Original PyTorch pickle (acoustic-only? both branches FP32 vs BF16 difference) |
| `config.json` | 2.53 kB | Dual-branch tokenizer config (acoustic DAC + semantic HuBERT) |
| `preprocessor_config.json` | 206 B | Audio feature-extractor config |

> Note: the consolidated `model.safetensors` is larger than the sharded form sums because it stores extra tied/duplicate tensors; an implementation only needs **one** of the two forms.

### 2. Architecture

#### 2.1 LLM Backbone (Llama-3.2-3B + audio extensions)

From the actual `config.json` (verbatim):

```json
{
  "architectures": ["HiggsAudioV2ForConditionalGeneration"],
  "model_type": "higgs_audio_v2",
  "hidden_size": 3072,
  "num_hidden_layers": 28,
  "num_attention_heads": 24,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "intermediate_size": 8192,
  "hidden_act": "silu",
  "rms_norm_eps": 1e-05,
  "vocab_size": 128256,
  "max_position_embeddings": 2048,
  "rope_parameters": {
    "factor": 32.0,
    "high_freq_factor": 0.5,
    "low_freq_factor": 0.125,
    "original_max_position_embeddings": 1024,
    "rope_theta": 500000.0,
    "rope_type": "llama3"
  },
  "attention_bias": false,
  "mlp_bias": false,
  "tie_word_embeddings": false,
  "dtype": "bfloat16",

  "num_codebooks": 8,
  "codebook_size": 1026,
  "audio_token_id": 128016,
  "audio_bos_token_id": 128013,
  "audio_delay_token_id": 128014,
  "audio_stream_bos_id": 1024,
  "audio_stream_eos_id": 1025,
  "bos_token_id": 1,
  "eos_token_id": 128009,
  "pad_token_id": 128001
}
```

**Confirmation vs. task brief.** The brief listed `28 layers, 24 heads, head_dim=128, ffn=8192` — confirmed. `hidden=3072` — confirmed. The brief said `max_position_embeddings=2048` is the *positional embedding table* size; with `rope_type=llama3` scaling (`factor=32`, `original_max=1024`) the effective context after RoPE scaling is `1024 × 32 = 32 768` positions. **GQA: 24 query heads / 8 KV heads (3× ratio)** — standard Llama-3.2 spec. **`tie_word_embeddings=false`** — separate embedding and LM head (so two `(128256, 3072)` matrices live in the weights).

**Audio token extensions in the text vocab.** Llama-3.2-3B's vocab is 128 256 tokens. Higgs repurposes the reserved-special-token slots:

| Token | ID | String form | Role |
|---|---|---|---|
| `<|AUDIO_OUT|>` | 128016 | placeholder | Marks positions in the text sequence where an audio frame's worth of codebook embeddings is spliced in (one position per audio frame). |
| `<|audio_out_bos|>` | 128013 | wrapper | Emitted just before the first audio frame of an assistant turn. |
| `<|audio_eos|>` | — (string token) | wrapper | Closes an audio span. |
| `<|reserved_special_token_6|>` | 128014 | "audio delay" | Filler token used in the delay-pattern column when no real audio token exists yet. |
| `<|scene_desc_start|>` / `<|scene_desc_end|>` | string tokens | Wrap the scene block in the rendered prompt. |
| `<|start_header_id|>` / `<|end_header_id|>` / `<|eot_id|>` | Llama-3 standard | Chat role headers and turn terminators. |

**Per-codebook stream tokens.** Each of the 8 audio codebooks has size **1026 = 1024 (real entries) + 1 (stream BOS, ID 1024) + 1 (stream EOS, ID 1025)**. Note the discrepancy between `config.json` (`codebook_size: 1026` — the embedding-table size including BOS/EOS) and the transformers docstring default (`codebook_size: 1024` — the codec's real entry count). The model uses ID 1024 to start an audio stream and ID 1025 to end it.

**DualFFN audio adapter.** This is the v2-specific architectural change. Standard Llama-3 layer = `RMSNorm → GQA → Add → RMSNorm → SwiGLU MLP → Add`. In Higgs each layer additionally carries a **second SwiGLU MLP** (the "audio FFN") with the same shape `(3072 → 8192 → 3072)`. At each layer the routing is **per-token**:

- For positions whose token came from the text vocab → run the standard text MLP.
- For positions whose token came from the audio vocab (the `<|AUDIO_OUT|>` placeholder positions) → run the audio MLP.

Because routing is by token type (not learned MoE), it's deterministic, adds zero gating compute, and the audio MLP only fires on audio positions — so the FLOP overhead during text-heavy prompts is near-zero. Boson reports the implementation preserves **~91 % of original Llama-3.2-3B training throughput**. Parameter cost: each layer adds one full Llama SwiGLU MLP — `2 × (3072 × 8192) + (8192 × 3072) ≈ 75.5 M` params/layer, × 28 layers ≈ 2.11 B parameters (matches the published "2.2 B DualFFN" figure including the per-codebook audio heads).

**Audio embedding and prediction.** For audio frames the model uses 8 parallel codebook embedding tables of shape `(1026, 3072)` and 8 parallel codebook output heads of shape `(3072, 1026)`. Per LM step the **input embedding is the sum** of the 8 per-codebook embeddings at that position (MusicGen-style), and the **output is 8 parallel logit vectors** of shape `(1026,)` (one head per codebook), each sampled independently. The text LM head (`128256 → 3072`) is optional at inference and skipped to save ~1.5 GiB; it's only loaded for training (`use_text_head=True`).

#### 2.2 Audio Tokenizer (`HiggsAudioV2TokenizerModel`)

This is a **unified dual-branch codec**: a "semantic" HuBERT branch operating at 16 kHz captures phonetic/linguistic content, and an "acoustic" DAC-style RVQ branch at 24 kHz captures timbre/prosody/audio detail. The two branches are trained jointly so the resulting code stream encodes both. Cross-reference [AUDIO_CODECS.md](AUDIO_CODECS.md) for the underlying DAC encoder/decoder ops (Snake1d activation, dilated residual blocks, ConvTranspose1d upsampling) and the HuBERT-base CNN feature extractor.

**Top-level config (verbatim from `config.json`):**

```json
{
  "architectures": ["HiggsAudioV2TokenizerModel"],
  "model_type": "higgs_audio_v2_tokenizer",
  "sample_rate": 24000,
  "downsample_factor": 320,
  "codebook_dim": 64,
  "codebook_size": 1024,
  "kernel_size": 3,
  "block_dilations": [1, 1],
  "channel_ratios": [1, 1],
  "strides": [1, 1],
  "unit_kernel_size": 3,
  "initializer_range": 0.02,
  "dtype": "float32",
  "semantic_sample_rate": 16000,
  "target_bandwidths": [0.5, 1, 1.5, 2]
}
```

**Acoustic branch (`acoustic_model_config`, `model_type: "dac"`):**

```json
{
  "sampling_rate": 16000,           // input branch resampled to 16k for the DAC encoder
  "hop_length": 960,                // 16000 / 960 ≈ 16.67 Hz?  See note below
  "encoder_hidden_size": 64,
  "decoder_hidden_size": 1024,
  "hidden_size": 256,
  "downsampling_ratios": [8, 5, 4, 2, 3],   // product = 960
  "upsampling_ratios": [8, 5, 4, 2, 3],
  "n_codebooks": 9,
  "codebook_size": 1024,
  "codebook_dim": 8,
  "codebook_loss_weight": 1.0,
  "commitment_loss_weight": 0.25,
  "quantizer_dropout": 0
}
```

> Implementation note on rates. The DAC branch operates internally at 16 kHz with hop 960 (5 strided convs ×8,5,4,2,3 = 960). The published end-to-end frame rate is 25 Hz at 24 kHz, with **12 codebooks** declared in the Boson blog (semantic + 9 acoustic + 2 reserved), giving the marketed `25 × 12 × 10 bits = 3000 bps` (blog rounds to 2 kbps using effective bits/codebook). However the **LM consumes only 8 codebooks** (`num_codebooks=8` in the generation config) — the 9th acoustic codebook and the auxiliary semantic projections are present in the tokenizer but not modelled by the LM at inference. The acoustic config's `sampling_rate: 16000` is the *internal* DAC sample rate; the decoder upsamples (mirror `[8,5,4,2,3]`) back to a waveform that the wrapper resamples to 24 kHz for output. **Validate this assumption against the reference Python tokenizer's `decode()` output sample rate before shipping.**

**Semantic branch (`semantic_model_config`, `model_type: "hubert"`):** Standard HuBERT-base config — 12 transformer layers, hidden 768, 12 heads, intermediate 3072, GELU. Conv feature extractor: 7 layers, dims all 512, kernels `[10, 3, 3, 3, 3, 2, 2]`, strides `[5, 2, 2, 2, 2, 2, 2]` → cumulative stride 320 → 16000 / 320 = **50 Hz** semantic frame rate. The semantic branch produces continuous features that condition / are fused with the acoustic VQ — implementers should treat it as **encoder-only at inference** (it's only needed when *encoding* reference audio for voice cloning, not when decoding LM-predicted codes back to waveform).

**Inference path for SharpInference.** Two directions:
- **Encode** (voice cloning / reference audio in the conversation): waveform → resample 16 k → semantic HuBERT features + acoustic DAC encoder → quantize → per-frame codebook IDs of shape `(num_frames, 8)`. These IDs are then embedded by the LM's per-codebook embedding tables and spliced into the text stream at `<|AUDIO_OUT|>` positions.
- **Decode** (the only direction needed for pure-TTS without cloning): per-frame codebook IDs `(num_frames, 8)` → look up VQ entries (factorized: `(1024, 8)` per codebook, then project up to acoustic latent) → sum residuals → DAC decoder (snake + ConvTranspose1d × 5 with strides `[8,5,4,2,3]`) → waveform → resample to 24 kHz. The semantic branch is **not needed for decoding**.

#### 2.3 Speaker Conditioning

Higgs has **no separate speaker embedding network** (no x-vector, no ECAPA, no learned speaker codebook). Speaker identity is conveyed entirely through the prompt:

1. **Smart-voice mode (no reference).** Speaker is hinted via free-text scene descriptors: `"SPEAKER0: feminine"`, `"SPEAKER1: masculine, deep, British accent"`, or a longer description. The LM picks an internally consistent voice. Output voice is not reproducible across runs (essentially "any plausible voice matching the description").
2. **Zero-shot cloning mode.** A reference audio clip is added as an `{"role": "assistant", "content": [{"type": "audio", "url": ...}]}` turn *before* the user turn whose text should be cloned. The tokenizer encodes the reference into audio tokens; the LM treats them as in-context demonstration and continues with the same timbre.
3. **Multi-speaker cloning mode.** Per-speaker reference audio is embedded in the `scene` block, each immediately preceded by a `"SPEAKER0:"` / `"SPEAKER1:"` text tag, then the user text uses the matching `[SPEAKERn]` brackets.

There's no fixed limit on speaker count, but performance degrades quickly past 2–3 speakers (Boson reports 18.88 % WER on two-speaker conversations).

### 3. Multi-Speaker Dialogue Mode

This is what makes Higgs unique among 2025 TTS models — a single autoregressive pass produces an entire dialogue with multiple voices, proper turn-taking, and per-speaker prosody. The **prompt format** is plain text with bracketed tags inside the user message. Verbatim from the official example (`examples/transcript/multi_speaker/en_argument.txt`):

```
[SPEAKER0] I can't believe you did that without even asking me first!

[SPEAKER1] Oh, come on! It wasn't a big deal, and I knew you would overreact like this.

[SPEAKER0] Overreact? You made a decision that affects both of us without even considering my opinion!

[SPEAKER1] Because I didn't have time to sit around waiting for you to make up your mind! Someone had to act.
```

Wrapped in the chat template (verbatim from the transformers docs example):

```python
system_message = """You are an AI assistant designed to convert text into speech.
If the user's message includes a [SPEAKER*] tag, do not read out the tag and generate speech for the following text, using the specified voice.
If no speaker tag is present, select a suitable voice on your own."""

conversation = [
  {"role": "system", "content": [{"type": "text", "text": system_message}]},
  {"role": "scene", "content": [
      {"type": "text", "text": "Audio is recorded from a quiet room."},
      {"type": "text", "text": "SPEAKER0: feminine"},
      {"type": "text", "text": "SPEAKER1: masculine"}]},
  {"role": "user", "content": [{"type": "text", "text": user_message}]}
]
```

After Jinja rendering this becomes (illustrative):

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an AI assistant designed to convert text into speech...

<|scene_desc_start|>
Audio is recorded from a quiet room.

SPEAKER0: feminine
SPEAKER1: masculine
<|scene_desc_end|><|eot_id|><|start_header_id|>user<|end_header_id|>

[SPEAKER0] I can't believe you did that without even asking me first!
...
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

<|audio_out_bos|>
```

The model then autoregressively emits codebook tokens for the entire dialogue, with embedded silence/breath gaps between speakers, ending with `<|audio_eos|>` and `<|eot_id|>`. No explicit per-speaker boundary tokens appear in the audio stream — the LM has learned voice switching from training data.

**Multi-speaker cloning** uses per-speaker reference clips in the scene role (`SPEAKER0:` label then `{"type":"audio","url":...}`, then `SPEAKER1:` label then audio) instead of textual descriptors.

### 4. Inference Pipeline

End-to-end, the per-utterance flow is:

1. **Render chat template** (pure string templating in C#; the Jinja file is small). Output: a string with text + special tokens + `<|AUDIO_OUT|>` placeholders for any reference-audio frames.
2. **BPE-tokenize** the rendered string with the Llama-3.2 tokenizer → text token IDs.
3. **If any reference audio is present**: load each `.wav`/`.flac` URL, resample to 16 kHz (for HuBERT) and 24 kHz (for the DAC branch passthrough), run the tokenizer encoder → per-clip codebook ID tensor of shape `(T_frames, 8)`. Expand the corresponding `<|AUDIO_OUT|>` placeholders into `T_frames` positions and attach the codebook IDs.
4. **Forward through Llama+DualFFN**: for each LM step, embeddings = (text embedding **OR** sum of 8 codebook embeddings depending on position type), then 28 decoder layers (GQA + RoPE-llama3 + RMSNorm + per-token-type FFN routing). The KV cache is shared across text and audio positions.
5. **Autoregressive generation loop** until `<|audio_eos|>` or `<|eot_id|>` is sampled. Per step:
   - Compute 8 parallel logit vectors `(1026,)` from the 8 codebook heads.
   - Apply RAS logits processor + temperature/top-k/top-p **independently per codebook**.
   - Sample 8 IDs, splice them into the delay pattern (codebook k uses the sample from step `t-k`), feed back as next-step embeddings.
6. **Strip delay**: undelay the `(T_gen, 8)` matrix, drop the first `K-1=7` and trailing `K-1=7` invalid columns, drop stream BOS/EOS.
7. **Decode**: tokenizer decoder (DAC mirror upsampler) → 24 kHz waveform.
8. **Write WAV** at 24 kHz mono.

### 5. Sampling Parameters

Defaults from `generation_config.json` (verbatim):

```json
{
  "do_sample": true,
  "temperature": 1.0,
  "top_k": 50,
  "top_p": 0.95,
  "ras_win_len": 7,
  "ras_win_max_num_repeat": 2,
  "use_cache": true,
  "use_text_head": true,
  "bos_token_id": 1,
  "eos_token_id": 128009,
  "pad_token_id": 128001
}
```

The official example notebooks override to **`temperature=0.3`, `top_p=0.95`, `top_k=50`, `max_new_tokens=1024`** for quality, and the streaming serve engine uses **`temperature=0.7`**. No `repetition_penalty` is configured by default — Higgs uses its own **Repetition-Aware Sampling (RAS)** instead:

- **`ras_win_len=7`** — sliding window of the last 7 sampled audio frames examined per step.
- **`ras_win_max_num_repeat=2`** — if any audio token has already appeared more than 2 times within the window for that codebook, its logit is suppressed before sampling. Setting `ras_win_len ≤ 0` disables RAS.

RAS applies per codebook stream independently, after temperature scaling and before top-k/top-p truncation.

**Greedy vs. sampled.** The HF docs examples use `do_sample=False` (pure greedy/argmax per codebook); Boson's reference serve engine uses sampling with the defaults above. For TTS production, sampling tends to give more natural prosody but greedy is more reproducible.

### 6. Streaming

Streaming is first-class. The reference Python implementation is `AsyncHiggsAudioStreamer` (`boson_multimodal/serve/serve_engine.py`):

- Generation runs on a background thread feeding chunks into an `asyncio.Queue()`.
- The consumer iterates `async for delta in streamer`, each delta being a `HiggsAudioStreamerDelta { text?, text_tokens?, audio_tokens?, finish_reason? }`.
- Audio is yielded as **codebook-token chunks**, not as PCM — the consumer collects N frames worth of `(num_frames, 8)` IDs, runs the tokenizer decoder on them, and emits PCM samples.

The delay pattern naturally aligns with streaming: codebook k is delayed by k LM steps, so after `K-1=7` warmup steps every additional LM step produces one fully-resolved 25 Hz audio frame (= 40 ms of waveform). The decoder is a pure CNN with no recurrence so it can be run on small frame batches (e.g. 12 frames = 480 ms of audio per decode call) for low latency.

**C# mapping.** `IAsyncEnumerable<ReadOnlyMemory<float>>` returning 24 kHz mono PCM chunks. KV cache is the only state to thread through generations; codec decoder state is trivially stateless because DAC has no recurrence (left-pad each chunk with the previous 7 frames of context to avoid edge artifacts, then drop the prefix samples).

### 7. HuggingFace Files

**`bosonai/higgs-audio-v2-generation-3B-base`** (~23 GB):

| File | Size | Purpose | Needed by SharpInference? |
|---|---|---|---|
| `model.safetensors` OR `model-0000{1..3}-of-00003.safetensors` + index | 11.5 GB consolidated, or 4.97+4.98+1.59 GB sharded | BF16 weights for backbone, DualFFN, audio embedding tables, audio heads, (optional) text LM head | **Yes** — load one form |
| `config.json` | 1.1 kB | Architecture config above | **Yes** |
| `generation_config.json` | 351 B | Default sampling params | **Yes** (for defaults) |
| `processor_config.json` | 682 B | Audio token + delay token mappings | **Yes** |
| `chat_template.jinja` | 3.05 kB | Prompt rendering template | Use as **reference only** — reimplement in C# string builder |
| `tokenizer.json` | 17.2 MB | Llama-3.2 BPE merges + vocab + special tokens | **Yes** (reuse dotLLM's Llama-3 tokenizer) |
| `tokenizer_config.json`, `special_tokens_map.json` | <1 kB each | Token id maps | Yes (read at load) |
| `LICENSE` | 9.17 kB | Apache-2.0 | Bundle for redistribution |
| `*.png`, `*.mp4` | 1.4 MB total | Docs/demo | No |

**`bosonai/higgs-audio-v2-tokenizer`** (~12.3 GB):

| File | Size | Purpose | Needed? |
|---|---|---|---|
| `model.safetensors` | 11.5 GB | Combined acoustic+semantic weights | **Yes** (for encoding); for decode-only see note below |
| `model.pth` | 806 MB | PyTorch pickle (likely acoustic-only / float32 only) | Either-or with safetensors |
| `config.json` | 2.53 kB | Dual-branch config (DAC + HuBERT) | **Yes** |
| `preprocessor_config.json` | 206 B | Resample/normalize spec for the encoder | Encode-only |

> The ~600 M acoustic + ~95 M HuBERT-base ≈ 700 M params should give a `model.safetensors` of ~1.4 GB at BF16, not 11.5 GB. The 11.5 GB suggests the safetensors file includes redundant copies, optimizer/EMA shadows, or float32 weights — verify and possibly extract just the acoustic decoder for a decode-only SharpInference build to save ~10 GB.

### 8. Memory and Performance

**VRAM at BF16/FP16.**

- Backbone (3.6 B params) ≈ 7.2 GB
- DualFFN adapter (2.2 B params) ≈ 4.4 GB
- Audio embedding/head tables (8 codebooks × 1026 × 3072 × 2 bytes × 2 for embed+head) ≈ 0.1 GB
- LLM total at BF16 ≈ **~12 GB**, or ~6 GB at INT8, ~3.5 GB at INT4 (Q4_K_M).
- Tokenizer decoder (~600 M acoustic only): ~1.2 GB BF16, ~600 MB INT8.
- KV cache: 28 layers × 8 KV heads × 128 head_dim × 2 (K,V) × 2 bytes = **115 kB per token**; at max context 2048 ≈ 235 MB; with RoPE scaling to 32 k tokens, 3.7 GB (rarely needed for TTS).

**End-to-end estimate (FP16, single batch, RTX 4090):** ~14 GB peak — fits comfortably; on a 12 GB card you'll need INT8 weights or to offload the unused text LM head.

The brief's "~7 GB at FP16" figure undercounts because it ignores the 2.2 B DualFFN adapter. **Realistic FP16 footprint = ~14 GB total (LLM + tokenizer + KV cache).**

**Real-time factor (RTF).** Boson cites ~25 fps audio token rate; at 8 codebooks per step, each LM step produces 40 ms of audio. On an H100 the LM does ~150 tok/s autoregressively for 3 B+2.2 B models → 6 s of audio per wall-clock second → **RTF ~0.17** (≈6× real-time). RTX 4090 closer to 70 tok/s → ~2.8 s/s → **RTF ~0.35**. The DAC decoder is negligible (<5 ms per second of audio on any modern GPU). v2.5's 1 B condensation roughly doubles throughput.

### 9. C# Implementation Notes

**Backbone reuse from dotLLM.** The text-only forward pass is **stock Llama-3.2-3B**:
- RMSNorm, SwiGLU MLP, GQA (24 Q / 8 KV), RoPE with llama3-type scaling (`factor=32, low=0.125, high=0.5, original_max=1024, theta=500000`).
- Tokenizer is the standard Llama-3.2 BPE — dotLLM's tokenizer loader works unchanged.
- All RoPE/attention/RMSNorm/MLP kernels from dotLLM port directly.

**New code required (SharpInference.Audio.HiggsAudio):**

1. **DualFFN routing**. Two SwiGLU MLPs per layer; per-token mask drives a gather/scatter (text positions → MLP_text, audio positions → MLP_audio). Two reasonable implementations:
   - **Mask-and-add**: run both MLPs on the full sequence, multiply outputs by `(1-audio_mask)` and `audio_mask` respectively, add. Simple, wastes ~2× MLP FLOPs but trivially batchable.
   - **Gather/scatter**: split positions into two contiguous buffers, run each MLP on its buffer, scatter back. Faster but adds two non-trivial kernels.
   - For initial implementation use mask-and-add; profile and switch only if MLP becomes the bottleneck.

2. **Audio embedding fusion**. At positions where the token is `<|AUDIO_OUT|>`, replace the standard text embedding lookup with `sum_k embed_k(codebook_id[t, k])` — 8 parallel `Gather` ops then a sum.

3. **Audio LM heads**. 8 parallel `Linear(3072, 1026)` heads, executed only at the last position during sampling (or all positions during training, which we don't need).

4. **Delay-pattern handling**. Reuse the MusicGen pattern logic from [MUSICGEN_ARCHITECTURE.md](MUSICGEN_ARCHITECTURE.md) — same `delays=[0,1,2,...,7]` shape, just with K=8 instead of K=4. Pre-apply delay before feeding to the LM; post-undelay before decoding.

5. **Audio tokenizer decoder** (decode-only is sufficient for non-cloning TTS). Implement:
   - 8 per-codebook lookups → factorized projection to acoustic latent (`codebook_dim=8` projected up to `decoder_hidden_size=1024` via per-codebook `Linear(8, 1024)`).
   - Sum the 8 latents.
   - DAC decoder: initial `Conv1d`, then 5 `DecoderBlock`s with `ConvTranspose1d` strides `[8, 5, 4, 2, 3]` (cumulative ×960) and dilated `ResidualUnit`s with Snake1d, final `Conv1d → 1ch`. Add `Snake1d` to the SharpInference IBackend (currently only documented for SNAC/DAC in [AUDIO_CODECS.md](AUDIO_CODECS.md), not yet implemented).
   - Resample 16 kHz → 24 kHz (or skip if decoder output is already 24 k — verify against reference).

6. **Audio tokenizer encoder** (only needed for voice cloning):
   - HuBERT-base CNN feature extractor + 12-layer transformer (port from dotLLM-style stack, no causal mask, GELU activation, weight-norm convs).
   - DAC encoder mirror of the decoder above.
   - Joint quantization to per-frame 8-tuple of codebook IDs.

7. **Chat template renderer**. The Jinja file is small (~3 kB, no loops over data structures) — port to a straightforward C# `StringBuilder` method `RenderHiggsPrompt(systemMsg, scene, dialogue, addGenerationPrompt)` that handles the three roles (system / scene / user / assistant) and the audio embedding placeholders. Trivial.

8. **Streaming API**. `IAsyncEnumerable<HiggsAudioDelta>` where `HiggsAudioDelta` mirrors the Python `HiggsAudioStreamerDelta { ushort[]? TextTokens, ushort[,]? AudioTokens, FinishReason? }`. A wrapper consumer can convert audio-token deltas to PCM by buffering K=8 frames of context, running the decoder, and emitting `ReadOnlyMemory<float>` chunks. KV cache is the only required state.

9. **RAS logits processor**. Per-codebook circular buffer of the last 7 sampled IDs; before sampling step t, count occurrences in the buffer and set `logits[id] = -inf` for any id with count > 2. ~30 lines of C#.

10. **Sampling**. Reuse dotLLM's top-k + top-p + temperature kernels; just run them 8 times in parallel (once per codebook).

**Validation plan.** Match outputs against the reference Python pipeline:
- Tokenize the same prompt — compare BPE IDs (must be exact).
- Run one forward pass — compare logits at the last text-position for both text head and the 8 audio heads (within BF16 tolerance, ~1e-2 relative).
- Decode a fixed audio-token tensor through the tokenizer — compare waveforms (within ~−40 dB error, since both DAC decoders are deterministic).
- End-to-end with greedy sampling and a fixed prompt — waveforms should match sample-for-sample up to tokenizer fp tolerance.

**Out of scope for v1.** v2.5 (different/smaller architecture, untested for open-source LM weights at time of writing), v3 STT (uses Whisper+Qwen3, completely different stack — see [WHISPER_ARCHITECTURE.md](WHISPER_ARCHITECTURE.md) and would belong in a Qwen3 dotLLM package), training/finetuning (we are inference-only), and the v1 understanding variant (not open-sourced for generation).

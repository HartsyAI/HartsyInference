# Higgs Audio v2 — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: HartsyInference.Audio (Higgs Audio pipeline)

> **Stub.** The narrative walkthrough, restated pseudocode and resolved open questions were
> removed on 2026-08-06 — this model is built and verified, so the C# is the source of truth for
> *how it works*. What remains is what the code cannot tell you: upstream provenance, reference
> constants to diff a suspect port against, where implementations disagree, and bring-up traps.
> Full history is in git. Parity evidence: `docs/Checklists/PARITY_VERIFICATION.md`.

## Summary

Higgs Audio v2 (Boson AI, released July 2025 under Apache 2.0) is an audio foundation model that turns a Llama-3.2-3B text LLM into a multimodal speech generator. The model is **autoregressive**: it predicts a sequence of discrete audio codec tokens (8 parallel codebooks per audio frame) interleaved with text in a ChatML-style conversation, then a separate **HiggsAudioV2Tokenizer** (a custom dual semantic + acoustic codec) decodes those tokens into a 24 kHz waveform. Three architectural pieces define v2: (1) a stock Llama-3.2-3B backbone (28 layers, 3072 hidden, GQA 24/8 heads, RoPE-llama3) extended with audio-stream BOS/EOS, audio placeholder, and an audio-delay token in the text vocab; (2) a **DualFFN audio adapter** — for audio positions only, a second parallel FFN block (2.2 B extra params) runs alongside the standard Llama FFN, giving the LM a dedicated acoustic expert at minimal compute overhead; (3) the **unified tokenizer**, a HuBERT-base "semantic" branch (16 kHz, 50 Hz) fused with a DAC-style RVQ "acoustic" branch (24 kHz, 25 Hz, 12 codebooks × 1024) — 2 kbps total, but the LM only consumes/produces 8 of those codebooks (`num_codebooks=8`, `codebook_size=1024` per the actual `config.json`).

The pipeline uniquely supports four modes from one checkpoint via the chat template alone — **single-speaker smart voice**, **multi-speaker dialogue** (`[SPEAKER0]`/`[SPEAKER1]` tags), **zero-shot voice cloning** (reference audio in assistant role), and **multi-speaker voice cloning** (per-speaker reference audio in scene role). Generation uses standard Llama sampling plus a custom **RAS (Repetition-Aware Sampling)** logits processor (`ras_win_len=7`, `ras_win_max_num_repeat=2`) to suppress repetitive audio loops. Codebooks are arranged in a MusicGen-style **delay pattern** so all 8 streams can be predicted in parallel per LM step.

Higgs v2.5 (Sep 2025) is a 1B-parameter condensation with the same tokenizer and chat template but stronger primary-language coverage (en/zh/ko/ja via GRPO) and explicit expressiveness control tags.

For HartsyInference this maps cleanly to: **the native `HartsyInference.LLM` transformer for the Llama-3.2-3B backbone + a new DualFFN MLP variant; a new audio codec implementation in HartsyInference.Audio that combines DAC-style decoder ops (already documented in [AUDIO_CODECS.md](AUDIO_CODECS.md)) with a HuBERT-style semantic encoder; pure string templating for prompt construction; KV-cache + `IAsyncEnumerable<float[]>` for streaming.**

Sources: [boson-ai/higgs-audio (GitHub)](https://github.com/boson-ai/higgs-audio), [bosonai/higgs-audio-v2-generation-3B-base (HF)](https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base), [bosonai/higgs-audio-v2-tokenizer (HF)](https://huggingface.co/bosonai/higgs-audio-v2-tokenizer), [HiggsAudioV2 transformers docs](https://huggingface.co/docs/transformers/model_doc/higgs_audio_v2), [Boson AI v2 blog](https://www.boson.ai/blog/higgs-audio-v2), [Boson AI v2.5 blog](https://www.boson.ai/blog/higgs-audio-v2.5), [erogol model-check writeup](https://erogol.substack.com/p/model-check-higgs-audio-v2-unified).

## Variants

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

## Sampling Parameters

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

## HuggingFace Files

**`bosonai/higgs-audio-v2-generation-3B-base`** (~23 GB):

| File | Size | Purpose | Needed by HartsyInference? |
|---|---|---|---|
| `model.safetensors` OR `model-0000{1..3}-of-00003.safetensors` + index | 11.5 GB consolidated, or 4.97+4.98+1.59 GB sharded | BF16 weights for backbone, DualFFN, audio embedding tables, audio heads, (optional) text LM head | **Yes** — load one form |
| `config.json` | 1.1 kB | Architecture config above | **Yes** |
| `generation_config.json` | 351 B | Default sampling params | **Yes** (for defaults) |
| `processor_config.json` | 682 B | Audio token + delay token mappings | **Yes** |
| `chat_template.jinja` | 3.05 kB | Prompt rendering template | Use as **reference only** — reimplement in C# string builder |
| `tokenizer.json` | 17.2 MB | Llama-3.2 BPE merges + vocab + special tokens | **Yes** (reuse the native LLM Llama-3 tokenizer) |
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

> The ~600 M acoustic + ~95 M HuBERT-base ≈ 700 M params should give a `model.safetensors` of ~1.4 GB at BF16, not 11.5 GB. The 11.5 GB suggests the safetensors file includes redundant copies, optimizer/EMA shadows, or float32 weights — verify and possibly extract just the acoustic decoder for a decode-only HartsyInference build to save ~10 GB.

## Memory and Performance

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

## C# Implementation Notes

**Backbone reuse from `HartsyInference.LLM`.** The text-only forward pass is **stock Llama-3.2-3B**:
- RMSNorm, SwiGLU MLP, GQA (24 Q / 8 KV), RoPE with llama3-type scaling (`factor=32, low=0.125, high=0.5, original_max=1024, theta=500000`).
- Tokenizer is the standard Llama-3.2 BPE — the engine's native LLM tokenizer loader handles it unchanged.
- All RoPE/attention/RMSNorm/MLP kernels come from the native LLM transformer directly.

**New code required (HartsyInference.Audio.HiggsAudio):**

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
   - DAC decoder: initial `Conv1d`, then 5 `DecoderBlock`s with `ConvTranspose1d` strides `[8, 5, 4, 2, 3]` (cumulative ×960) and dilated `ResidualUnit`s with Snake1d, final `Conv1d → 1ch`. Add `Snake1d` to the HartsyInference IBackend (currently only documented for SNAC/DAC in [AUDIO_CODECS.md](AUDIO_CODECS.md), not yet implemented).
   - Resample 16 kHz → 24 kHz (or skip if decoder output is already 24 k — verify against reference).

6. **Audio tokenizer encoder** (only needed for voice cloning):
   - HuBERT-base CNN feature extractor + 12-layer transformer (a standard encoder stack, no causal mask, GELU activation, weight-norm convs).
   - DAC encoder mirror of the decoder above.
   - Joint quantization to per-frame 8-tuple of codebook IDs.

7. **Chat template renderer**. The Jinja file is small (~3 kB, no loops over data structures) — port to a straightforward C# `StringBuilder` method `RenderHiggsPrompt(systemMsg, scene, dialogue, addGenerationPrompt)` that handles the three roles (system / scene / user / assistant) and the audio embedding placeholders. Trivial.

8. **Streaming API**. `IAsyncEnumerable<HiggsAudioDelta>` where `HiggsAudioDelta` mirrors the Python `HiggsAudioStreamerDelta { ushort[]? TextTokens, ushort[,]? AudioTokens, FinishReason? }`. A wrapper consumer can convert audio-token deltas to PCM by buffering K=8 frames of context, running the decoder, and emitting `ReadOnlyMemory<float>` chunks. KV cache is the only required state.

9. **RAS logits processor**. Per-codebook circular buffer of the last 7 sampled IDs; before sampling step t, count occurrences in the buffer and set `logits[id] = -inf` for any id with count > 2. ~30 lines of C#.

10. **Sampling**. Reuse the native `HartsyInference.LLM` top-k + top-p + temperature sampler kernels; just run them 8 times in parallel (once per codebook).

**Validation plan.** Match outputs against the reference Python pipeline:
- Tokenize the same prompt — compare BPE IDs (must be exact).
- Run one forward pass — compare logits at the last text-position for both text head and the 8 audio heads (within BF16 tolerance, ~1e-2 relative).
- Decode a fixed audio-token tensor through the tokenizer — compare waveforms (within ~−40 dB error, since both DAC decoders are deterministic).
- End-to-end with greedy sampling and a fixed prompt — waveforms should match sample-for-sample up to tokenizer fp tolerance.

**Out of scope for v1.** v2.5 (different/smaller architecture, untested for open-source LM weights at time of writing), v3 STT (uses Whisper+Qwen3, completely different stack — see [WHISPER_ARCHITECTURE.md](WHISPER_ARCHITECTURE.md) and would belong with the Qwen3 path in `HartsyInference.LLM`), training/finetuning (we are inference-only), and the v1 understanding variant (not open-sourced for generation).

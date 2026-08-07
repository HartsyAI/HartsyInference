# Whisper — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: HartsyInference.Audio (Whisper encoder, decoder, pipeline)

> **Stub.** The narrative walkthrough, restated pseudocode and resolved open questions were
> removed on 2026-08-06 — this model is built and verified, so the C# is the source of truth for
> *how it works*. What remains is what the code cannot tell you: upstream provenance, reference
> constants to diff a suspect port against, where implementations disagree, and bring-up traps.
> Full history is in git. Parity evidence: `docs/Checklists/PARITY_VERIFICATION.md`.

## Summary

Whisper is OpenAI's encoder-decoder transformer for speech-to-text. The architecture consists of a Conv1D feature extractor (2 layers), a transformer encoder with sinusoidal positional encoding, and an autoregressive transformer decoder with learned positional embeddings and cross-attention to encoder outputs. Models range from 39M (tiny) to 1.55B (large-v3) parameters. The `large-v3-turbo` (Sept 2024) keeps the full 32-layer encoder but distills the decoder down to 4 layers (~8x decoder speedup, ~2x faster end-to-end), at slight quality cost on the long tail.

Audio preprocessing (mel spectrogram, STFT, log compression) is documented in the companion [MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md) — Whisper's preprocessor is the reference implementation we validate against. This file covers the model architecture only.

Sources: [OpenAI Whisper repo](https://github.com/openai/whisper), [Whisper paper](https://arxiv.org/abs/2212.04356), [whisper.cpp](https://github.com/ggml-org/whisper.cpp), [HuggingFace Whisper configs](https://huggingface.co/openai/whisper-large-v3), [Whisper-Large-v3-Turbo card](https://huggingface.co/openai/whisper-large-v3-turbo), [Distil-Whisper paper](https://arxiv.org/abs/2311.00430)

## Key Numbers / Constants

| Constant | Value |
|----------|-------|
| Sample rate (input) | 16,000 Hz |
| Audio chunk length | 30 seconds |
| n_audio_ctx (encoder positions) | 1500 |
| n_text_ctx (decoder positions) | 448 |
| Head dim (all sizes) | 64 |
| Mel bins (standard) | 80 |
| Mel bins (large-v3+) | 128 |
| FFN ratio | 4x d_model |
| Conv layer 1 kernel / stride | 3 / 1 |
| Conv layer 2 kernel / stride | 3 / 2 |
| Sinusoidal max_timescale | 10000 |
| Q/K scale factor | `head_dim^(-0.25)` per side |
| Timestamp token range | 50364 - 51864 (1501 tokens) |
| Timestamp precision | 0.02 s |
| Compression-ratio fail threshold | 2.4 |
| Avg-logprob fail threshold | -1.0 |
| Length penalty alpha | 1.0 |
| Temperature fallback ladder | [0, 0.2, 0.4, 0.6, 0.8, 1.0] |

## Data Layouts / Formats

### Encoder I/O
```
Input:  [batch, n_mels, 3000]            (mel spectrogram from MEL_SPECTROGRAM.md)
Conv1:  [batch, d_model, 3000]
Conv2:  [batch, d_model, 1500]           (stride=2 halves time)
+ posE: [batch, 1500, d_model]           (transpose to time-last)
After N blocks + ln_post:
Output: [batch, 1500, d_model]            (cached as cross-attention K/V source)
```

### Decoder I/O (per step)
```
Token IDs:           [batch, current_pos]      (int64)
+ embedding:         [batch, current_pos, d_model]
+ pos embedding:     [batch, current_pos, d_model]   (learned, sliced [0:current_pos])
Through N blocks with KV cache:
After final ln:      [batch, current_pos, d_model]
Logits (matmul lm_head): [batch, current_pos, n_vocab]
```

### KV Cache Layout (per decoder layer)
```
Self-attn K:   [1, n_text_ctx=448, d_model]   pre-allocated, filled position-by-position
Self-attn V:   [1, n_text_ctx=448, d_model]   ditto
Cross-attn K:  [1, n_audio_ctx=1500, d_model] computed once from encoder output
Cross-attn V:  [1, n_audio_ctx=1500, d_model] ditto
```

## Reference Implementations

- [OpenAI Whisper](https://github.com/openai/whisper) — Canonical Python implementation. `whisper/model.py` for the model, `whisper/decoding.py` for sampling, `whisper/timing.py` for word-level timestamps.
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) — C++ implementation with GGML. Validation target (1e-4 tolerance on mel; bit-exact text decode on greedy).
- [HuggingFace transformers](https://github.com/huggingface/transformers/tree/main/src/transformers/models/whisper) — Model configs, feature extractor, runtime filterbank generation, safetensors keys.
- [distil-whisper](https://github.com/huggingface/distil-whisper) — Distilled variants. `WhisperForConditionalGeneration` config differs only in `decoder_layers`.
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2-based, useful for performance reference (RTF, batched decode).
- [Whisper-Large-v3-Turbo card](https://huggingface.co/openai/whisper-large-v3-turbo) — official turbo release notes.

## Differences Between Implementations

| Aspect | OpenAI Python | whisper.cpp | HF transformers |
|--------|---------------|-------------|-----------------|
| Long-form | Sequential (timestamp-driven) | Sequential | Chunked (5s overlap) |
| Beam search | Yes (beam_size=5 default) | Yes | Yes |
| Temperature fallback | 6-stage ladder | 6-stage ladder | 6-stage ladder |
| Word timestamps | Cross-attn DTW | Cross-attn DTW | Cross-attn DTW |
| Tokenizer | tiktoken | Custom BPE | HF tokenizers (Rust) |
| Tensor names | `encoder.blocks.i...` | OpenAI naming | `model.encoder.layers.i...` |
| Format | `.pt` (pickle) | `.bin` (GGML) | safetensors |
| Quantization | None | Q4_0/Q5_0/Q8_0 | bitsandbytes int8 |

For HartsyInference our reference is the **HF safetensors layout** (most stable, no pickle hazard, our SafeTensors loader is ready). Validation goes against whisper.cpp's GGML model (greedy, identical mel) for bit-level reproducibility.

## Implementation Notes for HartsyInference

1. **Attention scaling**: Use `head_dim^(-0.25)` on both Q and K, NOT the standard `head_dim^(-0.5)` on QK product. Easy to miss; check against whisper.cpp's `whisper_full_default_params`.

2. **Key has no bias**: The key projection in both self-attention and cross-attention omits the bias term. SafeTensors loader must not error on missing `k.bias`.

3. **Decoder positional embeddings are learned** (unlike encoder which uses sinusoidal). Slice `embed_positions[:current_pos]` per step.

4. **Weight tying**: Output logits share weights with token embedding. `proj_out` may or may not be present in safetensors — fall back to `embed_tokens.weight.T` when absent.

5. **Support 80 and 128 mel bins**: large-v3+ uses 128, all others use 80. Detect at config-load time, not hardcoded.

6. **GGML vs safetensors**: Load HuggingFace safetensors as the default path. GGML support is optional (whisper.cpp interop).

7. **KV-cache layout**: Pre-allocate self-attention K/V to `[1, n_text_ctx=448, d_model]` at pipeline construction. Cross-attention K/V are computed once after the encoder and reused for all decode steps.

8. **Suppress-tokens list**: ship the hardcoded list from `whisper/decoding.py` (~99 tokens including punctuation in some heuristics, all language tags except the active one, etc.). Apply as `-inf` mask on logits before sampling.

9. **Temperature fallback**: implement the 6-stage retry on quality failure. Compression-ratio is computed via `len(tokens) / len(zlib.compress(text.encode()))` — need a zlib implementation (built into .NET via `System.IO.Compression.DeflateStream`).

10. **Timestamp token rules**: timestamps must appear in start/end pairs and monotonically non-decreasing — enforce as logit suppression at sampling time (the OpenAI code masks invalid next-timestamp tokens to `-inf` based on the most recent timestamp seen).

11. **Distil-Whisper compatibility**: same model code, different `decoder_layers`. The safetensors loader should just read `decoder_layers` from the config and instantiate the right number of decoder blocks — no separate model class.

12. **Streaming**: see [STREAMING_AUDIO_INFERENCE.md](STREAMING_AUDIO_INFERENCE.md) for the chunked-and-overlap pattern used by streaming Whisper wrappers (and by Moonshine, Parakeet, etc.).

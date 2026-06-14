# Whisper — Architecture Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: HartsyInference.Audio (Whisper encoder, decoder, pipeline)

## Summary

Whisper is OpenAI's encoder-decoder transformer for speech-to-text. The architecture consists of a Conv1D feature extractor (2 layers), a transformer encoder with sinusoidal positional encoding, and an autoregressive transformer decoder with learned positional embeddings and cross-attention to encoder outputs. Models range from 39M (tiny) to 1.55B (large-v3) parameters. The `large-v3-turbo` (Sept 2024) keeps the full 32-layer encoder but distills the decoder down to 4 layers (~8x decoder speedup, ~2x faster end-to-end), at slight quality cost on the long tail.

Audio preprocessing (mel spectrogram, STFT, log compression) is documented in the companion [MEL_SPECTROGRAM.md](MEL_SPECTROGRAM.md) — Whisper's preprocessor is the reference implementation we validate against. This file covers the model architecture only.

Sources: [OpenAI Whisper repo](https://github.com/openai/whisper), [Whisper paper](https://arxiv.org/abs/2212.04356), [whisper.cpp](https://github.com/ggml-org/whisper.cpp), [HuggingFace Whisper configs](https://huggingface.co/openai/whisper-large-v3), [Whisper-Large-v3-Turbo card](https://huggingface.co/openai/whisper-large-v3-turbo), [Distil-Whisper paper](https://arxiv.org/abs/2311.00430)

## Detailed Findings

### Model Size Configurations

All data confirmed from OpenAI `model.py` and HuggingFace `config.json` files.

| Model | Layers (enc/dec) | Width (d_model) | Heads | FFN dim | n_mels | Vocab | Parameters |
|-------|-----------------|-----------------|-------|---------|--------|-------|------------|
| tiny | 4 / 4 | 384 | 6 | 1536 | 80 | 51865 | 39M |
| base | 6 / 6 | 512 | 8 | 2048 | 80 | 51865 | 74M |
| small | 12 / 12 | 768 | 12 | 3072 | 80 | 51865 | 244M |
| medium | 24 / 24 | 1024 | 16 | 4096 | 80 | 51865 | 769M |
| large-v2 | 32 / 32 | 1280 | 20 | 5120 | 80 | 51865 | 1550M |
| large-v3 | 32 / 32 | 1280 | 20 | 5120 | 128 | 51866 | 1550M |
| large-v3-turbo | 32 / 4 | 1280 | 20 | 5120 | 128 | 51866 | 809M |

Key: FFN dim = 4 * d_model. Head dim = 64 for all sizes. n_audio_ctx = 1500. n_text_ctx = 448. large-v3-turbo has only 4 decoder layers (distilled). large-v3+ uses 128 mel bins. large-v3 adds Cantonese to the language set (+1 vocab entry).

### Distil-Whisper Variants

Distil-Whisper (HuggingFace, 2023) is an independent distillation effort that uses pseudo-labeling on ~22k hours of public audio and trains a smaller student model with the WER-filter trick. The official distillations:

| Model | Enc Layers | Dec Layers | Width | Heads | Params | English Only |
|-------|-----------|-----------|-------|-------|--------|--------------|
| distil-large-v2 | 32 | 2 | 1280 | 20 | 756M | Yes |
| distil-large-v3 | 32 | 2 | 1280 | 20 | 756M | Yes |
| distil-medium.en | 24 | 2 | 1024 | 16 | 394M | Yes |
| distil-small.en | 12 | 2 | 768 | 12 | 166M | Yes |

Distil-Whisper drops the decoder to **2 layers** (turbo uses 4). Architecture is otherwise identical to the parent Whisper — same encoder, same tokenizer, same special tokens. Loading distil-whisper safetensors into our WhisperModel needs only a different `decoder_layers` value in the config.

**Distil-large-v3** uses sequential long-form decoding (the OpenAI Whisper paper's algorithm) rather than HuggingFace's chunked variant; expect different WER between the two long-form modes.

### Conv1D Feature Extractor

From [model.py AudioEncoder.__init__](https://github.com/openai/whisper/blob/main/whisper/model.py):

```
Conv1D Layer 1: in=n_mels(80/128), out=d_model, kernel=3, padding=1, stride=1, act=GELU
Conv1D Layer 2: in=d_model, out=d_model, kernel=3, padding=1, stride=2, act=GELU
```

Stride=2 on layer 2 halves time dimension: 3000 STFT frames -> 1500 encoder positions.

### Encoder Architecture

```
1. Conv1D layer 1 + GELU
2. Conv1D layer 2 (stride=2) + GELU
3. Add sinusoidal positional embedding [1500, d_model]
4. N x ResidualAttentionBlock (self-attention only)
5. LayerNorm (ln_post)
```

Sinusoidal encoding: standard transformer formula with max_timescale=10000. NOT learnable — registered as buffer.

ResidualAttentionBlock (pre-norm):
```
x = x + self_attention(attn_ln(x))
x = x + mlp(mlp_ln(x))
```

MLP: Linear(d_model, 4*d_model) -> GELU -> Linear(4*d_model, d_model)

### Attention Details

```
query = Linear(d_model, d_model, bias=True)
key   = Linear(d_model, d_model, bias=False)  // NO BIAS on key
value = Linear(d_model, d_model, bias=True)
out   = Linear(d_model, d_model, bias=True)
```

Unusual scaling — applied to both q and k separately:
```
scale = head_dim^(-0.25)    // NOT head_dim^(-0.5)
qk = (q * scale) @ (k * scale).T   // equivalent to qk / sqrt(head_dim)
```

### Decoder Architecture

```
1. Token embedding: Embedding(n_vocab, d_model)
2. Learned positional embedding: Parameter[448, d_model]  (NOT sinusoidal)
3. N x ResidualAttentionBlock (with cross-attention):
   x = x + causal_self_attention(attn_ln(x))
   x = x + cross_attention(cross_attn_ln(x), encoder_output)
   x = x + mlp(mlp_ln(x))
4. LayerNorm (ln)
5. Logits: x @ token_embedding.weight.T  (weight tying)
```

Causal mask: lower-triangular boolean mask [n_text_ctx, n_text_ctx].

### Special Tokens

From [tokenizer.py](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py):

| Token | ID |
|-------|-----|
| <\|endoftext\|> | 50257 |
| <\|startoftranscript\|> | 50258 |
| <\|en\|> through <\|yo\|> | 50259-50357 (99 languages) |
| <\|translate\|> | 50358 |
| <\|transcribe\|> | 50359 |
| <\|nospeech\|> | 50362 |
| <\|notimestamps\|> | 50363 |
| <\|0.00\|> through <\|30.00\|> | 50364-51864 (1501 tokens, 0.02s increments) |

Decode prompt: `<|startoftranscript|> <|lang|> <|task|> [<|notimestamps|>]`

### Timestamp Decoding

From [decoding.py](https://github.com/openai/whisper/blob/main/whisper/decoding.py):

```
seconds = (token_id - 50364) * 0.02
```

Rules: timestamps in start/end pairs, monotonically non-decreasing, must appear at start of decoding.

### Word-Level Timestamps (Cross-Attention Alignment)

From `whisper/timing.py`:

1. Run the decoder forward and capture cross-attention weights from a hardcoded subset of decoder heads (the "alignment_heads" — discovered offline per model size).
2. Average those head weights across selected heads.
3. Apply median filtering (length 7) along the time axis.
4. Run dynamic time warping (DTW) between token positions and audio frames using the smoothed cross-attention matrix as the cost matrix.
5. Map each token to its DTW-aligned audio frame; convert frames to seconds via `frame_index * 0.02`.

The alignment heads list ships with each model (`whisper.tokenizer.LANGUAGES`, `_ALIGNMENT_HEADS` constants encoded as a numpy-compressed bytestring). large-v3 / turbo use different head subsets than large-v2.

### Decoding Strategies

From `whisper/decoding.py`:

- **Greedy** (`temperature=0`): argmax at each step.
- **Beam search** (`beam_size=5`, temperature=0): standard length-normalized beam.
- **Temperature fallback**: if greedy result fails quality checks (compression ratio > 2.4, avg logprob < -1.0, or detected as no-speech), retry with temperatures [0.2, 0.4, 0.6, 0.8, 1.0] in sequence.
- **Suppress tokens**: a list of special tokens never sampled (~99 tokens including most language tags, all timestamp tokens before they should appear, etc.) — masked via `-inf` logits.
- **Length penalty** (alpha=1.0 default): `score = logprob / ((5 + n_tokens) / 6) ** alpha`.

### KV-Cache for Decoder

Decoder cross-attention: K, V from encoder output are constant across decode steps — compute once after encoder forward, reuse for all decoder positions. Per-layer cache shape: `[1, n_audio_ctx=1500, n_text_state]`.

Decoder self-attention: K, V grow as tokens are appended. Per-layer cache shape: `[1, current_text_pos, n_text_state]`, grown by 1 per step. Standard append-and-grow pattern.

For our backend: pre-allocate self-attention KV cache to `[1, n_text_ctx=448, n_text_state]` per layer at pipeline construction; write into rows as positions are added; slice valid prefix for each attention compute.

### GGML Whisper Format

From [whisper.cpp convert-pt-to-ggml.py](https://github.com/ggml-org/whisper.cpp/blob/master/models/convert-pt-to-ggml.py):

Binary structure:
```
1. Magic: 0x67676d6c ("ggml")
2. Hyperparams: 11 x int32 (n_vocab, n_audio_ctx, n_audio_state, n_audio_head,
   n_audio_layer, n_text_ctx, n_text_state, n_text_head, n_text_layer, n_mels, ftype)
3. Mel filters: shape[0], shape[1], float32 data
4. Tokenizer: n_tokens, then len+bytes per token
5. Tensors: n_dims, name_len, dtype, dims[], name, data
```

Tensor names use OpenAI format: `encoder.blocks.{i}.attn.query.weight`, `decoder.blocks.{i}.cross_attn.key.weight`, etc.

This is GGML (legacy) **not GGUF**. whisper.cpp ships its own GGML files; HuggingFace ships safetensors. Two distinct format paths.

### HuggingFace Safetensors Format

The HuggingFace transformers package ships Whisper as standard safetensors at:
- `openai/whisper-{tiny,base,small,medium,large-v2,large-v3,large-v3-turbo}`
- `distil-whisper/distil-{small.en,medium.en,large-v2,large-v3}`

Tensor key conventions (HF):
- `model.encoder.conv1.{weight,bias}`
- `model.encoder.conv2.{weight,bias}`
- `model.encoder.embed_positions.weight` (sinusoidal, materialized at conversion time)
- `model.encoder.layers.{i}.self_attn.{q,k,v,out}_proj.{weight,bias}` (k has no bias)
- `model.encoder.layers.{i}.self_attn_layer_norm.{weight,bias}`
- `model.encoder.layers.{i}.fc{1,2}.{weight,bias}`
- `model.encoder.layers.{i}.final_layer_norm.{weight,bias}`
- `model.encoder.layer_norm.{weight,bias}`
- `model.decoder.embed_tokens.weight` (tied to lm_head)
- `model.decoder.embed_positions.weight` (learned)
- `model.decoder.layers.{i}.{self_attn,encoder_attn,self_attn_layer_norm,encoder_attn_layer_norm,fc1,fc2,final_layer_norm}`
- `model.decoder.layer_norm.{weight,bias}`
- `proj_out.weight` (often missing — falls back to embed_tokens.weight transpose)

OpenAI's PyTorch `.pt` files use the original naming (`encoder.blocks.i.attn.query.weight`). The HF naming is what we should target for our SafeTensors loader.

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

## Algorithm Steps

### End-to-End Inference (single 30s chunk, greedy)

```
1. AUDIO PREPROCESSING (see MEL_SPECTROGRAM.md)
   audio[float32, 480000] -> mel[float32, 1, n_mels, 3000]

2. ENCODER FORWARD (once)
   x = mel
   x = GELU(Conv1D(x))                       # [1, d_model, 3000]
   x = GELU(Conv1D(x, stride=2))             # [1, d_model, 1500]
   x = transpose to time-last                 # [1, 1500, d_model]
   x = x + sinusoidal_pos_embed
   for block in encoder_blocks:
     x = block(x)                             # pre-norm self-attn + MLP
   x = LayerNorm(x)
   audio_features = x                         # [1, 1500, d_model]

3. CROSS-ATTENTION KV PRECOMPUTE
   for layer in decoder_layers:
     cross_k[layer] = layer.cross.k_proj(audio_features)
     cross_v[layer] = layer.cross.v_proj(audio_features)

4. DECODE PROMPT INIT
   prompt_tokens = [SOT, lang_token, TRANSCRIBE, NOTIMESTAMPS?]
   tokens = prompt_tokens
   pos = 0
   while not EOT and pos < n_text_ctx:
     a. embed = embed_tokens(tokens[-1]) + embed_positions[pos]
        (or initial: embed all prompt tokens)
     b. for layer in decoder_layers:
          # self-attention with KV append-and-grow
          q,k,v = self_attn_proj(layer_norm(x))
          self_k[layer][pos] = k
          self_v[layer][pos] = v
          x = x + attention(q, self_k[layer][:pos+1], self_v[layer][:pos+1], causal)
          # cross-attention with precomputed KV
          q = cross_attn.q_proj(cross_ln(x))
          x = x + attention(q, cross_k[layer], cross_v[layer], no_mask)
          # FFN
          x = x + mlp(mlp_ln(x))
     c. x = LayerNorm(x)
     d. logits = x @ embed_tokens.T
     e. suppress non-speech and forbidden tokens (set logits = -inf)
     f. next_token = argmax(logits)
     g. tokens.append(next_token); pos++

5. POST-PROCESSING
   - Strip prompt tokens, EOT
   - Decode token IDs via BPE -> text
   - If timestamps enabled, parse <|t.tt|> tokens into (start, end, text) segments
   - Apply quality checks (compression ratio, avg logprob); on failure restart at next temperature
```

### Long-Form Audio (>30s)

Two implementations:
- **OpenAI sequential**: transcribe chunk → use timestamp tokens to compute non-overlapping next-chunk offset → repeat. Higher quality on the tail; harder to parallelize.
- **HF chunked**: split audio into 30s chunks with 5s overlap → transcribe each independently → merge by deduplicating overlap regions via longest-common-substring on tokens. Trivially parallel; small quality regression.

We should implement sequential first (matches reference behavior, less brittle deduplication) and add chunked as an opt-in for parallelization in Phase 7 (server).

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

## Open Questions

- [ ] Whether we ship `_ALIGNMENT_HEADS` as a hardcoded table per model (matches OpenAI behavior) or skip word-timestamps in v1
- [ ] Whether to expose chunked long-form as an option in addition to sequential — chunked is easier to parallelize across GPU streams
- [ ] Beam search implementation strategy — beams add KV-cache duplication overhead (5x for default beam_size=5). May skip beam in v1, ship greedy + temperature fallback.

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

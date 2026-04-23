# Whisper — Research Notes

## Architecture

Whisper is OpenAI's encoder-decoder transformer for speech-to-text. The architecture consists of a Conv1D feature extractor (2 layers), a transformer encoder with sinusoidal positional encoding, and an autoregressive transformer decoder with learned positional embeddings and cross-attention to encoder outputs. Models range from 39M (tiny) to 1.55B (large-v3) parameters.

Sources: [OpenAI Whisper repo](https://github.com/openai/whisper), [Whisper paper](https://arxiv.org/abs/2212.04356), [whisper.cpp](https://github.com/ggml-org/whisper.cpp), [HuggingFace Whisper configs](https://huggingface.co/openai/whisper-large-v3)

### Model Size Configurations

All data confirmed from OpenAI model.py and HuggingFace config.json files.

| Model | Layers (enc/dec) | Width (d_model) | Heads | FFN dim | n_mels | Vocab | Parameters |
|-------|-----------------|-----------------|-------|---------|--------|-------|------------|
| tiny | 4 / 4 | 384 | 6 | 1536 | 80 | 51865 | 39M |
| base | 6 / 6 | 512 | 8 | 2048 | 80 | 51865 | 74M |
| small | 12 / 12 | 768 | 12 | 3072 | 80 | 51865 | 244M |
| medium | 24 / 24 | 1024 | 16 | 4096 | 80 | 51865 | 769M |
| large-v2 | 32 / 32 | 1280 | 20 | 5120 | 80 | 51865 | 1550M |
| large-v3 | 32 / 32 | 1280 | 20 | 5120 | 128 | 51866 | 1550M |
| large-v3-turbo | 32 / 4 | 1280 | 20 | 5120 | 128 | 51866 | 809M |

Key: FFN dim = 4 * d_model. Head dim = 64 for all sizes. n_audio_ctx = 1500. n_text_ctx = 448. large-v3-turbo has only 4 decoder layers (distilled). large-v3+ uses 128 mel bins.

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

## Mel Spectrogram

Whisper's audio preprocessing converts raw audio into a log-mel spectrogram with very specific parameters that must be matched exactly — even small deviations cause transcription errors. The pipeline is: resample to 16kHz -> zero-pad to 30s -> STFT (N_FFT=400, hop=160, Hann window) -> power spectrogram -> mel filterbank projection (80 or 128 bins, Slaney scale) -> log10 + dynamic range compression + normalization. Validation against whisper.cpp within 1e-4 tolerance is the acceptance criterion.

Sources: [OpenAI Whisper audio.py](https://github.com/openai/whisper/blob/main/whisper/audio.py), [whisper.cpp](https://github.com/ggml-org/whisper.cpp), [HuggingFace WhisperFeatureExtractor](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/feature_extraction_whisper.py)

### STFT Computation

From [audio.py](https://github.com/openai/whisper/blob/main/whisper/audio.py):

```python
stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=hann_window, return_complex=True)
magnitudes = stft[..., :-1].abs() ** 2   # power spectrogram, drop last frame
mel_spec = mel_filters @ magnitudes       # project to mel space
```

Key: last STFT frame is DROPPED, and `abs()^2` computes POWER spectrogram (squared magnitudes).

### Mel Filterbank

**Frequency range**: 0 Hz to 8000 Hz (Nyquist for 16kHz).

**Mel scale**: Slaney scale with Slaney area normalization (used by both OpenAI via librosa and HuggingFace).

Slaney mel scale formulas:
```
Hz to Mel:
  if f < 1000:  mel = 3 * f / 200
  if f >= 1000: mel = 15 + 27 * log(f / 1000) / log(6.4)

Mel to Hz:
  if mel < 15:  f = 200 * mel / 3
  if mel >= 15: f = 1000 * exp((mel - 15) * log(6.4) / 27)
```

**Filterbank shape**: [n_mels, N_FFT/2 + 1] = [80, 201] or [128, 201]

**Pre-computed**: OpenAI loads from `whisper/assets/mel_filters.npz` (generated with librosa). HuggingFace generates at runtime with `mel_filter_bank(mel_scale="slaney", norm="slaney")`.

### Log Compression (EXACT formula)

From [audio.py](https://github.com/openai/whisper/blob/main/whisper/audio.py):

```python
log_spec = torch.clamp(mel_spec, min=1e-10).log10()
log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
log_spec = (log_spec + 4.0) / 4.0
```

Step by step:
1. Floor clamp: `max(mel_spec, 1e-10)` — prevents log(0)
2. Log10: base-10 logarithm (NOT natural log)
3. Dynamic range: clamp to within 8.0 log10 units (80 dB) below max
4. Normalize: `(log_spec + 4.0) / 4.0` — maps typical range to ~[0, 1]

### Normalization

Global max-relative within each spectrogram:
- max_val = log_spec.max() across entire spectrogram
- Clamp to max_val - 8.0
- Shift and scale: (log_spec + 4.0) / 4.0

NO per-channel mean subtraction. NO per-channel normalization.

### Padding

Audio shorter than 30 seconds is zero-padded on the right to exactly 480,000 samples. Audio longer than 30s is processed in 30-second chunks.

### Output Tensor Shape

```
Input audio:        [480000]           (30s * 16000 Hz)
After STFT:         [201, 3001]        (N_FFT/2+1, N_SAMPLES/HOP+1)
Drop last frame:    [201, 3000]
After mel filters:  [80, 3000]         (or [128, 3000])
After log+norm:     [80, 3000]
Batched:            [1, 80, 3000]      (or [1, 128, 3000])
```

### Mel Filterbank Construction (Slaney)

```
1. Compute n_mels+2 mel-spaced center frequencies between 0 and 8000 Hz
2. Convert center frequencies to FFT bin indices
3. For each of n_mels filters:
   a. Create triangular filter from center[i] to center[i+2], peak at center[i+1]
   b. Normalize by mel band width (Slaney area normalization)
4. Result: [n_mels, N_FFT/2+1] filterbank matrix
```

### Complete Mel Spectrogram Pipeline

```
1. Resample audio to 16kHz if needed
2. Zero-pad to 480,000 samples (30s) on the right
3. Apply Hann window (periodic, length 400)
4. Compute STFT: window=400, hop=160 -> complex [201, 3001]
5. Drop last frame -> [201, 3000]
6. Compute power spectrogram: |STFT|^2 -> real [201, 3000]
7. Apply mel filterbank: [n_mels, 201] @ [201, 3000] -> [n_mels, 3000]
8. Log compress: log10(max(mel, 1e-10))
9. Dynamic range: max(log_spec, max(log_spec) - 8.0)
10. Normalize: (log_spec + 4.0) / 4.0
11. Output: [1, n_mels, 3000]
```

### Differences Between Implementations

| Aspect | OpenAI | whisper.cpp | HuggingFace |
|--------|--------|-------------|-------------|
| Filterbank | Pre-computed (mel_filters.npz) | Computed at load | Computed at runtime |
| Mel scale | Slaney (via librosa) | Slaney | Slaney |
| STFT | torch.stft | Custom FFT | numpy/torch |
| Normalization | Identical | Identical | Identical |

## Key Constants

| Constant | Value |
|----------|-------|
| Sample rate | 16,000 Hz |
| Chunk length | 30 seconds |
| N_SAMPLES | 480,000 |
| N_FFT | 400 (25ms window) |
| Hop length | 160 (10ms) |
| Window function | Hann (periodic) |
| N_FRAMES | 3000 |
| n_audio_ctx | 1500 |
| n_text_ctx | 448 |
| Head dim | 64 (all sizes) |
| Mel bins (standard) | 80 |
| Mel bins (large-v3) | 128 |
| Mel freq range | 0 - 8000 Hz |
| Filterbank shape | [80, 201] or [128, 201] |
| Log floor | 1e-10 |
| Dynamic range | 8.0 (log10 units = 80 dB) |
| Normalization offset | +4.0 |
| Normalization scale | /4.0 |
| Timestamp precision | 0.02s |
| Timestamp token offset | 50364 |

## Reference Implementations

- [OpenAI Whisper](https://github.com/openai/whisper) — Canonical Python implementation. Pre-computed filterbank from mel_filters.npz.
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) — C++ implementation with GGML. Validation target (1e-4 tolerance).
- [HuggingFace transformers](https://github.com/huggingface/transformers/tree/main/src/transformers/models/whisper) — Model configs, feature extractor, runtime filterbank generation.
- [librosa](https://librosa.org/doc/main/generated/librosa.filters.mel.html) — Original filterbank generation reference.

## Open Questions

- [ ] Whether large-v3-turbo decoder quality is sufficient for SharpInference use cases
- [ ] Optimal quantization level for Whisper models (Q8_0 vs Q5_K)
- [ ] Whether to ship pre-computed filterbank or generate at startup (generate recommended — avoids asset dependency)
- [ ] FFT implementation choice for C# (MathNet.Numerics vs custom)

## Implementation Notes

1. **Attention scaling**: Use `head_dim^(-0.25)` on both Q and K, NOT the standard `head_dim^(-0.5)` on QK product
2. **Key has no bias**: The key projection in both self-attention and cross-attention omits the bias term
3. **Decoder positional embeddings are learned** (unlike encoder which uses sinusoidal)
4. **Weight tying**: Output logits share weights with token embedding
5. **Support 80 and 128 mel bins**: large-v3+ uses 128, all others use 80
6. **GGML format**: Not GGUF — uses older GGML format with magic 0x67676d6c. Consider also supporting safetensors loading from HuggingFace
7. **KV-cache for decoder**: Cross-attention KV can be computed once and cached for all decoder steps
8. **Use Slaney mel scale** — both OpenAI and HuggingFace use this. HTK scale will produce wrong results.
9. **Power spectrum, not magnitude** — use |STFT|^2, not |STFT|
10. **Drop last STFT frame** — `stft[..., :-1]` is critical for correct shape
11. **Log10, not natural log** — the formula uses base-10 logarithm
12. **Generate filterbank at startup** — avoid shipping mel_filters.npz asset. Compute once and cache.
13. **Validation** — compare output against whisper.cpp mel computation within 1e-4 tolerance per element
14. **Hann window** — use periodic Hann window (not symmetric). In C#: `w[n] = 0.5 * (1 - cos(2*PI*n/N))` where N=400
15. **FFT** — need N_FFT=400 point FFT. Can use MathNet.Numerics or implement radix-2 with zero-padding to 512

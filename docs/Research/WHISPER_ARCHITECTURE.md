# Whisper Architecture — Research Notes

> Status: Complete
> Last Updated: 2026-04-16
> Needed Before: SharpInference.Audio

## Summary

Whisper is OpenAI's encoder-decoder transformer for speech-to-text. The architecture consists of a Conv1D feature extractor (2 layers), a transformer encoder with sinusoidal positional encoding, and an autoregressive transformer decoder with learned positional embeddings and cross-attention to encoder outputs. Models range from 39M (tiny) to 1.55B (large-v3) parameters.

Sources: [OpenAI Whisper repo](https://github.com/openai/whisper), [Whisper paper](https://arxiv.org/abs/2212.04356), [whisper.cpp](https://github.com/ggml-org/whisper.cpp), [HuggingFace Whisper configs](https://huggingface.co/openai/whisper-large-v3)

## Detailed Findings

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

## Key Numbers / Constants

| Constant | Value |
|----------|-------|
| Sample rate | 16,000 Hz |
| Chunk length | 30 seconds |
| N_SAMPLES | 480,000 |
| N_FFT | 400 (25ms window) |
| Hop length | 160 (10ms) |
| N_FRAMES | 3000 |
| n_audio_ctx | 1500 |
| n_text_ctx | 448 |
| Head dim | 64 (all sizes) |
| Timestamp precision | 0.02s |
| Timestamp token offset | 50364 |

## Reference Implementations

| Implementation | Location | Notes |
|---------------|----------|-------|
| OpenAI Whisper | [GitHub](https://github.com/openai/whisper) | Canonical Python implementation |
| whisper.cpp | [GitHub](https://github.com/ggml-org/whisper.cpp) | C++ implementation with GGML |
| HuggingFace | [transformers](https://github.com/huggingface/transformers/tree/main/src/transformers/models/whisper) | Model configs and feature extractor |

## Open Questions

- [x] ~~Conv1D kernel sizes~~ — Both kernel=3, first stride=1, second stride=2
- [x] ~~Timestamp decoding algorithm~~ — Pair constraint, monotonicity, probability thresholding
- [x] ~~GGUF Whisper metadata~~ — Uses GGML format (not GGUF), 11 int32 hyperparams
- [ ] Whether large-v3-turbo decoder quality is sufficient for SharpInference use cases
- [ ] Optimal quantization level for Whisper models (Q8_0 vs Q5_K)

## Implementation Notes

1. **Attention scaling**: Use `head_dim^(-0.25)` on both Q and K, NOT the standard `head_dim^(-0.5)` on QK product
2. **Key has no bias**: The key projection in both self-attention and cross-attention omits the bias term
3. **Decoder positional embeddings are learned** (unlike encoder which uses sinusoidal)
4. **Weight tying**: Output logits share weights with token embedding
5. **Support 80 and 128 mel bins**: large-v3+ uses 128, all others use 80
6. **GGML format**: Not GGUF — uses older GGML format with magic 0x67676d6c. Consider also supporting safetensors loading from HuggingFace
7. **KV-cache for decoder**: Cross-attention KV can be computed once and cached for all decoder steps

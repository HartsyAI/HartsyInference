# T5 Architecture — Research Notes

> Status: Complete
> Last Updated: 2026-04-16
> Needed Before: SharpInference.Diffusion, SharpInference.Tokenizers

## Summary

T5 v1.1 XXL is the text encoder used by Flux and SD3 (as `text_encoder_2`). At inference time, only the encoder portion is needed — it processes tokenized text through 24 transformer encoder blocks and outputs contextualized embeddings of dimension 4096. The encoder-only variant contains approximately 4.76 billion parameters (roughly half of the full 11B encoder-decoder model). In FP16 the encoder weighs 9.53 GB; in Q8_0 GGUF format it compresses to 5.06 GB.

Key architectural distinctions from standard transformers:
- **Relative position bias** via learned scalar lookup table (not sinusoidal, not rotary)
- **GEGLU feed-forward** (gated GeLU) instead of standard ReLU FFN
- **RMSNorm** (no bias term) with pre-normalization (norm before sublayer, not after)
- **No absolute positional embeddings** — position information comes entirely from relative bias
- **SentencePiece Unigram tokenizer** (not BPE) with 32128 vocabulary

T5 v1.1 differs from original T5: uses GEGLU instead of ReLU in FFN, does not share embedding/output weights (`tie_word_embeddings=false`), and was pre-trained without dropout.

## Detailed Findings

### Model Configuration (T5 v1.1 XXL)

The exact configuration from `google/t5-v1_1-xxl` on HuggingFace:

| Parameter | Value | Notes |
|---|---|---|
| `d_model` | 4096 | Hidden dimension / embedding size |
| `d_ff` | 10240 | Feed-forward intermediate dimension |
| `d_kv` | 64 | Key/value projection dimension per head |
| `num_heads` | 64 | Number of attention heads |
| `num_layers` | 24 | Encoder layers (also 24 decoder layers, unused) |
| `vocab_size` | 32128 | 32000 base + 100 sentinel + 28 special |
| `relative_attention_num_buckets` | 32 | Buckets for relative position bias |
| `relative_attention_max_distance` | 128 | Beyond this, all positions share one bucket |
| `feed_forward_proj` | `gated-gelu` | GEGLU activation (not ReLU) |
| `layer_norm_epsilon` | 1e-6 | RMSNorm epsilon |
| `dropout_rate` | 0.0 | v1.1 disables dropout at inference |
| `tie_word_embeddings` | false | Embeddings not shared with output layer |
| `pad_token_id` | 0 | `<pad>` |
| `eos_token_id` | 1 | `</s>` |

### Encoder Block Structure

Each of the 24 encoder blocks contains two sublayers with pre-norm residual connections:

```
Input
  |
  +---> RMSNorm --> Self-Attention --> Dropout --> (+) residual
  |                                                |
  +------------------------------------------------+
  |
  +---> RMSNorm --> GEGLU FeedForward --> Dropout --> (+) residual
  |                                                    |
  +----------------------------------------------------+
  |
Output
```

After all 24 blocks, a final RMSNorm is applied to the output, followed by dropout.

The full encoder forward pass:
1. Token IDs -> Embedding lookup (vocab_size x d_model = 32128 x 4096)
2. No positional embedding added (relative bias is computed inside attention)
3. Pass through 24 encoder blocks
4. Final RMSNorm
5. Output shape: (batch, seq_len, 4096)

### Relative Position Bias

T5 uses a learned relative position bias that replaces absolute positional embeddings entirely. The bias is a scalar added directly to the attention logits (before softmax), not a vector added to values.

**Bias table shape:** `(num_buckets, num_heads)` = `(32, 64)`

This is stored as an `nn.Embedding(32, 64)` — only 2048 parameters, shared across all layers. The bias is computed only in the first encoder layer and broadcast to all subsequent layers.

**Bucketing algorithm** (from HuggingFace source, adapted from Mesh TensorFlow):

For bidirectional attention (encoder), `num_buckets` is halved to 16 per direction:
- Buckets 0-15: negative relative positions (key before query)
- Buckets 16-31: positive relative positions (key after query)

Within each direction (16 buckets):
- **Linear range (buckets 0-7):** Direct mapping — relative distance `d` maps to bucket `d` for `d < 8` (`max_exact = num_buckets // 2 = 8`)
- **Logarithmic range (buckets 8-15):** Logarithmic bucketing for distances 8 to 128
- **Beyond 128:** All distances >= 128 map to bucket 15 (the last bucket)

The logarithmic bucketing formula:
```
max_exact = num_buckets // 2  # = 8 (per direction)
bucket = max_exact + floor(
    log(distance / max_exact) / log(max_distance / max_exact) * (num_buckets - max_exact)
)
bucket = min(bucket, num_buckets - 1)  # clamp to 15
```

With the default values (num_buckets=32, max_distance=128):
- Per direction: 16 buckets
- Exact positions: 0..7 (8 buckets for distances 0-7)
- Log positions: 8..15 (8 buckets covering distances 8-128+)
- Distances >= 128 all map to the same bucket

**Implementation note:** The bias is computed once for the maximum sequence length and cached. The relative position matrix is `memory_position - query_position`, making it a Toeplitz-like structure.

### GEGLU Feed-Forward Network

T5 v1.1 uses a gated feed-forward network with GeLU activation (GEGLU), implemented as `T5DenseGatedActDense`:

```
Input (batch, seq, 4096)
  |
  +---> wi_0: Linear(4096, 10240, bias=False) --> GeLU activation --> gate
  |
  +---> wi_1: Linear(4096, 10240, bias=False) ----------------------> linear
  |
  gate * linear  (element-wise multiply)
  |
  wo: Linear(10240, 4096, bias=False)
  |
Output (batch, seq, 4096)
```

Key details:
- Two parallel input projections (`wi_0` and `wi_1`), each d_model -> d_ff
- `wi_0` output passes through GeLU, `wi_1` output stays linear
- Element-wise multiplication of the two paths (gating mechanism)
- Single output projection `wo`: d_ff -> d_model
- **No bias** on any linear layer (T5 uses no biases throughout)
- Dropout applied after gating, before output projection

This is different from standard FFN (single linear + ReLU + linear) and from the original T5 (which used ReLU, not GEGLU). The gating mechanism provides better gradient flow and expressiveness. Note that GEGLU uses 3 weight matrices instead of 2, so each FFN layer has 3 * d_model * d_ff = 3 * 4096 * 10240 = 125,829,120 parameters.

### RMSNorm (T5LayerNorm)

T5 uses RMSNorm, not standard LayerNorm. The implementation:

```
RMSNorm(x) = weight * x / sqrt(mean(x^2) + epsilon)
```

- **No bias** parameter (unlike standard LayerNorm which has both weight and bias)
- **No mean subtraction** (unlike LayerNorm which subtracts mean before computing variance)
- `weight` shape: `(d_model,)` = `(4096,)` — a learnable scale parameter
- `epsilon` = 1e-6
- Variance computed as `mean(x^2)` over the last dimension

Each encoder block has 2 RMSNorm layers (one before attention, one before FFN), plus 1 final RMSNorm after the last block = 24 * 2 + 1 = 49 RMSNorm layers total in the encoder.

### Self-Attention

Each attention layer uses multi-head attention with:
- 64 heads, each with d_kv = 64 (so total Q/K/V projection = 64 * 64 = 4096 = d_model)
- Q, K, V projections: `Linear(4096, 4096, bias=False)` (reshaped to 64 heads x 64 dims)
- Output projection: `Linear(4096, 4096, bias=False)`
- **No bias** on any projection
- Attention formula: `softmax((Q @ K^T + relative_bias) / sqrt(d_kv))` where `d_kv = 64`

### Tokenizer (SentencePiece Unigram)

T5 uses a SentencePiece tokenizer with the Unigram algorithm (not BPE):

- **Base vocabulary:** 32000 tokens from SentencePiece Unigram model
- **Sentinel tokens:** 100 extra IDs (`<extra_id_0>` through `<extra_id_99>`), used for span corruption during pre-training. These are indexed from the end of the vocabulary downward: `<extra_id_0>` = 32099, `<extra_id_99>` = 32000.
- **Special tokens:** `<pad>` = 0, `</s>` = 1, `<unk>` = 2
- **Total vocabulary size:** 32128

The SentencePiece model file (`spiece.model`) is a binary protobuf containing the Unigram language model. Unlike BPE which merges character pairs, Unigram starts with a large vocabulary and iteratively removes tokens that least affect the overall likelihood.

**Key difference from CLIP's tokenizer:** CLIP uses BPE with a 49408 vocabulary and 77-token context limit. T5's Unigram tokenizer with 32128 vocabulary supports up to 512 tokens by default (though Flux/SD3 pipelines may use different context lengths, commonly 77 or 256 tokens for efficiency).

**Implementation note for C#:** The SentencePiece `.model` file must be parsed as a protobuf. The tokenizer logic involves:
1. Building a trie/lattice from the vocabulary
2. Running Viterbi decoding to find the optimal segmentation
3. Handling the `_` (U+2581, LOWER ONE EIGHTH BLOCK) character used as a space replacement

## Key Numbers/Constants

| Constant | Value | Context |
|---|---|---|
| d_model | 4096 | Hidden dimension |
| d_ff | 10240 | FFN intermediate dim |
| d_kv | 64 | Per-head key/value dim |
| num_heads | 64 | Attention heads |
| num_encoder_layers | 24 | Encoder depth |
| vocab_size | 32128 | Total vocabulary |
| relative_attention_num_buckets | 32 | Position bias buckets |
| relative_attention_max_distance | 128 | Max distance for bucketing |
| max_exact (per direction) | 8 | Linear range of position bias |
| layer_norm_epsilon | 1e-6 | RMSNorm epsilon |
| Encoder-only params | ~4.76B | Approximate |
| FP32 encoder size | ~19.1 GB | From GGUF repo |
| FP16 encoder size | ~9.53 GB | From GGUF repo |
| Q8_0 encoder size | ~5.06 GB | From GGUF repo |
| Q5_K_M encoder size | ~3.39 GB | From GGUF repo |
| Q4_K_M encoder size | ~2.90 GB | From GGUF repo |
| Bias table total params | 2048 | 32 buckets x 64 heads |
| RMSNorm layers in encoder | 49 | 24*2 + 1 final |
| Sentinel tokens | 100 | `<extra_id_0>` to `<extra_id_99>` |

## Data Layouts/Formats

### Weight Tensor Names (Encoder Only)

HuggingFace safetensors naming convention:

```
shared.weight                                          # (32128, 4096) - token embeddings
encoder.embed_tokens.weight                            # alias of shared.weight (or separate if not tied)

encoder.block.{i}.layer.0.SelfAttention.q.weight       # (4096, 4096) - Q projection
encoder.block.{i}.layer.0.SelfAttention.k.weight       # (4096, 4096) - K projection
encoder.block.{i}.layer.0.SelfAttention.v.weight       # (4096, 4096) - V projection
encoder.block.{i}.layer.0.SelfAttention.o.weight       # (4096, 4096) - output projection
encoder.block.{i}.layer.0.layer_norm.weight            # (4096,) - RMSNorm before attention

encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight   # (10240, 4096) - GEGLU gate projection
encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight   # (10240, 4096) - GEGLU linear projection
encoder.block.{i}.layer.1.DenseReluDense.wo.weight     # (4096, 10240) - FFN output projection
encoder.block.{i}.layer.1.layer_norm.weight            # (4096,) - RMSNorm before FFN

encoder.final_layer_norm.weight                        # (4096,) - final RMSNorm
```

Where `{i}` ranges from 0 to 23.

**Relative position bias** is stored only on the first layer:
```
encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight  # (32, 64)
```

**Note:** Despite the module name `DenseReluDense`, T5 v1.1 actually uses GEGLU (not ReLU). The class name is a legacy artifact in the HuggingFace implementation.

### Parameter Count Breakdown (Encoder Only)

| Component | Shape | Params | Count |
|---|---|---|---|
| Token embeddings | (32128, 4096) | 131,596,288 | 1 |
| Q, K, V, O projections | (4096, 4096) each | 16,777,216 each | 24 * 4 = 96 |
| Attention RMSNorm | (4096,) | 4,096 each | 24 |
| FFN wi_0, wi_1 | (10240, 4096) each | 41,943,040 each | 24 * 2 = 48 |
| FFN wo | (4096, 10240) | 41,943,040 each | 24 |
| FFN RMSNorm | (4096,) | 4,096 each | 24 |
| Relative position bias | (32, 64) | 2,048 | 1 |
| Final RMSNorm | (4096,) | 4,096 | 1 |
| **Total** | | **~4.76B** | |

Detailed calculation:
- Embeddings: 131,596,288
- Per-layer attention: 4 * 4096 * 4096 = 67,108,864
- Per-layer FFN: 3 * 4096 * 10240 = 125,829,120
- Per-layer norms: 2 * 4096 = 8,192
- Per-layer total: 192,946,176
- All 24 layers: 4,630,708,224
- Position bias: 2,048
- Final norm: 4,096
- **Grand total: 4,762,310,656 parameters**

At FP16 (2 bytes/param): ~8.86 GB of raw weight data (the 9.53 GB GGUF F16 includes metadata overhead).

## Algorithm Steps

### Encoder Forward Pass

```
1. Tokenize input text using SentencePiece Unigram tokenizer
2. Look up token embeddings: x = embed_tokens[token_ids]  # (batch, seq_len, 4096)
3. Compute relative position bias once:
   a. Build relative position matrix: rel_pos[i,j] = j - i
   b. Apply bucketing function to get bucket indices
   c. Look up bias from embedding table: bias[bucket_idx]  # (1, 64, seq_len, seq_len)
4. For each encoder block i = 0..23:
   a. Self-Attention sublayer:
      - norm_x = RMSNorm(x)
      - Q = norm_x @ Wq, K = norm_x @ Wk, V = norm_x @ Wv
      - Reshape Q, K, V to (batch, 64, seq_len, 64)
      - attn_scores = (Q @ K^T) / sqrt(64) + position_bias
      - attn_weights = softmax(attn_scores, dim=-1)
      - attn_output = attn_weights @ V
      - Reshape to (batch, seq_len, 4096)
      - attn_output = attn_output @ Wo
      - x = x + attn_output
   b. Feed-Forward sublayer:
      - norm_x = RMSNorm(x)
      - gate = GeLU(norm_x @ wi_0)
      - linear = norm_x @ wi_1
      - ff_output = (gate * linear) @ wo
      - x = x + ff_output
5. x = RMSNorm(x)  # final layer norm
6. Return x  # (batch, seq_len, 4096)
```

### Relative Position Bucketing Algorithm

```
function relative_position_bucket(rel_pos, bidirectional=true, num_buckets=32, max_distance=128):
    n = -rel_pos
    if bidirectional:
        num_buckets = num_buckets / 2  # 16
        offset = (n < 0) * num_buckets  # 0 or 16
        n = abs(n)
    else:
        n = max(n, 0)

    max_exact = num_buckets / 2  # 8
    is_small = n < max_exact

    # Logarithmic bucketing for large distances
    val_if_large = max_exact + floor(
        log(n / max_exact) / log(max_distance / max_exact) * (num_buckets - max_exact)
    )
    val_if_large = min(val_if_large, num_buckets - 1)

    bucket = offset + (is_small ? n : val_if_large)
    return bucket
```

### SentencePiece Unigram Tokenization

```
1. Load spiece.model protobuf (contains vocabulary with log-probabilities)
2. Normalize input text (NFKC normalization by default)
3. Replace spaces with U+2581 (LOWER ONE EIGHTH BLOCK)
4. Build a lattice of all possible tokenizations
5. Run Viterbi algorithm to find most probable segmentation
6. Map subword pieces to token IDs
7. Append </s> (token ID 1) at end of sequence
8. Pad to target length with <pad> (token ID 0)
```

## Reference Implementations

| Implementation | Language | Notes |
|---|---|---|
| [HuggingFace transformers T5](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) | Python/PyTorch | Canonical reference, `T5EncoderModel` class for encoder-only |
| [HuggingFace T5 tokenizer](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/tokenization_t5.py) | Python | Wraps SentencePiece library |
| [Google SentencePiece](https://github.com/google/sentencepiece) | C++ | Core tokenizer library with protobuf model format |
| [city96/t5-v1_1-xxl-encoder-gguf](https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf) | GGUF | Encoder-only quantized weights for diffusion |
| [city96/t5-v1_1-xxl-encoder-bf16](https://huggingface.co/city96/t5-v1_1-xxl-encoder-bf16) | Safetensors | Encoder-only BF16 weights |
| [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) | Python | GGUF loading for diffusion pipelines |
| Original T5 paper: [Raffel et al., 2020](https://jmlr.org/papers/volume21/20-074/20-074.pdf) | — | "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer", JMLR Vol 21 |

## Differences Between Implementations

### T5 v1.0 vs T5 v1.1

| Aspect | T5 v1.0 | T5 v1.1 (used by Flux/SD3) |
|---|---|---|
| FFN activation | ReLU (`T5DenseReluDense`) | GEGLU (`T5DenseGatedActDense`) |
| Embedding sharing | Shared between encoder/decoder/output | Not shared (`tie_word_embeddings=false`) |
| Pre-training dropout | Enabled | Disabled |
| Pre-training data | C4 + downstream task mixing | C4 only |
| FFN param count per layer | 2 * d_model * d_ff | 3 * d_model * d_ff (extra gate) |

### HuggingFace vs GGUF Weight Names

HuggingFace safetensors uses hierarchical names like `encoder.block.0.layer.0.SelfAttention.q.weight`. GGUF uses a flattened naming scheme. The city96 GGUF conversion extracts only encoder weights (no decoder).

### Flux vs SD3 Usage of T5

Both Flux and SD3 use the T5 v1.1 XXL encoder as a text encoder, but:
- **Flux:** Uses T5 as `text_encoder_2` alongside CLIP-L. T5 embeddings are concatenated with CLIP embeddings.
- **SD3:** Uses T5 as one of three text encoders (CLIP-L, CLIP-G, T5-XXL). All three outputs are combined for conditioning.
- Both truncate or pad T5 output to a fixed sequence length (commonly 256 tokens for Flux, 77 for SD3, though this varies by implementation).

## Open Questions

- [x] **Exact relative position bias table size and maximum distance** — Resolved: 32 buckets x 64 heads = 2048 parameters. Max distance = 128, with logarithmic bucketing from distance 8 to 128.
- [x] **Whether T5-XXL can be safely quantized to Q8_0** — Resolved: Q8_0 is generally near-lossless for the diffusion model itself, but for T5 text encoding specifically, some users report subtle quality degradation (especially in hand details). FP16 is preferred when VRAM allows. Q5_K_M and above are recommended minimums. Non-imatrix quantization is used because llama.cpp does not support imatrix for T5.
- [x] **T5 tokenizer vocabulary size and special tokens** — Resolved: 32128 total (32000 base Unigram + 100 sentinel `<extra_id>` tokens + 28 special). Special tokens: `<pad>`=0, `</s>`=1, `<unk>`=2.

## Implementation Notes

### For SharpInference (C#/.NET 10)

1. **Encoder-only extraction:** Only load encoder weights + embedding + final norm. Decoder weights can be skipped entirely, saving ~50% memory and load time.

2. **No bias terms anywhere:** All Linear layers in T5 are bias-free. This simplifies the MatMul implementation — every weight application is a pure matrix multiply with no bias addition.

3. **RMSNorm is simpler than LayerNorm:** No mean subtraction, no bias — just `weight * x / sqrt(mean(x^2) + eps)`. Can be implemented as a single fused kernel.

4. **Relative position bias caching:** Compute the bias matrix once for the maximum sequence length used, then slice for shorter sequences. The bias is shared across all 24 layers, so it only needs to be computed once per forward pass.

5. **GEGLU requires 3 weight matrices per FFN layer** instead of the usual 2. Each layer's FFN has `wi_0`, `wi_1`, and `wo`. The gating multiply (`gelu(wi_0(x)) * wi_1(x)`) can be fused for efficiency.

6. **SentencePiece tokenizer in C#:** The `.model` file is a protobuf that must be parsed. Key considerations:
   - Generate C# classes from the SentencePiece protobuf schema
   - Implement Viterbi decoding for Unigram segmentation
   - Handle the U+2581 space replacement character
   - The tokenizer is deterministic (no sampling needed at inference)

7. **Memory budget:** At Q8_0, the encoder is 5.06 GB. For consumer GPUs with 8 GB VRAM, this leaves only ~3 GB for the diffusion model. Consider:
   - CPU offloading: Run T5 on CPU, then transfer embeddings to GPU for diffusion
   - Sequential loading: Load T5, compute embeddings, unload, then load diffusion model
   - Q4_K_M (2.90 GB) or Q5_K_M (3.39 GB) if quality is acceptable

8. **Attention mask:** For padded sequences, create a boolean mask where pad tokens (ID=0) are masked out. The mask is applied as a large negative value added to attention scores before softmax.

9. **Output format:** The encoder outputs `(batch, seq_len, 4096)` float tensors. For diffusion conditioning, these are typically projected by the diffusion model's own text projection layer (not part of T5 itself).

10. **GGUF format support:** The city96 GGUF models are the standard for quantized T5 in diffusion pipelines. Load using the GGUF format parser (see GGUF_FORMAT.md research doc).

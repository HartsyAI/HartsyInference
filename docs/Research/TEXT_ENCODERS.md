# Text Encoders — Research Notes

## CLIP

### Summary

CLIP (Contrastive Language-Image Pre-training) is a dual-encoder model that jointly learns image and text representations via contrastive loss. It consists of a Vision Transformer (ViT) image encoder and a causal Transformer text encoder that project into a shared embedding space. For diffusion pipelines (SD1.5, SDXL), only the text encoder is used to produce conditioning embeddings. For vision tasks (zero-shot classification, image search), both encoders are used. There are two major weight families: OpenAI's original CLIP and LAION's OpenCLIP, which have subtle but critical differences in projection implementation, precision handling, and optional architectural features (LayerScale, extra normalization) that affect weight loading. SDXL uses a dual-encoder setup combining CLIP-L/14 (OpenAI) and CLIP-bigG/14 (OpenCLIP), which differ significantly in scale and architecture.

### Vision Transformer (ViT) Image Encoder

The vision encoder processes images through the following pipeline:

1. **Patch Embedding**: Input image (224x224 by default) is divided into fixed-size patches via a 2D convolution with `kernel_size = patch_size` and `stride = patch_size`. For ViT-L/14, this produces 16x16 = 256 patches, each projected from `(3 * 14 * 14)` to `hidden_size` dimensions.

2. **CLS Token Prepend**: A learnable `class_embedding` (CLS token) is prepended to the patch sequence, making the sequence length `num_patches + 1` (e.g., 257 for 224x224 with patch_size=14).

3. **Positional Embedding**: Learned absolute positional embeddings are added (shape: `[num_patches + 1, hidden_size]`).

4. **Pre-Transformer LayerNorm** (`ln_pre`): Applied before entering the transformer blocks.

5. **Transformer Encoder**: Stack of `ResidualAttentionBlock` layers with pre-norm structure:
   - `x = x + Attention(LayerNorm_1(x))`
   - `x = x + MLP(LayerNorm_2(x))`
   - Full (non-causal) self-attention is used.

6. **Post-Transformer LayerNorm** (`ln_post`): Applied to the CLS token output (`x[:, 0, :]`).

7. **Projection** (`proj`): Linear projection (no bias) from vision `hidden_size` to shared `embed_dim`.

### Text Encoder

The text encoder is a causal (masked) Transformer:

1. **Tokenization**: Byte-level BPE tokenizer with vocabulary size 49,408:
   - 256 byte-level base tokens
   - ~49,150 learned BPE merge tokens
   - 2 special tokens: `<|startoftext|>` (SOT, ID 49406) and `<|endoftext|>` (EOT, ID 49407)
   - Context length: 77 tokens (including SOT and EOT)
   - Padding uses the EOT token

2. **Token + Positional Embedding**: Learned token embeddings + learned absolute positional embeddings.

3. **Causal Transformer**: Stack of `ResidualAttentionBlock` layers with **causal attention mask** (lower-triangular). Same pre-norm structure as vision encoder.

4. **Feature Extraction**: The representation at the **EOT token position** (found via `argmax` over token IDs, since EOT is the highest-ID token in each sequence) is extracted from the final layer output.

5. **Final LayerNorm** (`ln_final`): Applied to the extracted EOT representation.

6. **Text Projection**: Linear map from text `hidden_size` to shared `embed_dim`.

### Contrastive Training Objective

Both image and text features are L2-normalized, then compared via cosine similarity scaled by a learned temperature parameter (`logit_scale = log(1/0.07)` initialized). The loss is symmetric cross-entropy over the similarity matrix.

### Usage in Diffusion Pipelines

**Stable Diffusion 1.5**:
- Uses CLIP ViT-L/14 text encoder only
- Feeds the **full sequence of hidden states** from the last transformer layer (not the pooled CLS/EOT output) as cross-attention conditioning to the UNet
- Shape: `(batch, 77, 768)`
- Default: last layer output (clip_skip=1). clip_skip=2 means penultimate layer.

**SDXL**:
- Uses **two** text encoders:
  - Text Encoder 1: OpenAI CLIP ViT-L/14 — penultimate layer hidden states (768-dim)
  - Text Encoder 2: OpenCLIP ViT-bigG/14 — penultimate layer hidden states (1280-dim)
- Hidden states are concatenated along the feature dimension: `(batch, 77, 2048)`
- Pooled output from CLIP-G is used for the `text_embeds` / vector conditioning

### OpenAI CLIP vs OpenCLIP: Critical Differences

| Aspect | OpenAI CLIP | OpenCLIP |
|--------|-------------|----------|
| **text_projection** | `nn.Parameter` (raw weight matrix); applied as `x @ self.text_projection` | `nn.Linear(bias=False)` or `nn.Parameter` (configurable); applied as `self.text_projection(x)` or `x @ self.text_projection` |
| **visual.proj** | `nn.Parameter` | Same flexibility as text_projection |
| **Weight key names** | `text_projection` (single tensor) | `text_projection.weight` (if nn.Linear) or `text_projection` (if nn.Parameter) |
| **Precision** | Manual mixed precision with float16 weights stored directly | Designed for AMP autocast (float32 weights, mixed precision at runtime) |
| **LayerScale** | Not present | Optional `ls_init_value` parameter; adds `self.ls_1` and `self.ls_2` scaling after attention and MLP |
| **Extra LayerNorm** | Not present | Optional `ln_attn` (attention-specific layer norm), `qk_norm` (query-key normalization) |
| **Final LN after pool** | Always before pooling | Configurable: `final_ln_after_pool` flag |
| **Text pooling** | Fixed `argmax` on token IDs to find EOT | Configurable: `argmax`, `eos` (using explicit `eos_token_id`), and other strategies |
| **Output tokens** | Not supported | `output_tokens=True` to return full token-level features alongside pooled features |

**Weight Loading Implications**: When loading OpenAI CLIP weights into an OpenCLIP-style model (or vice versa), the text/visual projection keys must be handled carefully. OpenAI's `text_projection` is a 2D weight matrix `[hidden_size, embed_dim]`; if the target model uses `nn.Linear`, the key becomes `text_projection.weight`. The same applies to `visual.proj`.

### SDXL's CLIP-bigG/14 vs Standard CLIP

CLIP-bigG/14 used in SDXL is **not** the same architecture as standard CLIP models. Key differences:

- Much larger vision encoder: 48 layers (vs 12-24 for standard CLIP)
- Vision width of 1664 with head_width of 104 (16 heads) — non-standard width
- Text encoder has 32 layers with width 1280 (vs 12 layers / 512-768 for standard CLIP)
- Shared embedding dimension is 1280 (vs 512-768 for standard CLIP)
- Trained by LAION on LAION-2B dataset using OpenCLIP codebase
- Uses the OpenCLIP architecture features (potential LayerScale, configurable projection, etc.)

### Pre-Norm vs Post-Norm

**All CLIP variants (both OpenAI and OpenCLIP) use pre-norm (Pre-LN) within transformer blocks.** This means LayerNorm is applied *before* the attention and MLP operations, not after. This is consistent with the original ViT design and differs from the vanilla Transformer which uses post-norm.

The distinction to be aware of:
- **Inside blocks**: Pre-norm (LN before attention/MLP, residual added after)
- **After patch embed** (vision only): `ln_pre` — a LayerNorm applied after positional embedding, before the first block
- **After final block** (vision): `ln_post` — a LayerNorm applied to the CLS token output after all blocks
- **After final block** (text): `ln_final` — a LayerNorm applied to the EOT token output after all blocks

These "boundary" norms are separate from the pre-norm pattern within blocks and are always present in both OpenAI CLIP and OpenCLIP.

### Key Numbers / Constants

#### Model Variant Configurations

| Variant | Vision Layers | Vision Width | Vision Heads | Patch Size | Text Layers | Text Width | Text Heads | Embed Dim | Image Size |
|---------|--------------|-------------|-------------|-----------|------------|-----------|-----------|----------|-----------|
| ViT-B/32 | 12 | 768 | 12 | 32 | 12 | 512 | 8 | 512 | 224 |
| ViT-B/16 | 12 | 768 | 12 | 16 | 12 | 512 | 8 | 512 | 224 |
| ViT-L/14 | 24 | 1024 | 16 | 14 | 12 | 768 | 12 | 768 | 224 |
| ViT-H/14 (OpenCLIP) | 32 | 1280 | 16 | 14 | 24 | 1024 | 16 | 1024 | 224 |
| ViT-bigG/14 (OpenCLIP) | 48 | 1664 | 16 | 14 | 32 | 1280 | 20 | 1280 | 224 |

#### Shared Constants

| Constant | Value |
|----------|-------|
| Vocabulary size | 49,408 |
| Context length (max tokens) | 77 |
| SOT token ID | 49,406 |
| EOT token ID | 49,407 |
| Default image size | 224 x 224 |
| Image normalization mean | (0.48145466, 0.4578275, 0.40821073) |
| Image normalization std | (0.26862954, 0.26130258, 0.27577711) |
| Logit scale init | ln(1/0.07) = ~2.6593 |
| FFN intermediate size | 4x hidden_size (all standard variants) |
| Activation function | quick_gelu (OpenAI), gelu/quick_gelu (OpenCLIP, configurable) |

#### Diffusion-Specific Constants

| Pipeline | Text Encoder | Layer Used | Output Shape | Pooled Output |
|----------|-------------|-----------|-------------|--------------|
| SD 1.5 | CLIP ViT-L/14 | Last layer (clip_skip=1) | (B, 77, 768) | Not used |
| SDXL (enc1) | CLIP ViT-L/14 | Penultimate (layer -2) | (B, 77, 768) | Not used |
| SDXL (enc2) | CLIP ViT-bigG/14 | Penultimate (layer -2) | (B, 77, 1280) | 1280-dim vector |
| SDXL combined | Concatenated | — | (B, 77, 2048) | 1280-dim vector |

### Data Layouts / Formats

#### Vision Encoder Input
```
Input image: [B, 3, 224, 224]  (float32, normalized with CLIP mean/std)
After patch embed (conv2d): [B, num_patches, hidden_size]
  ViT-L/14: [B, 256, 1024]
  ViT-B/32: [B, 49, 768]
After CLS prepend: [B, num_patches + 1, hidden_size]
  ViT-L/14: [B, 257, 1024]
Positional embeddings: [num_patches + 1, hidden_size]
```

#### Text Encoder Input
```
Token IDs: [B, 77]  (int64, padded with EOT token)
Token embeddings: [B, 77, text_width]
Positional embeddings: [77, text_width]
Causal mask: [77, 77] (lower-triangular, -inf for masked positions)
```

#### Weight Key Mapping (OpenAI CLIP format)

```
visual.conv1.weight                          # [hidden, 3, patch, patch]
visual.class_embedding                       # [hidden]
visual.positional_embedding                  # [num_patches+1, hidden]
visual.ln_pre.weight / .bias                 # [hidden]
visual.transformer.resblocks.{i}.ln_1.weight / .bias
visual.transformer.resblocks.{i}.attn.in_proj_weight   # [3*hidden, hidden]
visual.transformer.resblocks.{i}.attn.in_proj_bias     # [3*hidden]
visual.transformer.resblocks.{i}.attn.out_proj.weight  # [hidden, hidden]
visual.transformer.resblocks.{i}.attn.out_proj.bias    # [hidden]
visual.transformer.resblocks.{i}.ln_2.weight / .bias
visual.transformer.resblocks.{i}.mlp.c_fc.weight       # [intermediate, hidden]
visual.transformer.resblocks.{i}.mlp.c_fc.bias         # [intermediate]
visual.transformer.resblocks.{i}.mlp.c_proj.weight     # [hidden, intermediate]
visual.transformer.resblocks.{i}.mlp.c_proj.bias       # [hidden]
visual.ln_post.weight / .bias               # [hidden]
visual.proj                                  # [hidden, embed_dim] (nn.Parameter)

transformer.resblocks.{i}.ln_1.weight / .bias
transformer.resblocks.{i}.attn.in_proj_weight          # [3*text_width, text_width]
transformer.resblocks.{i}.attn.in_proj_bias            # [3*text_width]
transformer.resblocks.{i}.attn.out_proj.weight         # [text_width, text_width]
transformer.resblocks.{i}.attn.out_proj.bias           # [text_width]
transformer.resblocks.{i}.ln_2.weight / .bias
transformer.resblocks.{i}.mlp.c_fc.weight              # [intermediate, text_width]
transformer.resblocks.{i}.mlp.c_fc.bias
transformer.resblocks.{i}.mlp.c_proj.weight            # [text_width, intermediate]
transformer.resblocks.{i}.mlp.c_proj.bias

ln_final.weight / .bias                      # [text_width]
text_projection                              # [text_width, embed_dim] (nn.Parameter)
positional_embedding                         # [77, text_width]
token_embedding.weight                       # [49408, text_width]
logit_scale                                  # scalar
```

#### HuggingFace Transformers Key Mapping (CLIPTextModel)

HuggingFace uses a different naming convention. The text encoder keys follow the pattern:
```
text_model.embeddings.token_embedding.weight
text_model.embeddings.position_embedding.weight
text_model.encoder.layers.{i}.self_attn.q_proj.weight / .bias
text_model.encoder.layers.{i}.self_attn.k_proj.weight / .bias
text_model.encoder.layers.{i}.self_attn.v_proj.weight / .bias
text_model.encoder.layers.{i}.self_attn.out_proj.weight / .bias
text_model.encoder.layers.{i}.layer_norm1.weight / .bias
text_model.encoder.layers.{i}.layer_norm2.weight / .bias
text_model.encoder.layers.{i}.mlp.fc1.weight / .bias
text_model.encoder.layers.{i}.mlp.fc2.weight / .bias
text_model.final_layer_norm.weight / .bias
text_projection.weight                       # [embed_dim, text_width] (nn.Linear, note transposed vs OpenAI)
```

Key difference: HuggingFace splits the fused `in_proj_weight` into separate `q_proj`, `k_proj`, `v_proj` weights. OpenAI/OpenCLIP keep them fused as a single `[3*hidden, hidden]` matrix.

### Algorithm Steps

#### Vision Encoding (Forward Pass)
```
1. x = conv2d(image, patch_embed_weight)          # [B, hidden, grid_h, grid_w]
2. x = reshape(x, [B, hidden, num_patches])        # flatten spatial dims
3. x = transpose(x, [B, num_patches, hidden])      # [B, N, D]
4. cls = broadcast(class_embedding, [B, 1, hidden]) # expand CLS
5. x = concat([cls, x], dim=1)                     # [B, N+1, D]
6. x = x + positional_embedding                    # add pos embed
7. x = layer_norm(x, ln_pre)                       # pre-transformer LN
8. for each block i:
     residual = x
     x_norm = layer_norm(x, ln_1[i])
     x_attn = multi_head_attention(x_norm, x_norm, x_norm)  # full attention
     x = residual + x_attn
     residual = x
     x_norm = layer_norm(x, ln_2[i])
     x_mlp = mlp(x_norm)                           # fc -> quick_gelu -> fc
     x = residual + x_mlp
9. x = x[:, 0, :]                                  # extract CLS token
10. x = layer_norm(x, ln_post)                      # post-transformer LN
11. x = x @ visual_proj                             # project to embed_dim
12. return x                                        # [B, embed_dim]
```

#### Text Encoding (Forward Pass)
```
1. x = token_embedding[token_ids]                   # [B, 77, text_width]
2. x = x + positional_embedding                     # add pos embed
3. for each block i:
     residual = x
     x_norm = layer_norm(x, ln_1[i])
     x_attn = multi_head_attention(x_norm, x_norm, x_norm, causal_mask)
     x = residual + x_attn
     residual = x
     x_norm = layer_norm(x, ln_2[i])
     x_mlp = mlp(x_norm)
     x = residual + x_mlp
4. x = layer_norm(x, ln_final)                      # final LN (all positions)
5. pooled = x[arange(B), argmax(token_ids, dim=-1)] # EOT token features
6. pooled = pooled @ text_projection                 # project to embed_dim
7. return pooled                                     # [B, embed_dim]
```

#### Text Encoding for Diffusion (hidden states output)
```
Steps 1-3 same as above.
4. For SD1.5: return x from step 3 (last layer)     # [B, 77, text_width]
   For SDXL: return x from layer[-2] (penultimate)   # [B, 77, text_width]
   (No ln_final, no projection, no pooling — raw hidden states)
```

#### Quick GELU Activation
```
quick_gelu(x) = x * sigmoid(1.702 * x)
```
This is an approximation of GELU that is faster to compute but slightly different from standard GELU. Both OpenAI CLIP and many OpenCLIP models use this variant.

### Differences Between Implementations

#### OpenAI CLIP vs OpenCLIP
1. **Projection layers**: `nn.Parameter` (OpenAI) vs `nn.Linear` or `nn.Parameter` (OpenCLIP). Affects weight key names in checkpoints.
2. **Precision**: OpenAI stores float16 weights directly. OpenCLIP stores float32, designed for AMP autocast.
3. **Optional features in OpenCLIP**: LayerScale (`ls_init_value`), attention LayerNorm (`ln_attn`), QK normalization (`qk_norm`), configurable final LN placement (`final_ln_after_pool`).
4. **Text pooling**: OpenAI uses fixed `argmax` to find EOT. OpenCLIP supports multiple strategies including explicit `eos_token_id`.

#### OpenAI/OpenCLIP vs HuggingFace Transformers
1. **Attention projections**: Fused `in_proj_weight` [3D, D] in OpenAI/OpenCLIP vs separate `q_proj`, `k_proj`, `v_proj` [D, D] in HuggingFace.
2. **Key naming**: Completely different hierarchies (see Data Layouts section above).
3. **text_projection**: `nn.Parameter` [text_width, embed_dim] in OpenAI (applied as `x @ proj`) vs `nn.Linear` [embed_dim, text_width] in HuggingFace (weight is transposed).

#### CLIP-L/14 vs CLIP-bigG/14
1. **Scale**: 12 text layers / 768 width vs 32 text layers / 1280 width.
2. **Vision**: 24 layers / 1024 width vs 48 layers / 1664 width (non-power-of-2).
3. **Embed dim**: 768 vs 1280.
4. **Training**: OpenAI WIT-400M vs LAION-2B.
5. **Architecture features**: bigG may use OpenCLIP-specific features (LayerScale, etc.) that standard CLIP-L does not.

### Implementation Notes

#### For SharpInference.Diffusion (text encoder only)

1. **SD1.5**: Implement CLIP ViT-L/14 text encoder. Output full hidden states `[B, 77, 768]` from the last transformer layer. Do NOT apply `ln_final` or `text_projection` — the UNet cross-attention consumes raw hidden states.

2. **SDXL**: Implement both CLIP-L/14 and CLIP-bigG/14 text encoders. For both, output the **penultimate** layer hidden states. Concatenate along feature dim to get `[B, 77, 2048]`. Also extract pooled output from CLIP-G (EOT token with LN + projection) for vector conditioning.

3. **Weight loading**: Must detect whether weights use OpenAI format (`text_projection` as raw tensor) or HuggingFace format (`text_projection.weight` as nn.Linear weight, Q/K/V split). Consider supporting both formats with automatic detection based on key patterns.

4. **Attention fusing**: OpenAI/OpenCLIP fuse Q/K/V into a single `in_proj_weight` `[3*D, D]`. HuggingFace splits them. Internal implementation should pick one representation and convert at load time.

#### For SharpInference.Vision (full CLIP)

1. Implement both vision and text encoders. Output L2-normalized embeddings in the shared `embed_dim` space.

2. Vision encoder needs the full pipeline: patch embed (conv2d), CLS token, positional embedding, transformer, LN, projection.

3. **Image preprocessing**: Resize to 224x224 (bicubic), center crop, normalize with CLIP-specific mean/std values (see Key Numbers above).

4. **quick_gelu**: Must use `x * sigmoid(1.702 * x)`, not standard GELU. Using standard GELU will produce slightly different embeddings.

5. **Cosine similarity**: Normalize both image and text features, compute dot product, scale by `exp(logit_scale)`.

#### Tokenizer Implementation

1. Implement byte-level BPE with the CLIP merge file (available from OpenAI CLIP repo or HuggingFace).
2. Vocabulary: 49,408 tokens. SOT=49406, EOT=49407.
3. Max sequence length: 77 tokens (including SOT and EOT, so 75 content tokens max).
4. Padding: Fill remaining positions with EOT token ID (49407).
5. Text is lowercased before tokenization in the original CLIP tokenizer.

### Sources

- [OpenAI CLIP Repository](https://github.com/openai/CLIP)
- [OpenCLIP Repository](https://github.com/mlfoundations/open_clip)
- [OpenCLIP Discussion #337: Differences between CLIP and OpenCLIP text encoders](https://github.com/mlfoundations/open_clip/discussions/337)
- [HuggingFace CLIP Documentation](https://huggingface.co/docs/transformers/model_doc/clip)
- [HuggingFace openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)
- [HuggingFace openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)
- [HuggingFace laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
- [HuggingFace laion/CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)
- [LAION Blog: Large scale OpenCLIP](https://laion.ai/blog/large-openclip/)
- [LAION Blog: Giant OpenCLIP ViT-G/14](https://laion.ai/blog/giant-openclip/)
- [Stability AI SDXL Discussion #37](https://github.com/Stability-AI/generative-models/issues/37)
- [HuggingFace Diffusers: SDXL Pipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl)
- [OpenAI CLIP Tokenizer Source](https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py)
- [CLIP Paper (Radford et al., 2021)](https://arxiv.org/abs/2103.00020)

---

## T5 v1.1 XXL

### Summary

T5 v1.1 XXL is the text encoder used by Flux and SD3 (as `text_encoder_2`). At inference time, only the encoder portion is needed — it processes tokenized text through 24 transformer encoder blocks and outputs contextualized embeddings of dimension 4096. The encoder-only variant contains approximately 4.76 billion parameters (roughly half of the full 11B encoder-decoder model). In FP16 the encoder weighs 9.53 GB; in Q8_0 GGUF format it compresses to 5.06 GB.

Key architectural distinctions from standard transformers:
- **Relative position bias** via learned scalar lookup table (not sinusoidal, not rotary)
- **GEGLU feed-forward** (gated GeLU) instead of standard ReLU FFN
- **RMSNorm** (no bias term) with pre-normalization (norm before sublayer, not after)
- **No absolute positional embeddings** — position information comes entirely from relative bias
- **SentencePiece Unigram tokenizer** (not BPE) with 32128 vocabulary

T5 v1.1 differs from original T5: uses GEGLU instead of ReLU in FFN, does not share embedding/output weights (`tie_word_embeddings=false`), and was pre-trained without dropout.

### Model Configuration

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

### Key Numbers / Constants

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

### Data Layouts / Formats

#### Weight Tensor Names (Encoder Only)

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

#### Parameter Count Breakdown (Encoder Only)

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

### Algorithm Steps

#### Encoder Forward Pass
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

#### Relative Position Bucketing Algorithm
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

#### SentencePiece Unigram Tokenization
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

### Differences Between Implementations

#### T5 v1.0 vs T5 v1.1

| Aspect | T5 v1.0 | T5 v1.1 (used by Flux/SD3) |
|---|---|---|
| FFN activation | ReLU (`T5DenseReluDense`) | GEGLU (`T5DenseGatedActDense`) |
| Embedding sharing | Shared between encoder/decoder/output | Not shared (`tie_word_embeddings=false`) |
| Pre-training dropout | Enabled | Disabled |
| Pre-training data | C4 + downstream task mixing | C4 only |
| FFN param count per layer | 2 * d_model * d_ff | 3 * d_model * d_ff (extra gate) |

#### HuggingFace vs GGUF Weight Names

HuggingFace safetensors uses hierarchical names like `encoder.block.0.layer.0.SelfAttention.q.weight`. GGUF uses a flattened naming scheme. The city96 GGUF conversion extracts only encoder weights (no decoder).

#### Flux vs SD3 Usage of T5

Both Flux and SD3 use the T5 v1.1 XXL encoder as a text encoder, but:
- **Flux:** Uses T5 as `text_encoder_2` alongside CLIP-L. T5 embeddings are concatenated with CLIP embeddings.
- **SD3:** Uses T5 as one of three text encoders (CLIP-L, CLIP-G, T5-XXL). All three outputs are combined for conditioning.
- Both truncate or pad T5 output to a fixed sequence length (commonly 256 tokens for Flux, 77 for SD3, though this varies by implementation).

### Implementation Notes

#### For SharpInference (C#/.NET 10)

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

### Sources

- [HuggingFace transformers T5](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) — Canonical reference, `T5EncoderModel` class
- [HuggingFace T5 tokenizer](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/tokenization_t5.py) — Wraps SentencePiece
- [Google SentencePiece](https://github.com/google/sentencepiece) — Core tokenizer library with protobuf model format
- [city96/t5-v1_1-xxl-encoder-gguf](https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf) — Encoder-only quantized weights
- [city96/t5-v1_1-xxl-encoder-bf16](https://huggingface.co/city96/t5-v1_1-xxl-encoder-bf16) — Encoder-only BF16 weights
- [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) — GGUF loading for diffusion pipelines
- [T5 Paper (Raffel et al., 2020)](https://jmlr.org/papers/volume21/20-074/20-074.pdf) — "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"

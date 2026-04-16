# CLIP Architecture — Research Notes

> Status: Complete
> Last Updated: 2026-04-16
> Needed Before: SharpInference.Vision, SharpInference.Diffusion

## Summary

CLIP (Contrastive Language-Image Pre-training) is a dual-encoder model that jointly learns image and text representations via contrastive loss. It consists of a Vision Transformer (ViT) image encoder and a causal Transformer text encoder that project into a shared embedding space. For diffusion pipelines (SD1.5, SDXL), only the text encoder is used to produce conditioning embeddings. For vision tasks (zero-shot classification, image search), both encoders are used. There are two major weight families: OpenAI's original CLIP and LAION's OpenCLIP, which have subtle but critical differences in projection implementation, precision handling, and optional architectural features (LayerScale, extra normalization) that affect weight loading. SDXL uses a dual-encoder setup combining CLIP-L/14 (OpenAI) and CLIP-bigG/14 (OpenCLIP), which differ significantly in scale and architecture.

## Detailed Findings

### 1. Vision Transformer (ViT) Image Encoder

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

### 2. Text Encoder

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

### 3. Contrastive Training Objective

Both image and text features are L2-normalized, then compared via cosine similarity scaled by a learned temperature parameter (`logit_scale = log(1/0.07)` initialized). The loss is symmetric cross-entropy over the similarity matrix.

### 4. Usage in Diffusion Pipelines

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

### 5. OpenAI CLIP vs OpenCLIP: Critical Differences

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

### 6. SDXL's CLIP-bigG/14 vs Standard CLIP

CLIP-bigG/14 used in SDXL is **not** the same architecture as standard CLIP models. Key differences:

- Much larger vision encoder: 48 layers (vs 12-24 for standard CLIP)
- Vision width of 1664 with head_width of 104 (16 heads) — non-standard width
- Text encoder has 32 layers with width 1280 (vs 12 layers / 512-768 for standard CLIP)
- Shared embedding dimension is 1280 (vs 512-768 for standard CLIP)
- Trained by LAION on LAION-2B dataset using OpenCLIP codebase
- Uses the OpenCLIP architecture features (potential LayerScale, configurable projection, etc.)

### 7. Pre-Norm vs Post-Norm

**All CLIP variants (both OpenAI and OpenCLIP) use pre-norm (Pre-LN) within transformer blocks.** This means LayerNorm is applied *before* the attention and MLP operations, not after. This is consistent with the original ViT design and differs from the vanilla Transformer which uses post-norm.

The distinction to be aware of:
- **Inside blocks**: Pre-norm (LN before attention/MLP, residual added after)
- **After patch embed** (vision only): `ln_pre` — a LayerNorm applied after positional embedding, before the first block
- **After final block** (vision): `ln_post` — a LayerNorm applied to the CLS token output after all blocks
- **After final block** (text): `ln_final` — a LayerNorm applied to the EOT token output after all blocks

These "boundary" norms are separate from the pre-norm pattern within blocks and are always present in both OpenAI CLIP and OpenCLIP.

## Key Numbers / Constants

### Model Variant Configurations

| Variant | Vision Layers | Vision Width | Vision Heads | Patch Size | Text Layers | Text Width | Text Heads | Embed Dim | Image Size |
|---------|--------------|-------------|-------------|-----------|------------|-----------|-----------|----------|-----------|
| ViT-B/32 | 12 | 768 | 12 | 32 | 12 | 512 | 8 | 512 | 224 |
| ViT-B/16 | 12 | 768 | 12 | 16 | 12 | 512 | 8 | 512 | 224 |
| ViT-L/14 | 24 | 1024 | 16 | 14 | 12 | 768 | 12 | 768 | 224 |
| ViT-H/14 (OpenCLIP) | 32 | 1280 | 16 | 14 | 24 | 1024 | 16 | 1024 | 224 |
| ViT-bigG/14 (OpenCLIP) | 48 | 1664 | 16 | 14 | 32 | 1280 | 20 | 1280 | 224 |

### Shared Constants

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

### Diffusion-Specific Constants

| Pipeline | Text Encoder | Layer Used | Output Shape | Pooled Output |
|----------|-------------|-----------|-------------|--------------|
| SD 1.5 | CLIP ViT-L/14 | Last layer (clip_skip=1) | (B, 77, 768) | Not used |
| SDXL (enc1) | CLIP ViT-L/14 | Penultimate (layer -2) | (B, 77, 768) | Not used |
| SDXL (enc2) | CLIP ViT-bigG/14 | Penultimate (layer -2) | (B, 77, 1280) | 1280-dim vector |
| SDXL combined | Concatenated | — | (B, 77, 2048) | 1280-dim vector |

## Data Layouts / Formats

### Vision Encoder Input
```
Input image: [B, 3, 224, 224]  (float32, normalized with CLIP mean/std)
After patch embed (conv2d): [B, num_patches, hidden_size]
  ViT-L/14: [B, 256, 1024]
  ViT-B/32: [B, 49, 768]
After CLS prepend: [B, num_patches + 1, hidden_size]
  ViT-L/14: [B, 257, 1024]
Positional embeddings: [num_patches + 1, hidden_size]
```

### Text Encoder Input
```
Token IDs: [B, 77]  (int64, padded with EOT token)
Token embeddings: [B, 77, text_width]
Positional embeddings: [77, text_width]
Causal mask: [77, 77] (lower-triangular, -inf for masked positions)
```

### Weight Key Mapping (OpenAI CLIP format)

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

### HuggingFace Transformers Key Mapping (CLIPTextModel)

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

## Algorithm Steps

### Vision Encoding (Forward Pass)
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

### Text Encoding (Forward Pass)
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

### Text Encoding for Diffusion (hidden states output)
```
Steps 1-3 same as above.
4. For SD1.5: return x from step 3 (last layer)     # [B, 77, text_width]
   For SDXL: return x from layer[-2] (penultimate)   # [B, 77, text_width]
   (No ln_final, no projection, no pooling — raw hidden states)
```

### Quick GELU Activation
```
quick_gelu(x) = x * sigmoid(1.702 * x)
```
This is an approximation of GELU that is faster to compute but slightly different from standard GELU. Both OpenAI CLIP and many OpenCLIP models use this variant.

## Reference Implementations

- **OpenAI CLIP**: https://github.com/openai/CLIP — Original implementation with manual mixed precision, nn.Parameter projections
- **OpenCLIP**: https://github.com/mlfoundations/open_clip — Extended implementation supporting larger models, nn.Linear projections, LayerScale, AMP
- **HuggingFace Transformers CLIPModel**: https://huggingface.co/docs/transformers/model_doc/clip — Split Q/K/V projections, different key naming
- **HuggingFace Diffusers CLIPTextModel**: Used in SD1.5/SDXL pipelines, outputs hidden states rather than pooled features
- **OpenAI CLIP Tokenizer**: https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py — BPE tokenizer reference
- **CLIP Paper (Radford et al., 2021)**: https://arxiv.org/abs/2103.00020

## Differences Between Implementations

### OpenAI CLIP vs OpenCLIP
1. **Projection layers**: `nn.Parameter` (OpenAI) vs `nn.Linear` or `nn.Parameter` (OpenCLIP). Affects weight key names in checkpoints.
2. **Precision**: OpenAI stores float16 weights directly. OpenCLIP stores float32, designed for AMP autocast.
3. **Optional features in OpenCLIP**: LayerScale (`ls_init_value`), attention LayerNorm (`ln_attn`), QK normalization (`qk_norm`), configurable final LN placement (`final_ln_after_pool`).
4. **Text pooling**: OpenAI uses fixed `argmax` to find EOT. OpenCLIP supports multiple strategies including explicit `eos_token_id`.

### OpenAI/OpenCLIP vs HuggingFace Transformers
1. **Attention projections**: Fused `in_proj_weight` [3D, D] in OpenAI/OpenCLIP vs separate `q_proj`, `k_proj`, `v_proj` [D, D] in HuggingFace.
2. **Key naming**: Completely different hierarchies (see Data Layouts section above).
3. **text_projection**: `nn.Parameter` [text_width, embed_dim] in OpenAI (applied as `x @ proj`) vs `nn.Linear` [embed_dim, text_width] in HuggingFace (weight is transposed).

### CLIP-L/14 vs CLIP-bigG/14
1. **Scale**: 12 text layers / 768 width vs 32 text layers / 1280 width.
2. **Vision**: 24 layers / 1024 width vs 48 layers / 1664 width (non-power-of-2).
3. **Embed dim**: 768 vs 1280.
4. **Training**: OpenAI WIT-400M vs LAION-2B.
5. **Architecture features**: bigG may use OpenCLIP-specific features (LayerScale, etc.) that standard CLIP-L does not.

## Open Questions

- [x] Exact differences between OpenAI CLIP and OpenCLIP weight layouts — **Resolved**: Primary differences are `nn.Parameter` vs `nn.Linear` for projections (affecting key names), plus optional LayerScale and extra norm layers in OpenCLIP.
- [x] Whether SDXL's CLIP-G uses the same architecture as standard CLIP — **Resolved**: No. CLIP-bigG/14 is significantly larger (32 text layers vs 12, 1280 width vs 768, 48 vision layers vs 24) and uses OpenCLIP's extended architecture with potential LayerScale support.
- [x] Patch normalization order (pre-norm vs post-norm) for each variant — **Resolved**: All variants use pre-norm within transformer blocks. Both OpenAI and OpenCLIP have `ln_pre` (before transformer), `ln_post` / `ln_final` (after transformer, before projection). The only configurable difference is OpenCLIP's `final_ln_after_pool` flag.

## Implementation Notes

### For SharpInference.Diffusion (text encoder only)

1. **SD1.5**: Implement CLIP ViT-L/14 text encoder. Output full hidden states `[B, 77, 768]` from the last transformer layer. Do NOT apply `ln_final` or `text_projection` — the UNet cross-attention consumes raw hidden states.

2. **SDXL**: Implement both CLIP-L/14 and CLIP-bigG/14 text encoders. For both, output the **penultimate** layer hidden states. Concatenate along feature dim to get `[B, 77, 2048]`. Also extract pooled output from CLIP-G (EOT token with LN + projection) for vector conditioning.

3. **Weight loading**: Must detect whether weights use OpenAI format (`text_projection` as raw tensor) or HuggingFace format (`text_projection.weight` as nn.Linear weight, Q/K/V split). Consider supporting both formats with automatic detection based on key patterns.

4. **Attention fusing**: OpenAI/OpenCLIP fuse Q/K/V into a single `in_proj_weight` `[3*D, D]`. HuggingFace splits them. Internal implementation should pick one representation and convert at load time.

### For SharpInference.Vision (full CLIP)

1. Implement both vision and text encoders. Output L2-normalized embeddings in the shared `embed_dim` space.

2. Vision encoder needs the full pipeline: patch embed (conv2d), CLS token, positional embedding, transformer, LN, projection.

3. **Image preprocessing**: Resize to 224x224 (bicubic), center crop, normalize with CLIP-specific mean/std values (see Key Numbers above).

4. **quick_gelu**: Must use `x * sigmoid(1.702 * x)`, not standard GELU. Using standard GELU will produce slightly different embeddings.

5. **Cosine similarity**: Normalize both image and text features, compute dot product, scale by `exp(logit_scale)`.

### Tokenizer Implementation

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
- [HuggingFace openai/clip-vit-large-patch14 config](https://huggingface.co/openai/clip-vit-large-patch14)
- [HuggingFace openai/clip-vit-base-patch32 config](https://huggingface.co/openai/clip-vit-base-patch32)
- [HuggingFace laion/CLIP-ViT-bigG-14-laion2B-39B-b160k config](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
- [HuggingFace laion/CLIP-ViT-H-14-laion2B-s32B-b79K config](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)
- [LAION Blog: Large scale OpenCLIP](https://laion.ai/blog/large-openclip/)
- [LAION Blog: Giant OpenCLIP ViT-G/14](https://laion.ai/blog/giant-openclip/)
- [Stability AI SDXL Discussion #37: Penultimate layer usage](https://github.com/Stability-AI/generative-models/issues/37)
- [HuggingFace Diffusers: SDXL Pipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl)
- [OpenAI CLIP Tokenizer Source](https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py)
- [CLIP Paper (Radford et al., 2021)](https://arxiv.org/abs/2103.00020)

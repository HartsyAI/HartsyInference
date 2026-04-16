# LoRA Format — Research Notes

> Status: Complete
> Last Updated: 2026-04-16
> Needed Before: SharpInference.Diffusion (Adapters)

## Summary

LoRA (Low-Rank Adaptation) decomposes weight updates into pairs of low-rank matrices stored in safetensors files. For each targeted layer, two matrices are stored: a down-projection A (rank x in_dim) and an up-projection B (out_dim x rank). The weight delta is computed as `delta_W = B x A x (alpha / rank)`, where alpha is a per-layer scaling factor and rank is the intrinsic dimension (typically 4-128). Three distinct naming conventions exist depending on the model family: SD1.5/SDXL use `lora_unet_*` / `lora_te_*` prefixes with UNet block paths, Flux uses `double_blocks` / `single_blocks` paths corresponding to its DiT architecture, and the diffusers library uses yet another format with `transformer.*` prefixes and `lora_A` / `lora_B` suffixes. LyCORIS variants (LoHa, LoKr) extend the concept with Hadamard and Kronecker product decompositions respectively.

For implementation, LoRA deltas can be applied either in-place (patching model weights before inference, faster at diffusion time) or at forward time (computing deltas on each forward pass, more flexible for multi-LoRA and dynamic strength). ComfyUI uses a deferred patching approach where weight patches are computed on access. Multiple LoRAs compose through simple additive stacking of deltas, each scaled by an independent strength multiplier.

## Detailed Findings

### Core LoRA Mathematics

The original LoRA paper ([Hu et al., 2022, arXiv:2106.09685](https://arxiv.org/abs/2106.09685)) defines the weight update for a pre-trained weight matrix W0 in R^(d x k):

```
W = W0 + delta_W = W0 + B * A
```

Where:
- **A** in R^(r x k): down-projection matrix (rank x input_dim)
- **B** in R^(d x r): up-projection matrix (output_dim x rank)
- **r** << min(d, k): the rank (intrinsic dimension)

At initialization, A uses random Gaussian values and B is initialized to zero, so delta_W = 0 at the start of training ([source](https://arxiv.org/abs/2106.09685)). This ensures the model starts from the exact pre-trained behavior.

The scaling factor alpha/rank decouples the magnitude of weight adjustments from the choice of rank. In practice, when alpha = rank, the scaling factor is 1.0 and the LoRA has "full strength". Common convention is to set alpha = rank during training, then modulate strength at inference time via an external multiplier ([source](https://apxml.com/courses/lora-peft-efficient-llm-training/chapter-2-lora-in-depth/lora-scaling-alpha)).

The complete inference-time formula with user-controlled strength is:

```
W_effective = W0 + strength * (alpha / rank) * (B @ A)
```

Note: recent research (rsLoRA, [arXiv:2312.03732](https://arxiv.org/abs/2312.03732)) argues that dividing by sqrt(rank) instead of rank produces more stable behavior at higher ranks.

### Safetensors Storage Format

LoRA weights are stored in standard safetensors files (see SAFETENSORS_FORMAT.md). Each targeted layer produces 2-3 tensor entries in the file:

| Key Suffix | Shape | Description |
|-----------|-------|-------------|
| `.lora_down.weight` | `[rank, in_dim]` | The A matrix (down-projection) |
| `.lora_up.weight` | `[out_dim, rank]` | The B matrix (up-projection) |
| `.alpha` | scalar `[1]` | Per-layer alpha scaling factor |

For Conv2d layers, a mid-projection tensor may also be present:

| Key Suffix | Shape | Description |
|-----------|-------|-------------|
| `.lora_mid.weight` | `[rank, rank, kH, kW]` | Convolution kernel (LoCon variant) |

The safetensors `__metadata__` section often contains training metadata:
- `ss_network_module`: identifies the network type (e.g., `networks.lora`, `lycoris.kohya`)
- `ss_network_dim`: the rank used during training
- `ss_network_alpha`: the alpha used during training
- `ss_network_args`: JSON with additional arguments (critical for LyCORIS variant detection)

### SD1.5 / SDXL Naming Convention (Kohya Format)

SD1.5 and SDXL LoRA files trained with kohya-ss use a flat key naming convention that encodes the UNet and text encoder module path with dots replaced by underscores ([source](https://github.com/kohya-ss/sd-scripts), [source](https://github.com/huggingface/diffusers/pull/2403)):

**UNet keys** use prefix `lora_unet_`:
```
lora_unet_{block}_{layer_path}.lora_down.weight
lora_unet_{block}_{layer_path}.lora_up.weight
lora_unet_{block}_{layer_path}.alpha
```

Where `{block}` is one of:
- `down_blocks_{N}` (N = 0..3 for SD1.5, 0..2 for SDXL)
- `mid_block`
- `up_blocks_{N}` (N = 0..3 for SD1.5, 0..2 for SDXL)

And `{layer_path}` traverses the UNet structure, for example:
- `attentions_{N}_transformer_blocks_{N}_attn1_to_q` (self-attention query)
- `attentions_{N}_transformer_blocks_{N}_attn1_to_k` (self-attention key)
- `attentions_{N}_transformer_blocks_{N}_attn1_to_v` (self-attention value)
- `attentions_{N}_transformer_blocks_{N}_attn1_to_out_0` (self-attention output)
- `attentions_{N}_transformer_blocks_{N}_attn2_to_q` (cross-attention query)
- `attentions_{N}_transformer_blocks_{N}_attn2_to_k` (cross-attention key)
- `attentions_{N}_transformer_blocks_{N}_attn2_to_v` (cross-attention value)
- `attentions_{N}_transformer_blocks_{N}_attn2_to_out_0` (cross-attention output)
- `attentions_{N}_transformer_blocks_{N}_ff_net_0_proj` (feed-forward gate)
- `attentions_{N}_transformer_blocks_{N}_ff_net_2` (feed-forward output)
- `resnets_{N}_conv1` / `resnets_{N}_conv2` (ResNet convolutions, if targeted)

**Text encoder keys** use prefix `lora_te_` (SD1.5) or `lora_te1_` / `lora_te2_` (SDXL):
```
lora_te_text_model_encoder_layers_{N}_self_attn_q_proj.lora_down.weight
lora_te_text_model_encoder_layers_{N}_self_attn_k_proj.lora_down.weight
lora_te_text_model_encoder_layers_{N}_self_attn_v_proj.lora_down.weight
lora_te_text_model_encoder_layers_{N}_self_attn_out_proj.lora_down.weight
lora_te_text_model_encoder_layers_{N}_mlp_fc1.lora_down.weight
lora_te_text_model_encoder_layers_{N}_mlp_fc2.lora_down.weight
```

For SDXL, `lora_te1_*` targets CLIP-L (text_encoder) and `lora_te2_*` targets CLIP-G (text_encoder_2) ([source](https://huggingface.co/docs/diffusers/api/loaders/lora)).

SD1.5 UNet has transformer blocks at specific positions: IN01, IN02, IN04, IN05, IN07, IN08, MID, OUT03-OUT11. Blocks IN00, IN03, IN06, IN09-IN11, OUT00-OUT02 have only Conv2d layers and no transformer blocks ([source](https://github.com/hako-mikan/sd-webui-lora-block-weight)).

### Flux LoRA Naming Convention

Flux uses a DiT (Diffusion Transformer) architecture with 19 double-stream blocks and 38 single-stream blocks (see FLUX_ARCHITECTURE.md). The LoRA naming differs significantly from SD1.5/SDXL ([source](https://github.com/huggingface/diffusers/blob/main/src/diffusers/loaders/lora_conversion_utils.py)).

**Three competing formats exist:**

#### 1. Kohya/sd-scripts Format

Uses `lora_unet_` prefix with underscored paths:

**Double blocks** (19 blocks, index 0-18):
```
lora_unet_double_blocks_{i}_img_attn_qkv.lora_down.weight     # fused Q+K+V for image stream
lora_unet_double_blocks_{i}_img_attn_qkv.lora_up.weight
lora_unet_double_blocks_{i}_img_attn_proj.lora_down.weight     # image attention output projection
lora_unet_double_blocks_{i}_img_attn_proj.lora_up.weight
lora_unet_double_blocks_{i}_img_mlp_0.lora_down.weight         # image MLP gate
lora_unet_double_blocks_{i}_img_mlp_0.lora_up.weight
lora_unet_double_blocks_{i}_img_mlp_2.lora_down.weight         # image MLP output
lora_unet_double_blocks_{i}_img_mlp_2.lora_up.weight
lora_unet_double_blocks_{i}_img_mod_lin.lora_down.weight       # image modulation
lora_unet_double_blocks_{i}_img_mod_lin.lora_up.weight
lora_unet_double_blocks_{i}_txt_attn_qkv.lora_down.weight     # fused Q+K+V for text stream
lora_unet_double_blocks_{i}_txt_attn_qkv.lora_up.weight
lora_unet_double_blocks_{i}_txt_attn_proj.lora_down.weight     # text attention output projection
lora_unet_double_blocks_{i}_txt_attn_proj.lora_up.weight
lora_unet_double_blocks_{i}_txt_mlp_0.lora_down.weight         # text MLP gate
lora_unet_double_blocks_{i}_txt_mlp_0.lora_up.weight
lora_unet_double_blocks_{i}_txt_mlp_2.lora_down.weight         # text MLP output
lora_unet_double_blocks_{i}_txt_mlp_2.lora_up.weight
lora_unet_double_blocks_{i}_txt_mod_lin.lora_down.weight       # text modulation
lora_unet_double_blocks_{i}_txt_mod_lin.lora_up.weight
```

**Single blocks** (38 blocks, index 0-37):
```
lora_unet_single_blocks_{i}_linear1.lora_down.weight           # fused Q+K+V+proj_mlp
lora_unet_single_blocks_{i}_linear1.lora_up.weight
lora_unet_single_blocks_{i}_linear2.lora_down.weight           # output projection
lora_unet_single_blocks_{i}_linear2.lora_up.weight
lora_unet_single_blocks_{i}_modulation_lin.lora_down.weight    # modulation
lora_unet_single_blocks_{i}_modulation_lin.lora_up.weight
```

Important: In Kohya format, `img_attn_qkv` and `txt_attn_qkv` are **fused** Q+K+V projections. Similarly, `single_blocks.linear1` fuses attention Q+K+V and the MLP gate projection into a single matrix. Loading these LoRAs requires splitting the fused weights to map them to the diffusers architecture.

#### 2. Diffusers (PEFT) Format

Uses dotted module paths with `lora_A` / `lora_B` suffixes:

**Double blocks** mapped to `transformer_blocks`:
```
transformer.transformer_blocks.{i}.attn.to_q.lora_A.weight
transformer.transformer_blocks.{i}.attn.to_q.lora_B.weight
transformer.transformer_blocks.{i}.attn.to_k.lora_A.weight
transformer.transformer_blocks.{i}.attn.to_k.lora_B.weight
transformer.transformer_blocks.{i}.attn.to_v.lora_A.weight
transformer.transformer_blocks.{i}.attn.to_v.lora_B.weight
transformer.transformer_blocks.{i}.attn.to_out.0.lora_A.weight
transformer.transformer_blocks.{i}.attn.to_out.0.lora_B.weight
transformer.transformer_blocks.{i}.attn.add_q_proj.lora_A.weight
transformer.transformer_blocks.{i}.attn.add_q_proj.lora_B.weight
transformer.transformer_blocks.{i}.attn.add_k_proj.lora_A.weight
transformer.transformer_blocks.{i}.attn.add_k_proj.lora_B.weight
transformer.transformer_blocks.{i}.attn.add_v_proj.lora_A.weight
transformer.transformer_blocks.{i}.attn.add_v_proj.lora_B.weight
transformer.transformer_blocks.{i}.attn.to_add_out.lora_A.weight
transformer.transformer_blocks.{i}.attn.to_add_out.lora_B.weight
transformer.transformer_blocks.{i}.ff.net.0.proj.lora_A.weight
transformer.transformer_blocks.{i}.ff.net.0.proj.lora_B.weight
transformer.transformer_blocks.{i}.ff.net.2.lora_A.weight
transformer.transformer_blocks.{i}.ff.net.2.lora_B.weight
transformer.transformer_blocks.{i}.ff_context.net.0.proj.lora_A.weight
transformer.transformer_blocks.{i}.ff_context.net.0.proj.lora_B.weight
transformer.transformer_blocks.{i}.ff_context.net.2.lora_A.weight
transformer.transformer_blocks.{i}.ff_context.net.2.lora_B.weight
transformer.transformer_blocks.{i}.norm1.linear.lora_A.weight
transformer.transformer_blocks.{i}.norm1.linear.lora_B.weight
transformer.transformer_blocks.{i}.norm1_context.linear.lora_A.weight
transformer.transformer_blocks.{i}.norm1_context.linear.lora_B.weight
```

**Single blocks** mapped to `single_transformer_blocks`:
```
transformer.single_transformer_blocks.{i}.attn.to_q.lora_A.weight
transformer.single_transformer_blocks.{i}.attn.to_q.lora_B.weight
transformer.single_transformer_blocks.{i}.attn.to_k.lora_A.weight
transformer.single_transformer_blocks.{i}.attn.to_k.lora_B.weight
transformer.single_transformer_blocks.{i}.attn.to_v.lora_A.weight
transformer.single_transformer_blocks.{i}.attn.to_v.lora_B.weight
transformer.single_transformer_blocks.{i}.proj_mlp.lora_A.weight
transformer.single_transformer_blocks.{i}.proj_mlp.lora_B.weight
transformer.single_transformer_blocks.{i}.proj_out.lora_A.weight
transformer.single_transformer_blocks.{i}.proj_out.lora_B.weight
transformer.single_transformer_blocks.{i}.norm.linear.lora_A.weight
transformer.single_transformer_blocks.{i}.norm.linear.lora_B.weight
```

#### 3. XLabs Format

Uses a different prefix scheme:
```
double_blocks.{i}.processor.proj_lora1.down.weight  →  attn.to_out.0
double_blocks.{i}.processor.proj_lora2.down.weight  →  attn.to_add_out
double_blocks.{i}.processor.qkv_lora1.down.weight   →  fused Q+K+V (image)
double_blocks.{i}.processor.qkv_lora2.down.weight   →  fused Q+K+V (text)
single_blocks.{i}.proj_lora.down.weight              →  proj_out
single_blocks.{i}.qkv_lora.down.weight               →  fused Q+K+V
```

#### Format Detection Strategy

Detect the format by inspecting key prefixes:
1. Keys starting with `lora_unet_double_blocks_` or `lora_unet_single_blocks_` → Kohya format
2. Keys starting with `transformer.transformer_blocks.` or `transformer.single_transformer_blocks.` → Diffusers format
3. Keys containing `.processor.` → XLabs format
4. Keys starting with `lora_unet_down_blocks_` / `lora_unet_up_blocks_` → SD1.5/SDXL Kohya format

The diffusers library implements conversion functions for all three Flux formats in `lora_conversion_utils.py` ([source](https://github.com/huggingface/diffusers/blob/main/src/diffusers/loaders/lora_conversion_utils.py)):
- `_convert_kohya_flux_lora_to_diffusers()`
- `_convert_xlabs_flux_lora_to_diffusers()`

#### QKV Fusion Handling

When converting from Kohya format (fused QKV) to diffusers format (separate Q/K/V), the conversion must split the fused matrices. The Kohya `_convert_to_ai_toolkit_cat()` function handles two cases:
- **Non-sparse**: the down_weight is distributed across all splits, and the up_weight is split along dimension 0
- **Sparse**: chunks are detected and split separately per head

For SharpInference, the recommended approach is to support all three Flux formats at load time and normalize internally to a single representation.

### LyCORIS Variants

LyCORIS ([KohakuBlueleaf/LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS)) extends the LoRA concept with alternative matrix decomposition methods. The main variants relevant for community models are LoHa and LoKr ([source](https://arxiv.org/html/2309.14859v2)).

#### LoHa (Low-rank Hadamard Product)

LoHa decomposes the weight update as a Hadamard (element-wise) product of two low-rank factorizations ([source](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/loha_lokr.md)):

```
delta_W = (W1b @ W1a) ⊙ (W2b @ W2a)
```

Where `⊙` denotes element-wise multiplication. This allows the effective rank to be up to r^2 while storing only 2x the parameters of standard LoRA at rank r.

**Safetensors key suffixes for LoHa:**
```
{prefix}.hada_w1_a    # W1a: [rank, in_dim]
{prefix}.hada_w1_b    # W1b: [out_dim, rank]
{prefix}.hada_w2_a    # W2a: [rank, in_dim]
{prefix}.hada_w2_b    # W2b: [out_dim, rank]
{prefix}.hada_t1      # Tucker decomposition tensor (optional, for Conv2d)
{prefix}.hada_t2      # Tucker decomposition tensor (optional, for Conv2d)
{prefix}.alpha        # Scalar scaling factor
```

The optional `hada_t1` and `hada_t2` tensors appear when Tucker decomposition is used for convolutional layers.

#### LoKr (Low-rank Kronecker Product)

LoKr decomposes the weight update using a Kronecker product ([source](https://huggingface.co/docs/peft/package_reference/lokr)):

```
delta_W = W1 ⊗ W2
```

Where W1 is always a full (small) matrix and W2 can be either full or low-rank decomposed (W2 = W2b @ W2a). The original weight dimensions are factored: e.g., a 512x512 matrix might become W1 (16x16) and W2 (32x32).

**Safetensors key suffixes for LoKr:**
```
{prefix}.lokr_w1      # W1: full matrix [factor_d, factor_k]
{prefix}.lokr_w2      # W2: full matrix (when rank is large enough)
{prefix}.lokr_w2_a    # W2a: [rank, remaining_k] (low-rank mode)
{prefix}.lokr_w2_b    # W2b: [remaining_d, rank] (low-rank mode)
{prefix}.lokr_w1_a    # W1a: [rank, factor_k] (rare, both factors low-rank)
{prefix}.lokr_w1_b    # W1b: [factor_d, rank] (rare, both factors low-rank)
{prefix}.lokr_t2      # Tucker tensor (optional, for Conv2d)
{prefix}.alpha        # Scalar scaling factor
```

The `factor` parameter controls dimension splitting: `factor=-1` (default) finds balanced factors automatically (e.g., 512 -> 16 x 32), `factor=N` forces a specific factorization.

#### LoCon (LoRA for Convolution)

LoCon is standard LoRA extended to Conv2d layers. It uses the same `lora_down.weight` / `lora_up.weight` keys but adds an optional `lora_mid.weight` for the spatial kernel dimensions.

#### Variant Detection

The safetensors `__metadata__` field `ss_network_module` identifies the variant:
- `networks.lora` → standard LoRA/LoCon
- `lycoris.kohya` → LyCORIS (check `ss_network_args` for specific algorithm)

At the key level, detect by suffix pattern:
- `lora_down.weight` / `lora_up.weight` → standard LoRA or LoCon
- `hada_w1_a` / `hada_w2_a` → LoHa
- `lokr_w1` / `lokr_w2` (or `lokr_w2_a`) → LoKr

ComfyUI uses a pluggable adapter system that checks each key against registered adapter classes, falling back to standard LoRA if no specific variant matches ([source](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/lora.py)).

### Multi-LoRA Stacking

Multiple LoRAs can be applied simultaneously with independent strength multipliers. There are several composition strategies ([source](https://arxiv.org/html/2402.16843v1)):

#### 1. Weight-Space Addition (In-Place Merge)

The simplest and most common approach. Deltas are summed:

```
W_effective = W0 + sum_i(strength_i * (alpha_i / rank_i) * (B_i @ A_i))
```

This is equivalent to merging all LoRA deltas into the base weights before inference. Pros: zero runtime overhead during diffusion. Cons: cannot easily undo or adjust strengths without recomputing from base weights; can destabilize as the number of LoRAs grows.

#### 2. Sequential Patching (Forward Time)

Each LoRA patch is applied sequentially during the forward pass. ComfyUI's `calculate_weight()` applies patches in order with individual strengths ([source](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/lora.py)). Mathematically equivalent to weight-space addition for standard LoRA, but allows:
- Dynamic strength adjustment between generations
- Easy addition/removal of individual LoRAs
- Support for non-additive patch types (replacement, scaling)

#### 3. LoRA Switch (Timestep-Based)

Different LoRAs are activated at different denoising timesteps. Each element is rendered by the most appropriate LoRA for that stage of denoising ([source](https://arxiv.org/html/2402.16843v1)).

#### 4. LoRA Composite (Score-Based)

Inspired by classifier-free guidance. Unconditional and conditional score estimates are computed from each LoRA at every denoising step, then averaged ([source](https://arxiv.org/html/2402.16843v1)).

For SharpInference, **weight-space addition** (strategy 1) should be the primary implementation, with deferred patching (strategy 2) as an option for flexibility.

### In-Place vs Forward-Time Application

This is a critical implementation decision ([source](https://github.com/lllyasviel/stable-diffusion-webui-forge/discussions/1038)):

| Aspect | In-Place (Weight Patching) | Forward-Time (Hook-Based) |
|--------|---------------------------|--------------------------|
| Diffusion speed | No overhead during inference | Slightly slower per step per LoRA |
| Multi-LoRA cost | No per-step cost | Linear cost with number of LoRAs |
| Memory | Must store patched weights (or base + delta) | Must store LoRA tensors + compute buffer |
| Flexibility | Must re-patch to change strength | Can change strength between steps |
| LoRA switching | Requires full re-patch | Instant |

ComfyUI uses a **deferred patching** hybrid: patches are registered on the ModelPatcher and computed when weights are accessed, but the result is cached. This gives the flexibility of forward-time with the performance of in-place once computed ([source](https://deepwiki.com/patientx/ComfyUI-Zluda/4.1-model-patching-and-lora)).

**Recommendation for SharpInference**: Implement in-place weight patching as the default (simplest, fastest for single-LoRA workflows). Provide an API to apply/unapply LoRAs by storing the original base weights separately. For multi-LoRA with dynamic strengths, consider a deferred-compute approach similar to ComfyUI.

## Key Numbers / Constants

| Parameter | Typical Range | Notes |
|-----------|--------------|-------|
| Rank (r) | 4 - 128 | Most community LoRAs use 4, 8, 16, 32, or 64 |
| Alpha | Often = rank | When alpha = rank, scaling factor = 1.0 |
| Strength multiplier | 0.0 - 2.0 | User-controlled; 1.0 = full trained effect |
| SD1.5 UNet LoRA modules | ~192 | Attention layers in transformer blocks |
| SDXL UNet LoRA modules | ~384 | Larger UNet with more attention layers |
| Flux LoRA modules | ~193 | 19 double blocks + 38 single blocks |
| Flux double blocks | 19 | Dual-stream (image + text) |
| Flux single blocks | 38 | Unified stream |
| LoRA file size (rank 16, SD1.5) | ~10-50 MB | Depends on targeted layers |
| LoRA file size (rank 64, Flux) | ~70-150 MB | More parameters due to larger hidden dims |
| LoKr file size | ~3-40 MB | Typically smaller than LoRA at same quality |
| LoHa parameter count | ~2x LoRA | At same rank, stores 4 matrices instead of 2 |

## Data Layouts / Formats

### Standard LoRA Tensor Shapes

For a Linear layer with weight shape `[out_features, in_features]`:

```
lora_down.weight: float16/float32 [rank, in_features]   # A matrix
lora_up.weight:   float16/float32 [out_features, rank]   # B matrix
alpha:            float32 [1]                             # scalar
```

For a Conv2d layer with weight shape `[out_channels, in_channels, kH, kW]`:

```
lora_down.weight: float16/float32 [rank, in_channels, kH, kW]  # or [rank, in_channels, 1, 1]
lora_up.weight:   float16/float32 [out_channels, rank, 1, 1]
lora_mid.weight:  float16/float32 [rank, rank, kH, kW]          # optional (LoCon)
alpha:            float32 [1]
```

### LoHa Tensor Shapes

For a Linear layer with weight shape `[out_features, in_features]`:

```
hada_w1_a: [rank, in_features]
hada_w1_b: [out_features, rank]
hada_w2_a: [rank, in_features]
hada_w2_b: [out_features, rank]
alpha:     [1]
```

### LoKr Tensor Shapes

For a weight `[d, k]` factored as `[d1*d2, k1*k2]`:

```
lokr_w1:   [d1, k1]                    # always full matrix
lokr_w2:   [d2, k2]                    # full matrix (high-rank mode)
  -- OR --
lokr_w2_a: [rank, k2]                  # low-rank mode
lokr_w2_b: [d2, rank]                  # low-rank mode
alpha:     [1]
```

## Algorithm Steps

### Loading a LoRA File

```
1. Read safetensors header to get tensor metadata
2. Check __metadata__ for ss_network_module to determine variant type
3. Scan key prefixes to determine format:
   a. "lora_unet_down_blocks_" / "lora_unet_up_blocks_" → SD1.5/SDXL
   b. "lora_unet_double_blocks_" / "lora_unet_single_blocks_" → Flux (Kohya)
   c. "transformer.transformer_blocks." → Flux (Diffusers)
   d. Keys with ".processor." → Flux (XLabs)
4. Scan key suffixes to determine LoRA variant:
   a. "lora_down.weight" / "lora_up.weight" → Standard LoRA
   b. "hada_w1_a" → LoHa
   c. "lokr_w1" → LoKr
5. For each layer group, load the tensor data
6. Build a mapping from LoRA keys → model weight keys
```

### Applying a Standard LoRA Delta

```
1. For each targeted weight W in the model:
   a. Load A = lora_down.weight  (shape [rank, in_dim])
   b. Load B = lora_up.weight    (shape [out_dim, rank])
   c. Load alpha (scalar)
   d. Compute rank = A.shape[0]
   e. Compute scale = strength * (alpha / rank)
   f. If Conv2d with mid weight:
      delta = B @ (mid @ A)  (with appropriate reshape)
   g. Else:
      delta = B @ A
   h. W_new = W + scale * delta
2. Replace model weight with W_new
```

### Applying a LoHa Delta

```
1. For each targeted weight W:
   a. Load hada_w1_a, hada_w1_b, hada_w2_a, hada_w2_b
   b. Load alpha (scalar)
   c. rank = hada_w1_a.shape[0]
   d. scale = strength * (alpha / rank)
   e. term1 = hada_w1_b @ hada_w1_a    # [out_dim, in_dim]
   f. term2 = hada_w2_b @ hada_w2_a    # [out_dim, in_dim]
   g. delta = term1 ⊙ term2            # element-wise multiply
   h. W_new = W + scale * delta
```

### Applying a LoKr Delta

```
1. For each targeted weight W with shape [d, k]:
   a. Load lokr_w1 (shape [d1, k1])
   b. Load lokr_w2 or (lokr_w2_b, lokr_w2_a)
   c. Load alpha (scalar)
   d. If lokr_w2 exists (full mode):
      w2 = lokr_w2
   e. Else (low-rank mode):
      w2 = lokr_w2_b @ lokr_w2_a
   f. rank = lokr_w2_a.shape[0] if low-rank else min(w2.shape)
   g. scale = strength * (alpha / rank)
   h. delta = kronecker_product(lokr_w1, w2)   # [d1*d2, k1*k2] = [d, k]
   i. W_new = W + scale * delta
```

### Converting Kohya Flux Keys to Model Weights

```
For double_blocks.{i}.img_attn_qkv (fused):
  1. Load lora_down: [rank, 3*head_dim*num_heads]
  2. Load lora_up:   [3*head_dim*num_heads, rank]
  3. Split lora_up along dim 0 into 3 equal parts → up_q, up_k, up_v
  4. Map to:
     - model.double_blocks.{i}.img_attn.to_q  → (lora_down, up_q)
     - model.double_blocks.{i}.img_attn.to_k  → (lora_down, up_k)
     - model.double_blocks.{i}.img_attn.to_v  → (lora_down, up_v)

For single_blocks.{i}.linear1 (fused Q+K+V+proj_mlp):
  1. Load lora_down: [rank, hidden_dim]
  2. Load lora_up:   [qkv_dim + mlp_dim, rank]
  3. Split lora_up along dim 0 into 4 parts with known dims
  4. Map to separate Q, K, V, and proj_mlp targets
```

## Reference Implementations

| Implementation | Language | Notes |
|---------------|----------|-------|
| [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) | Python | Defines the community LoRA format; training scripts for SD1.5/SDXL/Flux |
| [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner) | Python | Flux-specific LoRA training with LoHa/LoKr support |
| [huggingface/diffusers](https://github.com/huggingface/diffusers) | Python | `lora_conversion_utils.py` handles all format conversions; PEFT integration |
| [Comfy-Org/ComfyUI](https://github.com/comfyanonymous/ComfyUI) | Python | `comfy/lora.py` for loading, pluggable adapter system, deferred weight patching |
| [KohakuBlueleaf/LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) | Python | Authoritative implementation for LoHa, LoKr, and other variants |
| [huggingface/peft](https://github.com/huggingface/peft) | Python | LoKr and LoHa support adapted from LyCORIS |
| [XLabs-AI/x-flux](https://github.com/XLabs-AI/x-flux-comfyui) | Python | XLabs Flux LoRA format (less common) |
| [cloneofsimo/lora](https://github.com/cloneofsimo/lora) | Python | Original SD LoRA implementation (historical) |
| LoRA paper ([Hu et al., 2022](https://arxiv.org/abs/2106.09685)) | — | Original LoRA formulation |
| LyCORIS paper ([KohakuBlueleaf et al., 2023](https://arxiv.org/abs/2309.14859)) | — | LyCORIS variants formulation |

## Differences Between Implementations

### Key Naming Divergence

| Format | Down-projection suffix | Up-projection suffix | Alpha suffix |
|--------|----------------------|---------------------|-------------|
| Kohya/sd-scripts | `.lora_down.weight` | `.lora_up.weight` | `.alpha` |
| Diffusers/PEFT | `.lora_A.weight` | `.lora_B.weight` | embedded in config |
| XLabs | `.down.weight` | `.up.weight` | not stored (uses default) |
| ComfyUI internal | `.lora_down.weight` | `.lora_up.weight` | `.alpha` |

### QKV Fusion Differences

- **Kohya format**: Stores fused QKV as a single LoRA pair. Requires splitting during conversion.
- **Diffusers format**: Stores separate Q, K, V LoRA pairs. No fusion handling needed.
- **XLabs format**: Stores fused QKV. Uses `torch.split()` with equal dimension splits.

### Scaling Convention

- **Kohya**: Stores alpha as a per-layer scalar tensor in the safetensors file
- **Diffusers/PEFT**: Stores alpha in the adapter config JSON, not in the weight file
- **ComfyUI**: Reads alpha from the safetensors file; if missing, defaults to rank (scale = 1.0)

### Multi-LoRA Application

- **A1111 WebUI**: Hook-based forward-time application; LoRA effect persists without weight modification; slower with many LoRAs
- **Forge WebUI**: In-place weight patching with cached base weights; faster at diffusion time
- **ComfyUI**: Deferred patching via ModelPatcher; patches computed on weight access, then cached
- **Diffusers**: PEFT-based; applies LoRA layers as wrapper modules; supports `set_adapters()` for multi-LoRA with weights

## Open Questions

- [x] Complete Flux LoRA naming convention mapping — documented above for Kohya, Diffusers, and XLabs formats
- [x] Whether LoRA deltas should be applied in-place or at forward time — in-place recommended as default, with deferred patching for multi-LoRA flexibility
- [x] LyCORIS variant support priority — **Priority 1**: Standard LoRA (most common), **Priority 2**: LoHa (moderate community usage, captures more nuanced patterns), **Priority 3**: LoKr (smaller files but less common), **Priority 4**: (IA)^3, DyLoRA (rare in diffusion context)
- [ ] Exact handling of DoRA (Weight-Decomposed Low-Rank Adaptation) — ComfyUI recognizes `.dora_scale` suffix but detailed implementation not yet researched
- [ ] Whether rsLoRA (sqrt(rank) scaling) should be supported as an option
- [ ] Performance benchmarks for in-place vs deferred patching in a .NET context

## Implementation Notes

### For SharpInference

1. **Format auto-detection**: Scan the first few keys of a safetensors file to determine the format (SD1.5, SDXL, Flux-Kohya, Flux-Diffusers, Flux-XLabs) and variant (LoRA, LoHa, LoKr). Use the `__metadata__` `ss_network_module` field as a hint when available.

2. **Internal representation**: Normalize all formats to an internal representation keyed by model weight path (e.g., `double_blocks.0.img_attn.to_q`). Store A, B, alpha, and variant type per layer.

3. **Fused QKV handling**: For Kohya Flux format, implement the QKV split logic at load time. Pre-split into separate Q/K/V LoRA pairs to simplify the application code.

4. **Weight application**: Implement `ApplyLoRA(model, lora, strength)` that:
   - Clones the base weight (or stores a reference for undo)
   - Computes `delta = B @ A` (or variant-specific formula)
   - Scales by `strength * alpha / rank`
   - Adds delta to the weight tensor

5. **Multi-LoRA**: Implement as sequential application of deltas to the base weight. Store the original base weights to allow re-application with different strengths.

6. **Memory optimization**: LoRA tensors are small (a few MB to ~150MB). Keep them in CPU memory and stream to GPU only during application. The delta computation itself is a single matmul per layer.

7. **LyCORIS support**: Implement LoHa and LoKr as separate `ILoRAVariant` strategies that plug into the same application pipeline. The Kronecker product for LoKr can be computed efficiently as a block matrix operation.

8. **Dtype handling**: LoRA weights are typically stored in float16. The delta computation should match the model's working precision. Cast A and B to the model weight dtype before computing the delta.

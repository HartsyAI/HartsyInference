# Flux Architecture — Research Notes

> Status: Complete
> Last Updated: 2026-04-16
> Needed Before: SharpInference.Diffusion (Flux)

## Table of Contents

1. [Summary](#1-summary)
2. [Detailed Findings](#2-detailed-findings)
3. [Key Numbers/Constants](#3-key-numbersconstants)
4. [Data Layouts/Formats](#4-data-layoutsformats)
5. [Algorithm Steps](#5-algorithm-steps)
6. [Reference Implementations](#6-reference-implementations)
7. [Differences Between Implementations](#7-differences-between-implementations)
8. [Open Questions](#8-open-questions)
9. [Implementation Notes](#9-implementation-notes)

---

## 1. Summary

Flux is a 12B-parameter Diffusion Transformer (DiT) from Black Forest Labs that uses **flow matching** instead of DDPM-based diffusion and processes image and text tokens **jointly** through transformer blocks. The architecture has two stages: **double-stream blocks** (19 layers) where image and text tokens have separate projections but share a single joint attention operation, followed by **single-stream blocks** (38 layers) where image and text tokens are concatenated into a single sequence and processed with a unified transformer block.

Key differences from SD1.5/SDXL:
- **Flow matching** replaces DDPM noise schedules with a continuous-time ODE (velocity prediction along straight interpolation paths)
- **DiT** replaces the UNet — no convolutions, skip connections, or downsampling/upsampling stages
- **Dual text encoders**: T5-XXL (up to 512 tokens, 4096-dim) + CLIP-L/14 (77 tokens, 768-dim pooled)
- **16-channel VAE** with 8x spatial compression (vs. 4-channel in SD1.5/SDXL)
- **2x2 patchification** of latents into token sequences
- **RoPE** (Rotary Position Embeddings) for both spatial and text positions
- **Guidance distillation** (dev) or **full distillation** (schnell) eliminates the need for true classifier-free guidance in most cases

---

## 2. Detailed Findings

### 2.1 Overall Architecture

The Flux transformer (called `Flux` in BFL source, `FluxTransformer2DModel` in diffusers) follows this data flow:

1. **VAE encode** the image to a 16-channel latent at 1/8 resolution
2. **Patchify** latents into 2x2 patches, producing a sequence of tokens each with `16 * 4 = 64` channels
3. **Project** image tokens from 64 channels to hidden_size (3072) via a linear layer (`img_in` / `x_embedder`)
4. **Project** T5 text embeddings from 4096 to 3072 via a linear layer (`txt_in` / `context_embedder`)
5. **Embed** timestep via sinusoidal embedding + MLP, combined with CLIP pooled embedding via MLP (`time_in` + `vector_in`)
6. Optionally embed guidance scale via MLP (`guidance_in`, only in dev/non-schnell models)
7. **Compute RoPE** positional embeddings from concatenated text+image position IDs
8. Pass through **19 DoubleStreamBlocks** — image and text streams maintained separately but share joint attention
9. **Concatenate** image and text tokens into a single sequence
10. Pass through **38 SingleStreamBlocks** — unified self-attention on the full sequence
11. Extract image tokens (discard text tokens from the single-stream output)
12. **Final layer** with adaptive LayerNorm + linear projection back to `patch_size^2 * out_channels = 4 * 64 = 256`
13. **Unpatchify** to reconstruct the latent tensor

### 2.2 FluxParams Configuration

From BFL source (`src/flux/util.py`):

```python
@dataclass
class FluxParams:
    in_channels: int        # 64 (dev/schnell), 128 (canny/depth), 384 (fill)
    out_channels: int       # 64
    vec_in_dim: int         # 768  (CLIP pooled embedding dim)
    context_in_dim: int     # 4096 (T5-XXL output dim)
    hidden_size: int        # 3072
    mlp_ratio: float        # 4.0
    num_heads: int          # 24
    depth: int              # 19   (number of double-stream blocks)
    depth_single_blocks: int # 38  (number of single-stream blocks)
    axes_dim: list[int]     # [16, 56, 56]
    theta: int              # 10_000
    qkv_bias: bool          # True
    guidance_embed: bool    # True (dev), False (schnell)
```

**flux-dev** and **flux-schnell** share identical architecture except `guidance_embed`:
- **dev**: `guidance_embed=True` — guidance scale is embedded as a conditioning signal, enabling guidance-distilled single-pass inference
- **schnell**: `guidance_embed=False` — fully distilled, generates in 1-4 steps with no guidance

### 2.3 DoubleStreamBlock

Each double-stream block maintains **separate** image and text processing paths that share a **joint attention** operation. From BFL `src/flux/modules/layers.py`:

**Components per block (image stream):**
- `img_mod`: Modulation layer (6 outputs: shift1, scale1, gate1, shift2, scale2, gate2)
- `img_norm1`: LayerNorm (no affine)
- `img_attn`: SelfAttention (QKV projection + output projection + QKNorm)
- `img_norm2`: LayerNorm (no affine)
- `img_mlp`: Linear(3072, 12288) -> GELU(tanh) -> Linear(12288, 3072)

**Components per block (text stream):** Identical structure with `txt_` prefix.

**Forward pass:**
```
1. Compute modulation params from timestep vec for both streams (6 params each)
2. For image: norm1 -> modulate(scale1, shift1) -> QKV projection -> QKNorm
3. For text:  norm1 -> modulate(scale1, shift1) -> QKV projection -> QKNorm
4. Concatenate [txt_q, img_q], [txt_k, img_k], [txt_v, img_v]
5. Apply RoPE to concatenated Q, K
6. Compute scaled_dot_product_attention on the concatenated sequence
7. Split attention output back into txt_attn and img_attn
8. img = img + gate1 * proj(img_attn)
9. img = img + gate2 * mlp(modulate(norm2(img), scale2, shift2))
10. txt = txt + gate1 * proj(txt_attn)
11. txt = txt + gate2 * mlp(modulate(norm2(txt), scale2, shift2))
```

The key insight: **joint attention** means Q, K, V from both streams are concatenated before the attention operation, so image tokens attend to text tokens and vice versa. But each stream has its own Q/K/V projections, output projection, and MLP. This is the MM-DiT (Multi-Modal DiT) pattern.

### 2.4 SingleStreamBlock

After double-stream processing, image and text tokens are concatenated into a single sequence. Each single-stream block processes this unified sequence.

**Components per block:**
- `modulation`: Modulation layer (3 outputs: shift, scale, gate)
- `pre_norm`: LayerNorm (no affine)
- `norm`: QKNorm (RMSNorm on Q and K separately)
- `linear1`: Linear(3072, 3072*3 + 12288) — fused QKV + MLP gate projection
- `linear2`: Linear(3072 + 12288, 3072) — fused attention output + MLP output
- `mlp_act`: GELU(approximate="tanh")

**Forward pass (parallel attention + MLP, from DiT paper arXiv:2302.05442):**
```
1. Compute modulation params (shift, scale, gate) from timestep vec
2. x_mod = (1 + scale) * pre_norm(x) + shift
3. [qkv, mlp_input] = split(linear1(x_mod), [3*3072, 12288])
4. q, k, v = reshape(qkv) to [B, H, L, D]
5. q, k = QKNorm(q, k)
6. Apply RoPE to q, k
7. attn_out = scaled_dot_product_attention(q, k, v)
8. output = linear2(concat(attn_out, gelu(mlp_input)))
9. x = x + gate * output
```

The single-stream block uses **parallel** attention and MLP computation — `linear1` computes both QKV and MLP input in one matrix multiply, then attention and MLP are computed independently and combined by `linear2`.

### 2.5 RoPE (Rotary Position Embeddings)

Flux uses axial RoPE across 3 position axes, with the head dimension (128) split across the axes as `[16, 56, 56]`.

**Position ID construction:**

Text tokens get position IDs of shape `(num_text_tokens, 3)` initialized to all zeros — they have no spatial position.

Image tokens get position IDs of shape `(H_packed * W_packed, 3)` where:
```python
latent_image_ids = torch.zeros(height, width, 3)
latent_image_ids[..., 1] = torch.arange(height)[:, None]  # y-coordinate
latent_image_ids[..., 2] = torch.arange(width)[None, :]   # x-coordinate
# Channel 0 is always 0 (reserved for a batch/temporal axis)
```

The combined position IDs are `concat(txt_ids, img_ids)` along the sequence dimension.

**Frequency computation (`rope` function in BFL `src/flux/math.py`):**

For each axis `i` with dimension `axes_dim[i]` and position values `pos`:
```python
def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out),
                       torch.sin(out),  torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()
```

This produces a 2x2 rotation matrix per frequency per position:
```
R(theta) = [[cos(theta), -sin(theta)],
            [sin(theta),  cos(theta)]]
```

where `theta = pos * omega` and `omega[k] = 1 / (10000 ^ (2k / dim))` for the k-th frequency pair.

**Per-axis breakdown:**
- Axis 0 (dim=16): 8 frequency pairs, encoding the batch/temporal dimension (always 0 for text and image in standard Flux)
- Axis 1 (dim=56): 28 frequency pairs, encoding the y-coordinate (height position)
- Axis 2 (dim=56): 28 frequency pairs, encoding the x-coordinate (width position)

Total: 8 + 28 + 28 = 64 frequency pairs = 128 dimensions = `hidden_size / num_heads`

The per-axis RoPE outputs are concatenated along the frequency dimension, then applied to Q and K:

```python
def apply_rope(xq, xk, freqs_cis):
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)
```

This applies the 2x2 rotation matrix to each pair of adjacent elements in the Q and K vectors. The matrix multiply is implemented as two dot products rather than an explicit matmul.

### 2.6 Modulation (Adaptive LayerNorm)

All blocks use **adaptive LayerNorm** conditioned on the timestep embedding. The timestep vector `vec` is processed through SiLU activation + linear projection to produce shift, scale, and gate parameters:

```python
class Modulation(nn.Module):
    def __init__(self, dim, double):
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec):
        out = self.lin(F.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)
        return ModulationOut(*out[:3]), ModulationOut(*out[3:]) if self.is_double else None
```

- **DoubleStreamBlock**: 6 modulation outputs (shift1, scale1, gate1 for attention; shift2, scale2, gate2 for MLP)
- **SingleStreamBlock**: 3 modulation outputs (shift, scale, gate)

Application: `x_modulated = (1 + scale) * LayerNorm(x) + shift`

### 2.7 QKNorm

Flux applies **RMSNorm** to Q and K before attention (QK-normalization), which improves training stability at scale:

```python
class RMSNorm(nn.Module):
    def __init__(self, dim):
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rrms = torch.rsqrt(torch.mean(x.float()**2, dim=-1, keepdim=True) + 1e-6)
        return (x.float() * rrms).to(x.dtype) * self.scale
```

QKNorm has separate learned scale parameters for Q and K. It normalizes per-head (applied after reshaping to `[B, H, L, D]`).

### 2.8 Text Encoders

**T5-XXL (T5-v1.1-XXL):**
- Parameters: ~4.7B (encoder only)
- Output dimension: 4096
- Max sequence length: 512 tokens (dev), 256 tokens (schnell)
- Projected to hidden_size 3072 via `txt_in` / `context_embedder`
- Size: ~11GB FP32, ~5.5GB FP16, ~3GB Q8_0

**CLIP-L/14:**
- Provides pooled embedding of dimension 768
- Projected through `vector_in` MLP and added to the timestep embedding
- Max 77 tokens (but only the pooled output is used, not per-token embeddings)
- Does NOT provide per-token conditioning — only the pooled vector

### 2.9 VAE

- **Channels**: 16 latent channels (vs. 4 in SD1.5/SDXL)
- **Spatial compression**: 8x (a 1024x1024 image becomes 128x128 latents)
- **Overall compression ratio**: 3 * 64 / 16 = 12:1
- Standard in_channels=64 for the DiT comes from patchifying the 16-channel latent with 2x2 patches: `16 * 2 * 2 = 64`

### 2.10 Flow Matching

Flux uses **rectified flow matching** rather than DDPM:

**Interpolation path (optimal transport):**
```
x_t = (1 - t) * x_0 + t * x_1
```
where x_0 is noise ~ N(0, I), x_1 is data, and t in [0, 1].

Equivalently, using sigma = 1 - t (where sigma represents noise level):
```
x_sigma = sigma * noise + (1 - sigma) * data
```

**Velocity prediction:**
The model predicts `v = x_1 - x_0 = data - noise`, which is constant along the straight interpolation path. The ODE is:
```
dx/dt = v_theta(x_t, t)
```

**Recovering data from velocity:**
```
x_0_pred (data) = x_t - sigma * v    [where sigma = 1 - t]
```

**Euler step (the entire scheduler step):**
```
x_next = x + v * dt
```
where `dt = sigma_next - sigma` (negative, moving toward less noise).

### 2.11 Sigma Schedule and Dynamic Shifting

**Base sigma schedule:**
```python
timesteps = linspace(1, num_train_timesteps, num_train_timesteps)  # reversed
sigmas = timesteps / num_train_timesteps
# sigmas go from 1.0 (pure noise) to ~0 (clean)
```

**Static shift (SD3 uses shift=3.0):**
```python
sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
```

**Dynamic shift (Flux):**

The shift varies based on image resolution (sequence length):
```python
def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096,
                    base_shift=0.5, max_shift=1.15):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu
```

Then the shift is applied using the `time_shift_type`:

**Exponential (default):**
```python
sigmas = exp(mu) * sigmas / (exp(mu) + (1/sigmas - 1))
# Equivalently: sigmas = exp(mu) / (exp(mu) + (1 - sigmas) / sigmas)
```

**Linear:**
```python
sigmas = mu * sigmas / (mu + (1/sigmas - 1))
```

**Example image sequence lengths:**
- 512x512 image: latent 64x64, packed 32x32 = 1024 tokens
- 1024x1024 image: latent 128x128, packed 64x64 = 4096 tokens
- 768x1024 image: latent 96x128, packed 48x64 = 3072 tokens

### 2.12 Guidance Handling

**Flux-dev (guidance distillation):**
The model was trained to replicate CFG output in a single forward pass. The guidance scale is passed as an embedded conditioning signal through `guidance_in` MLP. No second (unconditional) pass needed. Default guidance_scale = 3.5.

**Flux-schnell (full distillation):**
Fully distilled — no guidance mechanism at all. `guidance_embed=False`. Generates in 1-4 steps. Default guidance_scale = 0.0 (or 1.0, effectively unused).

**True CFG (optional for dev):**
When `true_cfg_scale > 1` and a negative prompt is provided:
```python
noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
```
This requires two forward passes (doubling compute).

### 2.13 Flux LoRA Format

Flux LoRA weights differ significantly from SD1.5/SDXL LoRA due to the DiT architecture. There are three main naming conventions in the ecosystem:

**BFL / PEFT format (native):**
```
double_blocks.{i}.img_attn.qkv.lora_A.weight
double_blocks.{i}.img_attn.qkv.lora_B.weight
double_blocks.{i}.img_attn.proj.lora_A.weight
double_blocks.{i}.img_mlp.0.lora_A.weight
...
single_blocks.{i}.linear1.lora_A.weight
single_blocks.{i}.linear1.lora_B.weight
single_blocks.{i}.linear2.lora_A.weight
...
```

**Diffusers format:**
```
transformer.transformer_blocks.{i}.attn.to_q.lora_A.weight
transformer.transformer_blocks.{i}.attn.to_k.lora_A.weight
transformer.transformer_blocks.{i}.attn.to_v.lora_A.weight
transformer.transformer_blocks.{i}.attn.to_out.0.lora_A.weight
transformer.transformer_blocks.{i}.ff.net.0.proj.lora_A.weight
transformer.transformer_blocks.{i}.ff.net.2.lora_A.weight
...
transformer.single_transformer_blocks.{i}.attn.to_q.lora_A.weight
transformer.single_transformer_blocks.{i}.proj_out.lora_A.weight
...
```

**ComfyUI / kohya format (flattened underscores):**
```
lora_unet_double_blocks_0_img_attn_proj.lora_down.weight
lora_unet_double_blocks_0_img_attn_proj.lora_up.weight
lora_unet_single_blocks_5_linear1.lora_down.weight
lora_unet_single_blocks_5_linear1.lora_up.weight
```

**Key conversion mapping (BFL to diffusers):**

| BFL Key | Diffusers Key |
|---------|---------------|
| `time_in.in_layer` | `time_text_embed.timestep_embedder.linear_1` |
| `time_in.out_layer` | `time_text_embed.timestep_embedder.linear_2` |
| `vector_in.in_layer` | `time_text_embed.text_embedder.linear_1` |
| `vector_in.out_layer` | `time_text_embed.text_embedder.linear_2` |
| `guidance_in.in_layer` | `time_text_embed.guidance_embedder.linear_1` |
| `guidance_in.out_layer` | `time_text_embed.guidance_embedder.linear_2` |
| `txt_in` | `context_embedder` |
| `img_in` | `x_embedder` |
| `double_blocks.{i}.img_mod.lin` | `transformer_blocks.{i}.norm1.linear` |
| `double_blocks.{i}.txt_mod.lin` | `transformer_blocks.{i}.norm1_context.linear` |
| `double_blocks.{i}.img_attn.qkv` | `transformer_blocks.{i}.attn.to_q/k/v` (split) |
| `double_blocks.{i}.txt_attn.qkv` | `transformer_blocks.{i}.attn.add_q/k/v_proj` (split) |
| `double_blocks.{i}.img_attn.proj` | `transformer_blocks.{i}.attn.to_out.0` |
| `double_blocks.{i}.txt_attn.proj` | `transformer_blocks.{i}.attn.to_add_out` |
| `double_blocks.{i}.img_mlp.{j}` | `transformer_blocks.{i}.ff.net.{j}` |
| `double_blocks.{i}.txt_mlp.{j}` | `transformer_blocks.{i}.ff_context.net.{j}` |
| `single_blocks.{i}.modulation.lin` | `single_transformer_blocks.{i}.norm.linear` |
| `single_blocks.{i}.linear1` | split into `attn.to_q/k/v` + `proj_mlp` |
| `single_blocks.{i}.linear2` | `single_transformer_blocks.{i}.proj_out` |

**Critical note on QKV splitting:** BFL stores a fused QKV matrix. Diffusers stores separate Q, K, V projections. For LoRA, BFL LoRA `lora_A` on a fused QKV must be split/replicated when converting to diffusers format. Similarly, `single_blocks.{i}.linear1` fuses QKV (3 * 3072) and MLP gate (12288) into a single weight, so conversion must split accordingly.

**LoRA delta computation (same as SD LoRA):**
```
delta_W = lora_B @ lora_A * (alpha / rank)
```
Typical ranks: 4, 8, 16, 32, 64, 128.

---

## 3. Key Numbers/Constants

### Architecture Constants

| Parameter | Value |
|-----------|-------|
| Total parameters | ~12B |
| Hidden size | 3072 |
| Number of attention heads | 24 |
| Head dimension | 128 (= 3072 / 24) |
| MLP ratio | 4.0 |
| MLP hidden dimension | 12288 (= 3072 * 4) |
| Double-stream blocks (depth) | 19 |
| Single-stream blocks (depth_single_blocks) | 38 |
| In channels (latent patch dim) | 64 |
| Out channels | 64 |
| Patch size | 2x2 (applied to 16-channel latents) |
| RoPE theta | 10,000 |
| RoPE axes dimensions | [16, 56, 56] (sum = 128 = head_dim) |
| QKV bias | True |
| LayerNorm epsilon | 1e-6 |
| RMSNorm epsilon | 1e-6 |
| GELU approximation | tanh |

### Text Encoder Constants

| Parameter | T5-XXL | CLIP-L/14 |
|-----------|--------|-----------|
| Output dimension | 4096 | 768 (pooled) |
| Max tokens | 512 (dev) / 256 (schnell) | 77 |
| Usage | Per-token embeddings -> context | Pooled embedding -> vec |
| Projected to | 3072 (context_in_dim -> hidden_size) | Added to timestep embed |
| Approximate size (FP16) | ~5.5 GB | ~250 MB |

### VAE Constants

| Parameter | Value |
|-----------|-------|
| Latent channels | 16 |
| Spatial compression | 8x |
| Scaling factor | 0.3611 (for Flux VAE) |
| Shift factor | 0.1159 (for Flux VAE) |

### Scheduler Constants (Flow-Matching Euler)

| Parameter | Value |
|-----------|-------|
| num_train_timesteps | 1000 |
| base_shift | 0.5 |
| max_shift | 1.15 |
| base_seq_len | 256 |
| max_seq_len | 4096 |
| time_shift_type | "exponential" (default) |
| Typical inference steps (dev) | 20-50 |
| Typical inference steps (schnell) | 1-4 |
| Default guidance_scale (dev) | 3.5 |

### Variant Differences

| Parameter | dev | schnell | canny/depth | fill |
|-----------|-----|---------|-------------|------|
| in_channels | 64 | 64 | 128 | 384 |
| guidance_embed | True | False | True | True |
| Blocks | 19+38 | 19+38 | 19+38 | 19+38 |
| Hidden size | 3072 | 3072 | 3072 | 3072 |

---

## 4. Data Layouts/Formats

### Latent Packing

Input latents of shape `[B, 16, H_lat, W_lat]` are packed into 2x2 patches:

```python
# Pack: [B, C, H, W] -> [B, (H/2)*(W/2), C*4]
latents = latents.view(B, C, H // 2, 2, W // 2, 2)
latents = latents.permute(0, 2, 4, 1, 3, 5)
latents = latents.reshape(B, (H // 2) * (W // 2), C * 4)
# Result: [B, seq_len, 64] where seq_len = H_lat/2 * W_lat/2
```

Unpacking reverses this:
```python
# Unpack: [B, seq_len, C*4] -> [B, C, H, W]
latents = latents.view(B, H // 2, W // 2, C, 2, 2)
latents = latents.permute(0, 3, 1, 4, 2, 5)
latents = latents.reshape(B, C, H, W)
```

### Position ID Layout

```
Text IDs:  [num_text_tokens, 3] — all zeros (no spatial position)
Image IDs: [H_packed * W_packed, 3] — channel 0 = 0, channel 1 = row, channel 2 = col

Combined:  [num_text_tokens + num_image_tokens, 3]
```

For a 1024x1024 image with max 512 text tokens:
- Latent: 128x128, packed: 64x64 = 4096 image tokens
- Text: up to 512 tokens
- Total sequence: up to 4608 tokens

### RoPE Embedding Layout

The `EmbedND` class produces position embeddings of shape `[1, total_seq_len, head_dim/2, 2, 2]`:

```python
# For each axis i:
#   rope(ids[..., i], axes_dim[i], theta)
#   produces shape [total_seq_len, axes_dim[i]//2, 2, 2]
# Concatenated along dim=-3 (the frequency dimension):
#   shape [total_seq_len, 64, 2, 2]  (since 8+28+28=64)
# Then unsqueeze(1) for broadcast across heads:
#   shape [1, total_seq_len, 64, 2, 2]
```

### Attention Tensor Shapes

In double-stream blocks:
```
Q, K, V (per-stream): [B, num_heads, stream_seq_len, head_dim]
Q, K, V (concatenated): [B, num_heads, total_seq_len, head_dim]
Attention output: [B, total_seq_len, hidden_size]
Split back: txt_attn [B, txt_len, hidden_size], img_attn [B, img_len, hidden_size]
```

In single-stream blocks:
```
Input x: [B, total_seq_len, hidden_size]   (text + image concatenated)
Q, K, V: [B, num_heads, total_seq_len, head_dim]
Output:  [B, total_seq_len, hidden_size]
```

### Timestep Embedding

The sinusoidal timestep embedding function:
```python
def timestep_embedding(t, dim, max_period=10000, time_factor=1000.0):
    t = time_factor * t       # Scale [0,1] timesteps to [0, 1000]
    half = dim // 2
    freqs = exp(-log(max_period) * arange(0, half) / half)
    args = t[:, None] * freqs[None]
    return concat([cos(args), sin(args)], dim=-1)
```

The time_factor=1000 scales the [0,1] flow-matching timestep to the [0, 1000] range expected by the sinusoidal embedding.

---

## 5. Algorithm Steps

### 5.1 Full Inference Pipeline (Text-to-Image)

```
Input: prompt (string), num_steps, guidance_scale, height, width, seed

1. Encode text:
   a. CLIP-L: tokenize(prompt) -> clip_model -> pooled_embed [1, 768]
   b. T5-XXL: tokenize(prompt, max_length=512) -> t5_model -> text_embeds [1, seq_len, 4096]

2. Prepare latents:
   a. H_lat = height // 8, W_lat = width // 8
   b. latents = randn(1, 16, H_lat, W_lat, seed=seed) * init_noise_sigma
   c. packed_latents = pack_latents(latents)  -> [1, H_lat/2 * W_lat/2, 64]

3. Prepare position IDs:
   a. txt_ids = zeros(seq_len, 3)
   b. img_ids = grid_positions(H_lat//2, W_lat//2, 3)  # [0, y, x] per position

4. Compute scheduler sigmas:
   a. image_seq_len = (H_lat // 2) * (W_lat // 2)
   b. mu = calculate_shift(image_seq_len)
   c. sigmas = linspace(1.0, 0.0, num_steps + 1)  # or with dynamic shift applied
   d. Apply exponential time shift with mu

5. For each step i = 0..num_steps-1:
   a. sigma = sigmas[i]
   b. timestep = sigma  (flow matching uses sigma directly as timestep in [0,1])
   c. Embed timestep: t_emb = timestep_embedding(sigma) -> MLP
   d. Embed guidance: g_emb = guidance_embedding(guidance_scale) -> MLP  [dev only]
   e. vec = t_emb + clip_pooled_emb + g_emb
   f. Compute RoPE from concat(txt_ids, img_ids)
   g. Project: img_tokens = img_in(packed_latents), txt_tokens = txt_in(text_embeds)
   h. Run 19 DoubleStreamBlocks: (img_tokens, txt_tokens) = block(img, txt, vec, pe)
   i. Concatenate: x = concat(txt_tokens, img_tokens)
   j. Run 38 SingleStreamBlocks: x = block(x, vec, pe)
   k. Extract image tokens: img_tokens = x[:, txt_len:]
   l. Final layer: output = last_layer(img_tokens, vec)  -> [B, seq_len, 256]
   m. Euler step: packed_latents = packed_latents + output * (sigmas[i+1] - sigmas[i])

6. Unpack latents: latents = unpack_latents(packed_latents)  -> [1, 16, H_lat, W_lat]
7. VAE decode: image = vae.decode(latents / scaling_factor + shift_factor)
8. Post-process: clip to [0,1], convert to uint8
```

### 5.2 Flow-Matching Euler Step (complete)

```python
def step(model_output, sigma, sigma_next, sample):
    dt = sigma_next - sigma   # negative value (moving toward clean)
    prev_sample = sample + model_output * dt
    return prev_sample
```

### 5.3 Dynamic Shift Calculation

```python
def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096,
                    base_shift=0.5, max_shift=1.15):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

# Then apply to sigmas (exponential type):
shift = exp(mu)
sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
```

---

## 6. Reference Implementations

### Primary Sources

- **BFL official repo**: [github.com/black-forest-labs/flux](https://github.com/black-forest-labs/flux) — canonical architecture implementation
  - `src/flux/model.py` — `FluxParams` dataclass, `Flux` class
  - `src/flux/modules/layers.py` — `DoubleStreamBlock`, `SingleStreamBlock`, `EmbedND`, `Modulation`, `QKNorm`
  - `src/flux/math.py` — `rope()`, `apply_rope()`, `attention()`
  - `src/flux/util.py` — model configs for dev/schnell/canny/depth/fill

- **Diffusers**: [github.com/huggingface/diffusers](https://github.com/huggingface/diffusers)
  - `src/diffusers/models/transformers/transformer_flux.py` — `FluxTransformer2DModel`
  - `src/diffusers/pipelines/flux/pipeline_flux.py` — `FluxPipeline`
  - `src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py` — `FlowMatchEulerDiscreteScheduler`
  - `src/diffusers/loaders/lora_conversion_utils.py` — LoRA key conversion functions

### Papers

- [Esser et al., 2024 — "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2403.03206) — SD3/Flux foundation (MMDiT, flow matching for image generation)
- [Lipman et al., 2023 — "Flow Matching for Generative Modeling"](https://arxiv.org/abs/2210.02747) — Flow matching framework
- [Peebles & Xie, 2023 — "Scalable Diffusion Models with Transformers (DiT)"](https://arxiv.org/abs/2212.09748) — DiT architecture
- [Dehghani et al., 2023 — "Scaling Vision Transformers to 22 Billion Parameters"](https://arxiv.org/abs/2302.05442) — Parallel attention+MLP blocks (used in SingleStreamBlock)
- [Su et al., 2024 — "RoFormer: Enhanced Transformer with Rotary Position Embedding"](https://arxiv.org/abs/2104.09864) — RoPE
- [Hu et al., 2022 — "LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685) — LoRA

### Other Resources

- [DeepWiki — Flux Model Architecture](https://deepwiki.com/black-forest-labs/flux/4.1-flux-model-architecture)
- [Demystifying Flux Architecture](https://arxiv.org/html/2507.09595v1)
- [HuggingFace FluxTransformer2DModel docs](https://huggingface.co/docs/diffusers/main/en/api/models/flux_transformer)
- [HuggingFace Flux Pipeline docs](https://huggingface.co/docs/diffusers/main/api/pipelines/flux)
- [FlowMatchEulerDiscreteScheduler docs](https://huggingface.co/docs/diffusers/api/schedulers/flow_match_euler_discrete)
- [Nunchaku LoRA Format Conversion](https://deepwiki.com/mit-han-lab/nunchaku/4.1-lora-format-conversion)
- [diffusers issue #9291 — kohya Flux LoRA key compatibility](https://github.com/huggingface/diffusers/issues/9291)

---

## 7. Differences Between Implementations

### BFL vs Diffusers Naming

| Concept | BFL Name | Diffusers Name |
|---------|----------|----------------|
| Image input projection | `img_in` | `x_embedder` |
| Text input projection | `txt_in` | `context_embedder` |
| Timestep embedder | `time_in` | `time_text_embed.timestep_embedder` |
| CLIP vec embedder | `vector_in` | `time_text_embed.text_embedder` |
| Guidance embedder | `guidance_in` | `time_text_embed.guidance_embedder` |
| Double-stream blocks | `double_blocks` | `transformer_blocks` |
| Single-stream blocks | `single_blocks` | `single_transformer_blocks` |
| Image modulation | `double_blocks.{i}.img_mod.lin` | `transformer_blocks.{i}.norm1.linear` |
| Text modulation | `double_blocks.{i}.txt_mod.lin` | `transformer_blocks.{i}.norm1_context.linear` |
| Image QKV (fused) | `double_blocks.{i}.img_attn.qkv` | `transformer_blocks.{i}.attn.to_q/k/v` (split) |
| Text QKV (fused) | `double_blocks.{i}.txt_attn.qkv` | `transformer_blocks.{i}.attn.add_q/k/v_proj` (split) |
| Single fused linear | `single_blocks.{i}.linear1` | split into `attn.to_q/k/v` + `proj_mlp` |
| Output projection | `single_blocks.{i}.linear2` | `single_transformer_blocks.{i}.proj_out` |
| Final norm | `final_layer.norm_final` | `norm_out.norm` |
| Final linear | `final_layer.linear` | `proj_out` |
| Final modulation | `final_layer.adaLN_modulation` | `norm_out.linear` |

### QKV Fusion Difference

BFL stores a **fused QKV** weight `[3 * hidden_size, hidden_size]` in each attention layer. Diffusers stores **separate** Q, K, V weights `[hidden_size, hidden_size]` each. This affects:
- Weight loading (must split/merge during conversion)
- LoRA application (fused LoRA on QKV vs. separate LoRA on Q, K, V)
- Memory layout and matmul patterns

### Single Block linear1 Fusion

BFL `single_blocks.{i}.linear1` is a single weight `[3*3072 + 12288, 3072] = [21504, 3072]` that fuses:
- QKV projection (3 * 3072 = 9216)
- MLP gate projection (12288)

Diffusers splits this into separate `to_q`, `to_k`, `to_v`, and `proj_mlp` weights.

### RoPE Implementation

BFL uses a 2x2 rotation matrix representation and applies it via element-wise operations with reshape tricks. Diffusers uses `apply_rotary_emb` with complex number or cos/sin pair representation. The mathematical result is identical.

---

## 8. Open Questions

- [x] Exact number of double-stream vs single-stream blocks: **19 double, 38 single** (same for dev and schnell)
- [x] RoPE frequency encoding for 2D image positions: **Axial RoPE with axes_dim=[16,56,56], theta=10000, applied per-axis then concatenated**
- [x] Flux LoRA naming convention: **Three formats documented (BFL, diffusers, ComfyUI/kohya) with full key mapping**
- [x] Flow-matching sigma schedule: **Linear sigmas with dynamic exponential shift based on image resolution**
- [ ] Exact VAE architecture details (encoder/decoder layer counts, channel progression) — deferred to VAE_ARCHITECTURE.md
- [ ] Whether BFL-format or diffusers-format LoRA is more common in the community (affects which to support first)
- [ ] FLUX.2 architecture changes vs FLUX.1 (reportedly 8 double-stream + 48 single-stream blocks)
- [ ] Optimal quantization strategy for the 12B transformer (Q8_0, Q4_K_M, etc.)

---

## 9. Implementation Notes

### Memory Considerations

The 12B parameter model at FP16 requires ~24GB VRAM. Key memory optimization strategies:
- **Attention**: The combined sequence length (text + image) can be large (e.g., 4608 for 1024x1024). Flash attention or memory-efficient attention is essential.
- **Sequential block execution**: Process blocks sequentially (not in parallel) to limit activation memory.
- **Offloading**: T5-XXL can be offloaded to CPU after encoding text (it is only needed once per generation).

### Precision Requirements

- **RoPE frequencies**: Compute in float32 (the `rope()` function explicitly returns `.float()`). Apply in float32, then cast back.
- **RMSNorm**: Cast to float32 for the mean/rsqrt computation, then back to model dtype.
- **Modulation**: The SiLU + linear can stay in model dtype (BF16/FP16).
- **Attention**: Can use FP16/BF16 with flash attention. The QKNorm before attention improves numerical stability.

### C# Implementation Strategy

1. **Shared infrastructure with SD3**: The MMDiT pattern (joint attention with separate streams) is shared. The main difference is SD3 uses 3 text encoders and different block counts.

2. **RoPE as a reusable component**: The `rope()` function is a simple frequency computation + 2x2 matrix construction. Can be precomputed for a given resolution and cached. The `apply_rope()` function is a simple element-wise operation that can be SIMD-vectorized.

3. **Modulation as a pattern**: Every block uses the same modulation pattern (SiLU -> Linear -> chunk -> apply as shift/scale/gate). This should be a shared utility.

4. **Fused vs. split QKV**: For implementation, start with the diffusers-style separate Q/K/V projections for clarity. Optimize to fused QKV later if profiling shows benefit.

5. **Packing/unpacking**: The latent packing is a simple reshape/permute — implement as a view operation if the tensor layout supports it, otherwise a copy.

6. **LoRA support**: Must handle both BFL-format (fused QKV) and diffusers-format (split Q/K/V) LoRA weights. Implement conversion at load time, applying in the diffusers layout.

7. **Weight loading**: Support both BFL safetensors (fused QKV, single linear1) and diffusers safetensors (split projections). Convert BFL format to internal representation at load time by splitting fused weights.

### Key Formulas for SIMD/GPU Kernels

**RMSNorm:**
```
rrms = 1 / sqrt(mean(x^2) + eps)
output = x * rrms * scale
```

**Modulation application:**
```
x_out = (1 + scale) * LayerNorm(x) + shift
```

**RoPE application (per pair of elements):**
```
x_out[2i]   = cos(theta) * x[2i] - sin(theta) * x[2i+1]
x_out[2i+1] = sin(theta) * x[2i] + cos(theta) * x[2i+1]
```

**Flow-matching Euler step (per element):**
```
x_next = x + v * dt
```

### Testing Strategy

1. Load a known BFL safetensors checkpoint, verify weight shapes match expected layout
2. Run single DoubleStreamBlock with known input, compare output against diffusers to within 1e-4
3. Run single SingleStreamBlock with known input, compare output against diffusers to within 1e-4
4. Verify RoPE output matches BFL implementation for a grid of positions
5. Run full pipeline for schnell (1-4 steps) and compare against diffusers output pixel-by-pixel (tolerance ~1e-2 for full pipeline)
6. Test LoRA loading from all three formats (BFL, diffusers, kohya/ComfyUI)

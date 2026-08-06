# Oasis-500m — Research Notes

> Status: Complete (full inference code + all configs captured from `etched-ai/open-oasis`; safetensors key dump still required) | Last Updated: 2026-05-24 | Needed Before: HartsyInference.World (`OasisPipeline`, Phase 10 — world models)
> Source of truth: [etched-ai/open-oasis (GitHub)](https://github.com/etched-ai/open-oasis), [HF `Etched/oasis-500m`](https://huggingface.co/Etched/oasis-500m), [Oasis blog](https://oasis-model.github.io/), [Decart publication](https://decart.ai/publications/oasis-interactive-ai-video-game-model)
> License: **MIT** (both code and weights)
> Related: [`DIFFUSION_SCHEDULERS.md`](DIFFUSION_SCHEDULERS.md), [`LANCE_ARCHITECTURE.md`](LANCE_ARCHITECTURE.md) (DiT lineage), [`FLOW_MATCHING_AUDIO.md`](FLOW_MATCHING_AUDIO.md), [`CONV2D_CUDA.md`](CONV2D_CUDA.md). Net-new module-class introduced by this doc: **continuous ViT-VAE** (not VQ — see § 6).

## Summary

Oasis is an **interactive, action-conditioned, autoregressive video world model** trained on Minecraft gameplay, released by Decart and Etched on 2024-10-31 under MIT. The publicly released checkpoint is a **down-scaled 500M-parameter "DiT-S/2"** transformer; the production demo at oasis.us.decart.ai is a larger model (referred to in press as 7B) that is **not** released. For HartsyInference, "Oasis-500m" means **only the 500M open checkpoint** — the cheap, pedagogical, real-time world model that serves as our CI smoke test and reference port for future world-model pipelines.

The architecture has two parts, both pure Transformers:

1. A **continuous Gaussian ViT-VAE** (`vit-l-20-shallow-encoder`, ~917 MB), patch-size 20, encoding 360 × 640 RGB frames into 18 × 32 = 576 latent tokens with 16 channels each. This is **NOT** a discrete VQ-VAE/VQ-GAN — it is a vanilla KL-regularised continuous autoencoder with a 16-dim Gaussian latent. (Section 6 explains why this matters for HartsyInference: no new VQ codebook infrastructure is required.)
2. A **spatio-temporal DiT** (`DiT-S/2`, 16 layers / hidden 1024 / 16 heads, ~2.43 GB FP16) operating in this 16-channel latent space. Each transformer block alternates a **spatial axial attention** (over the 18 × 32 = 576 patch tokens per frame, bidirectional, with 2-D rotary position embeddings) and a **temporal axial attention** (over the time axis at each spatial location, **causal** with 1-D rotary). Per-block adaptive layer-norm (adaLN-zero, DiT-style) is conditioned on a sum of (a) sinusoidal diffusion-timestep embedding and (b) a 25-dim Minecraft action vector projected to hidden through `nn.Linear(25, 1024)`.

Generation is **autoregressive in time, diffusion in space**: starting from N prompt frames already encoded to clean latents, the loop appends a Gaussian-noise latent for the next frame and denoises **only that frame** using 10 DDIM steps, while the prompt frames are held at a near-clean "stabilisation" noise level of 15/1000. This is a degenerate form of [**Diffusion Forcing**](https://www.boyuan.space/diffusion-forcing/) (Chen et al., NeurIPS 2024) where each frame can carry an independent noise level — at inference Oasis exploits this by giving the context near-zero noise and the target frame full noise. After 10 DDIM steps the new frame's clean latent is committed and the loop advances. The noise schedule is the **sigmoid β-schedule** with `T=1000`. A sliding window caps the visible context to `max_frames = 32` latent frames.

Action conditioning is a 25-dim per-frame vector covering the Minecraft VPT action space: 23 binary keys (`inventory`, `ESC`, `hotbar.1..9`, `forward/back/left/right`, `jump/sneak/sprint/swapHands/attack/use/pickItem/drop`) and 2 normalised camera floats (`cameraX`, `cameraY` ∈ [−1, 1]). The model emits frames at **20 FPS at 360 × 640** on H100; on Etched's Sohu it claims real-time at up to 4K. Inference cost is **47 ms / frame** on H100 (10 DDIM steps × the DiT-S/2 forward pass).

For HartsyInference, Oasis-500m is the **Phase 10 reference world model**: tiny (~3.4 GB total FP16), fully MIT, end-to-end in pure C# without any new heavy primitives, and trivially validated frame-by-frame against the upstream Python (because both action stream and prompt frames are deterministic given a fixed seed).

## Detailed Findings

### 1. Release artefacts and family

The released model on Hugging Face is a single checkpoint. There is no "tiny / base / large" family.

| File | Size | Format | Notes |
|---|---|---|---|
| `oasis500m.safetensors` | 2.43 GB | safetensors (FP32) | DiT-S/2 backbone |
| `oasis500m.pt` | 2.43 GB | PyTorch pickle | Same weights, `.pt` |
| `vit-l-20.safetensors` | 917 MB | safetensors (FP32) | ViT-VAE encoder + decoder |
| `vit-l-20.pt` | 917 MB | PyTorch pickle | Same weights, `.pt` |
| `README.md` | 893 B | — | Pointer to GitHub |
| `LICENSE` | 1.07 kB | — | MIT |
| `.gitattributes` | 115 B | — | LFS pointers |
| `media/` | — | dir | demo images, not weights |

**No `config.json` is shipped.** The architecture is hard-coded in `etched-ai/open-oasis/dit.py` and `vae.py`. A HartsyInference loader must hard-code these constants too (or read them from a `oasis_500m.json` we ship in our `model-data/` directory; recommended).

The HF repo is **gated** (requires accepting terms). HTTP 401 is returned for unauthenticated `raw/main/config.json` reads; this is incidental — there is no config to read.

A separate larger model (often referenced as "Oasis 7B") powers the live demo at oasis.us.decart.ai. **This is not released and is out of scope.** All numbers in this doc refer to the 500M open release.

### 2. Top-level inference loop (verbatim semantics)

From `generate.py`. Inputs: a prompt image or short video (resized to 360 × 640), a length-`total_frames` action stream of shape `(1, T, 25)`, sampling args.

Key numerical defaults:

| `generate.py` arg | Default | Meaning |
|---|---|---|
| `--num-frames` | 32 | Total output frames (including prompt) |
| `--n-prompt-frames` | 1 | Frames provided as clean prompt |
| `--ddim-steps` | 10 | DDIM denoising steps per new frame |
| `--fps` | 20 | Saved video frame rate |
| `max_noise_level` | 1000 | DDIM schedule discretisation |
| `noise_abs_max` | 20 | Clamp on initial Gaussian noise sample |
| `stabilization_level` | 15 | Noise index held on already-generated context frames |
| `scaling_factor` | **0.07843137255** (= 20 / 255) | VAE latent scaling |
| Seed | `torch.manual_seed(0)` | Reproducibility for CI |

Pipeline:

1. Load DiT (`DiT_models["DiT-S/2"]()`) and VAE (`VAE_models["vit-l-20-shallow-encoder"]()`).
2. Read prompt → resize to (360, 640) → normalise to `[0, 1]` → `* 2 - 1` to `[-1, 1]` → VAE encode → take `posterior.mean` (NOT a sample) → multiply by `scaling_factor`. Reshape from `(B*T, 576, 16)` to `(B, T, 16, 18, 32)`.
3. Load actions (`.actions.pt` raw dict-of-keys or `.one_hot_actions.pt` already-encoded `(T, 25)` float tensor). Prepend an all-zero frame so action `t` corresponds to "action that produced frame `t`". Shape becomes `(1, T+1, 25)`, sliced to `[:, :total_frames]`.
4. Pre-compute `betas = sigmoid_beta_schedule(1000)`, `alphas_cumprod = cumprod(1 - betas)`.
5. For each frame index `i ∈ [n_prompt_frames, total_frames)`:
   a. Append a fresh `N(0, I)` latent (clamped to ±20) as the (i+1)-th frame.
   b. Compute sliding-window `start_frame = max(0, i + 1 - 32)`.
   c. For `noise_idx` from `ddim_steps` down to 1 (10 → 1):
      - Build per-frame timestep tensors: every context frame gets `stabilization_level - 1 = 14`; the noisy target frame gets `noise_range[noise_idx]`.
      - Forward: `v = model(x[:, start_frame:], t, actions[:, start_frame:i+1])` — predicts **v-parameterisation** velocity (DiT-S/2 returns `v`).
      - Recover `x_start = √ᾱ_t · x_t − √(1−ᾱ_t) · v`, then `x_noise = (√(1/ᾱ_t) · x_t − x_start) / √(1/ᾱ_t − 1)`.
      - DDIM step: `x_pred = √ᾱ_{t_next} · x_start + √(1 − ᾱ_{t_next}) · x_noise`, except for the **context frames** which keep `α_next = 1` (i.e. they don't change), and on the **last step** the target frame also gets `α_next = 1` (i.e. clean).
      - Write only the target frame back: `x[:, -1:] = x_pred[:, -1:]`.
6. After the loop, decode: `pixels = (vae.decode(x / scaling_factor) + 1) / 2`, clamp to `[0, 1]`, scale to byte, write `.mp4` at 20 FPS.

The "stabilisation_level=15" trick is a Diffusion-Forcing inference-time noise schedule: previously committed frames are held at a tiny synthetic noise level (≈ 0.015 of max) to keep the model in distribution while it denoises the new frame at full noise. Without this, the DiT-S/2 model accumulates drift quickly.

### 3. DiT-S/2 backbone (`dit.py`)

Factory: `def DiT_S_2(): return DiT(patch_size=2, hidden_size=1024, depth=16, num_heads=16)`.

`DiT.__init__` signature with defaults (all but the factory overrides are class-level defaults that survive):

| Param | Value | Meaning |
|---|---|---|
| `input_h` | **18** | Latent grid height (= 360 / 20) |
| `input_w` | **32** | Latent grid width (= 640 / 20) |
| `patch_size` | **2** | DiT-internal latent patch size → 9 × 16 = **144 tokens per frame** after patchify |
| `in_channels` | **16** | VAE latent channels (= ViT-VAE `latent_dim`) |
| `hidden_size` | **1024** | Model dim |
| `depth` | **16** | Number of `SpatioTemporalDiTBlock`s |
| `num_heads` | **16** | Head dim = 1024/16 = **64** |
| `mlp_ratio` | **4.0** | MLP hidden = 4096 |
| `external_cond_dim` | **25** | Action vector dim (matches `ACTION_KEYS`) |
| `max_frames` | **32** | Sliding window cap on time axis |

Submodules:

- `x_embedder = PatchEmbed(18, 32, patch_size=2, in_chans=16, embed_dim=1024, flatten=False)` — `nn.Conv2d(16, 1024, kernel_size=2, stride=2)` followed by `rearrange("B C H W -> B H W C")`. Output shape `(B*T, 9, 16, 1024)`.
- `t_embedder = TimestepEmbedder(hidden_size=1024, frequency_embedding_size=256)` — standard DiT sinusoidal-then-MLP. Time scalars enter as integer indices in `[0, 1000)`.
- `spatial_rotary_emb = RotaryEmbedding(dim=1024 // 16 // 2 = 32, freqs_for="pixel", max_freq=256)`. Used for the **2-D axial** spatial RoPE inside `SpatialAxialAttention`.
- `temporal_rotary_emb = RotaryEmbedding(dim=1024 // 16 = 64)`. 1-D RoPE on the time axis.
- `external_cond = nn.Linear(25, 1024)` — projects per-frame action vector to hidden dim. Result is **added to** the per-frame timestep embedding `c`. So `c[b, t] = t_embedder(t[b, t]) + external_cond(action[b, t])`.
- `blocks = ModuleList(SpatioTemporalDiTBlock(...) × 16)`.
- `final_layer = FinalLayer(hidden_size=1024, patch_size=2, out_channels=16)` — adaLN-modulated `LayerNorm` followed by `Linear(1024, 2*2*16) = Linear(1024, 64)`. Then `unpatchify` to `(B*T, 16, 18, 32)`.

#### 3.1 `SpatioTemporalDiTBlock`

One block, given `x: (B, T, H, W, D)` and per-frame condition `c: (B, T, D)`:

```
# Spatial half — bidirectional, per-frame
s_shift_msa, s_scale_msa, s_gate_msa, s_shift_mlp, s_scale_mlp, s_gate_mlp = s_adaLN_modulation(c).chunk(6, dim=-1)
x = x + gate( s_attn( modulate(s_norm1(x), s_shift_msa, s_scale_msa) ),                s_gate_msa )
x = x + gate( s_mlp ( modulate(s_norm2(x), s_shift_mlp, s_scale_mlp) ),                s_gate_mlp )

# Temporal half — causal across time, per-(h,w)
t_shift_msa, t_scale_msa, t_gate_msa, t_shift_mlp, t_scale_mlp, t_gate_mlp = t_adaLN_modulation(c).chunk(6, dim=-1)
x = x + gate( t_attn( modulate(t_norm1(x), t_shift_msa, t_scale_msa) ),                t_gate_msa )
x = x + gate( t_mlp ( modulate(t_norm2(x), t_shift_mlp, t_scale_mlp) ),                t_gate_mlp )
```

All four norms are `nn.LayerNorm(elementwise_affine=False, eps=1e-6)`. Both adaLN modulators are `Sequential(SiLU, Linear(1024, 6*1024 = 6144))`. Both MLPs are `timm.models.vision_transformer.Mlp(in=1024, hidden=4096, act=GELU(approximate="tanh"))`.

#### 3.2 `SpatialAxialAttention` (per-frame, bidirectional)

`SpatialAxialAttention(dim=1024, heads=16, dim_head=64, rotary_emb=spatial_rotary_emb)`.

Reshape pattern: `(B, T, H, W, D)` →fused-QKV→ reshape to `((B*T), heads=16, H, W, head_dim=64)`. Apply **2-D axial RoPE** via `spatial_rotary_emb.get_axial_freqs(H, W)` → `apply_rotary_emb(freqs, q)` / `apply_rotary_emb(freqs, k)`. Flatten H and W into a single token axis of length **H × W = 9 × 16 = 144**. `F.scaled_dot_product_attention(q, k, v, is_causal=False)`. Output projection `nn.Linear(1024, 1024)`. **No qk-norm, no bias on `to_qkv`, bias on `to_out`.**

#### 3.3 `TemporalAxialAttention` (per-spatial-location, causal)

`TemporalAxialAttention(dim=1024, heads=16, dim_head=64, rotary_emb=temporal_rotary_emb, is_causal=True)`.

Reshape: `(B, T, H, W, D)` →fused-QKV→ `((B*H*W), heads=16, T, head_dim=64)`. Apply **1-D RoPE** via `rotary_emb.rotate_queries_or_keys(q, rotary_emb.freqs)`. Run `F.scaled_dot_product_attention(q, k, v, is_causal=True)`. Reshape back. Output projection.

The causal mask is the autoregressive backbone: at training time the model sees the whole window with independent noise levels per frame (Diffusion Forcing); at inference the future is masked off so frames are generated strictly left-to-right.

### 4. Action conditioning — Minecraft VPT action space

From `utils.py`. The 25-dim action vector:

```python
ACTION_KEYS = [
    "inventory", "ESC",                                         # 2 modal
    "hotbar.1", "hotbar.2", "hotbar.3", "hotbar.4", "hotbar.5", # 9 hotbar slots
    "hotbar.6", "hotbar.7", "hotbar.8", "hotbar.9",
    "forward", "back", "left", "right",                         # 4 WASD
    "cameraX", "cameraY",                                       # 2 continuous mouse
    "jump", "sneak", "sprint", "swapHands",                     # 4 mod keys
    "attack", "use", "pickItem", "drop",                        # 4 mouse / interaction
]
# len(ACTION_KEYS) == 25 — matches DiT external_cond_dim=25.
```

Encoding: all binary keys are 0.0 or 1.0 floats (asserted `0 ≤ value ≤ 1`). The two camera floats come in as raw VPT pixel deltas and are renormalised:

```python
max_val = 20
bin_size = 0.5
num_buckets = int(max_val / bin_size)   # = 40
value = (raw_camera_delta - num_buckets) / num_buckets   # NB: assumes raw is in [0, 80]
# clamp asserts -1 - 1e-3 <= value <= 1 + 1e-3
```

This normalisation is **inherited from OpenAI's VPT camera bucketisation**. HartsyInference users typically won't have raw VPT camera streams — for ad-hoc / programmatic input we should expose a high-level helper that takes `(deltaX, deltaY) ∈ [-1, 1]` floats and bypasses the bucket math. Document the formula but provide the float-direct path as the primary API.

Action streams are normally loaded from `.one_hot_actions.pt` (already a `(T, 25)` float tensor) or `.actions.pt` (a list of per-frame dicts). The loader **prepends an all-zero action** so action index `i` corresponds to "input that produced frame `i`" (so the first prompt frame has no preceding input).

### 5. ViT-VAE (`vit-l-20-shallow-encoder`, `vae.py`)

Continuous Gaussian KL autoencoder, **not VQ**. Architecture mirrors a vanilla ViT autoencoder.

`AutoencoderKL` parameters (as instantiated by `ViT_L_20_Shallow_Encoder`):

| Param | Value |
|---|---|
| `latent_dim` | **16** |
| `patch_size` | **20** (square) |
| `input_height` | **360** |
| `input_width` | **640** |
| `enc_dim` | **1024** |
| `enc_depth` | **6** (this is the "shallow" part — encoder is half as deep as decoder) |
| `enc_heads` | **16** (head_dim = 64) |
| `dec_dim` | **1024** |
| `dec_depth` | **12** |
| `dec_heads` | **16** |
| `mlp_ratio` | **4.0** |
| `norm_layer` | `LayerNorm(eps=1e-6)` |
| `use_variational` | **True** |
| `seq_h` = `input_height / patch_size` | **18** |
| `seq_w` = `input_width / patch_size` | **32** |
| `seq_len` = `seq_h * seq_w` | **576** |
| `patch_dim` = `3 * patch_size**2` | **1200** |

Computed sizes:

- Patch embed: `Conv2d(3 → 1024, kernel=20, stride=20)`, output `(B, 1024, 18, 32)`, flattened to `(B, 576, 1024)`.
- Encoder: 6 × `AttentionBlock(dim=1024, heads=16, frame_h=18, frame_w=32, mlp_ratio=4.0, qkv_bias=True)`.
- Encoder norm: `LayerNorm(1024)`.
- Bottleneck: `quant_conv = nn.Linear(1024, 2 * 16) = Linear(1024, 32)` → split into `(mean, logvar)` along last axis. Diagonal Gaussian. `logvar` clamped to `[-30, 20]`.
- At inference time `generate.py` uses `posterior.mean` (i.e. **deterministic** encode), not a sample.
- Decoder input: `post_quant_conv = nn.Linear(16, 1024)`.
- Decoder: 12 × `AttentionBlock(...)`.
- Decoder norm: `LayerNorm(1024)`.
- Predictor: `nn.Linear(1024, 1200)` (predicts per-patch RGB pixels).
- Unpatchify: `(B, 576, 1200)` → `(B, 3, 360, 640)`.

#### 5.1 `AttentionBlock` (inside VAE)

```python
norm1 → Attention → +residual
norm2 → Mlp(in=1024, hidden=4096, act=GELU) → +residual
```

`Attention` is **NOT** the same as `SpatialAxialAttention` in `dit.py` — it is a simpler per-frame self-attention with **2-D axial RoPE** (`RotaryEmbedding(dim=64 // 4 = 16, freqs_for="pixel", max_freq=18*32=576).get_axial_freqs(18, 32)`) baked in as a **non-persistent buffer** `rotary_freqs`. Note the divisor: `head_dim // 4 = 64 // 4 = 16` (vs. the DiT spatial attention which uses `head_dim // 2 = 32`). `qkv_bias=True`.

#### 5.2 Latent scaling factor

`generate.py` uses `scaling_factor = 0.07843137255` (= 20 / 255 = `bin_size / 255` reusing the VPT camera constant; coincidence, but exactly the value in the source). Encode: `latent = vae.encode(rgb * 2 - 1).mean * scaling_factor`. Decode: `rgb = (vae.decode(latent / scaling_factor) + 1) / 2`. This is a **scalar, per-tensor** factor — there are no per-channel mean/std vectors like Wan/SDXL.

### 6. **Continuous vs discrete tokenizer — important architectural call-out**

**Oasis uses a continuous Gaussian ViT-VAE (`AutoencoderKL`), NOT a discrete VQ-VAE or VQ-GAN.** The `vae.py` source confirms this explicitly: `DiagonalGaussianDistribution`, `quant_conv` produces `(mean, logvar)`, and the DiT operates on continuous 16-channel float latents.

This is good news for HartsyInference: **no new VQ codebook / nearest-neighbour-quantize / codebook-EMA module is required for Oasis itself.** The continuous VAE plumbing already exists (we ship SDXL VAE, Flux VAE, AudioLDM2 VAE, etc.). The only net-new piece in the VAE side of the port is that this is a **pure Transformer VAE** — patch-embed → ViT blocks → linear bottleneck → ViT blocks → linear predictor → unpatchify — with no conv encoder/decoder, no GroupNorm, no ResBlocks, no upsample/downsample 2×2 nearest stages. This is a strictly **simpler** family than SDXL's VAE.

**However** — and this is the place to flag it for the doc index — *future* world models very likely WILL use discrete tokenizers (e.g. Magvit-v2, FSQ, LFQ, RQ-VAE). Examples on the horizon:

- **MagViT-v2 / Lookup-Free Quantization (LFQ)** — used by VideoPoet, Genie, Genie-2, MarioGPT-style world models.
- **MineWorld** ([arXiv 2504.08388](https://arxiv.org/html/2504.08388v1)) — Minecraft world model, explicitly uses a discrete tokenizer.
- **Solaris** ([project page](https://solaris-wm.github.io/)) — multiplayer Minecraft, discrete.
- Possibly **Oasis-7B production** (no public details, but the press release mentions transformer ASIC throughput optimisations that favour discrete tokens).

When the **second** world model lands in HartsyInference, **that** is where the new VQ-codebook module class belongs:

```
src/HartsyInference.World/Models/Tokenizers/
    ContinuousVitVae.cs           ← shared base (Oasis lives here)
    IVideoTokenizer.cs            ← interface: Encode(RGB) → latent, Decode(latent) → RGB
    Magvit2Tokenizer.cs           ← first discrete impl (future)
    FsqTokenizer.cs / LfqTokenizer.cs   ← codebook-free quantizers
```

The Oasis port should define `IVideoTokenizer` so the discrete-tokenizer abstraction is **anticipated** even though the first concrete impl is continuous. The interface should expose latent shape `(B, T, C, H, W)` for continuous and `(B, T, H, W)` int indices for discrete; the pipeline asks for one of two flavours.

### 7. Schedule and sampler

#### 7.1 `sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5)`

From [Chen 2022, "On the importance of noise scheduling for diffusion models"](https://arxiv.org/abs/2212.11972), Figure 8 — used for images > 64×64.

```python
T = timesteps              # = 1000
steps = T + 1
t = linspace(0, T, steps, float64) / T            # [0,1]
v_start = sigmoid(start / tau)                    # sigmoid(-3) ≈ 0.04743
v_end   = sigmoid(end   / tau)                    # sigmoid( 3) ≈ 0.95257
alphas_cumprod = (-sigmoid((t * (end - start) + start) / tau) + v_end) / (v_end - v_start)
alphas_cumprod = alphas_cumprod / alphas_cumprod[0]                # normalize to 1 at t=0
betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
return clamp(betas, 0, 0.999)                                       # length T = 1000
```

This produces a sigmoid-shaped β schedule between effectively 0 and 0.999, smoother than the cosine schedule near `t=0`. HartsyInference should reproduce this exactly; no off-the-shelf scheduler we ship today returns it.

#### 7.2 DDIM step indexing

```python
ddim_noise_steps = 10
noise_range = linspace(-1, max_noise_level - 1, ddim_noise_steps + 1)  # 11 values, [-1, 999]
# noise_range = [-1, 99, 199, 299, 399, 499, 599, 699, 799, 899, 999]
```

The loop iterates `noise_idx ∈ {10, 9, ..., 1}` (skipping 0) — so `t = noise_range[10] = 999` down to `t = noise_range[1] = 99`, and `t_next = noise_range[0] = -1` on the last step (handled by `where(t_next < 0, t, t_next)`).

#### 7.3 v-parameterisation DDIM update

```
x_start = sqrt(ᾱ_t) · x_t       − sqrt(1 − ᾱ_t) · v
x_noise = (sqrt(1/ᾱ_t) · x_t − x_start) / sqrt(1/ᾱ_t − 1)
x_pred  = sqrt(ᾱ_{t_next}) · x_start + sqrt(1 − ᾱ_{t_next}) · x_noise
```

On the last step (`noise_idx == 1`) the **target frame** also gets `α_next = 1`, i.e. it lands at the clean latent. Context frames always have `α_next = 1` (they don't change inside the loop).

#### 7.4 Diffusion-Forcing inference (the only novel sampler trick)

The model is trained with [Diffusion Forcing](https://arxiv.org/abs/2407.01392) — at training, each frame in a window gets an **independent** uniform-random noise level. At inference, Oasis exploits this to use **two** noise levels simultaneously:

- All committed context frames: `t = stabilization_level - 1 = 14` (very low noise, ≈ 0.015 of max).
- Target frame being denoised: `t = noise_range[noise_idx]` (full range).

This works because the temporal-attention cross-talk between the noisy target token and the near-clean context tokens lets the network "lean on" the context's clean signal as a conditioning prior. With `stabilization_level = 0` (full clean context) Oasis is reported to drift in 20-30 frames; with `15` it stays stable for hundreds.

### 8. Performance & deployment

- **Per-frame latency:** 47 ms on a single H100 (10 DDIM × 4.7 ms DiT-S/2 forward at the 144-token-per-frame, sliding-window ≤ 32-frame size). Quoted by Decart.
- **Frame rate:** 20 FPS at 360 × 640 on H100. Sohu projected: 4K real-time.
- **Live demo size:** the production demo is a separate, larger model (referenced in press as "7B"). The released 500m is positioned as "code release / community use", not the demo model.
- **Quality on the 500M release:** "nightmarish hallucination" per TechSpot — short coherent clips, persistent drift over long horizons. This is **fine for our purposes**: Phase 10's job is correctness vs. the Python reference, not benchmark scores.

### 9. Reference implementations referenced inside `etched-ai/open-oasis`

The codebase calls out three lineage sources in docstrings:

1. **DiT** ([facebookresearch/DiT](https://github.com/facebookresearch/DiT/blob/main/models.py)) — the base spatial DiT block with adaLN-zero modulation, sinusoidal-MLP timestep embedding, `FinalLayer` shape. HartsyInference's existing `DiTBlock` family already implements this exactly.
2. **Diffusion Forcing** ([buoyancy99/diffusion-forcing](https://github.com/buoyancy99/diffusion-forcing/blob/main/algorithms/diffusion_forcing/models/)) — the per-frame noise level scheme, the temporal-axial attention, the spatial-axial attention. The `attention.py` and the inference loop are direct ports.
3. **Latte** ([Vchitect/Latte](https://github.com/Vchitect/Latte/blob/main/models/latte.py)) — the spatio-temporal interleaving pattern (alternate spatial and temporal blocks in each layer).

The VAE references **VQGAN** and **MAE** as lineage but the final implementation is a vanilla `AutoencoderKL` — no quantization, no patch-masking. Those references are aspirational, not behavioural.

## Key Numbers / Constants

| Constant | Value | Where used |
|---|---|---|
| Image resolution | **360 × 640** | VAE input/output |
| VAE patch size | **20** | square |
| VAE latent grid | **18 × 32 = 576 tokens** | per frame |
| VAE latent channels | **16** | `latent_dim` |
| VAE encoder dim | **1024** | `enc_dim` |
| VAE encoder depth | **6** | `enc_depth` (shallow) |
| VAE encoder heads | **16** | head_dim = 64 |
| VAE decoder dim | **1024** | `dec_dim` |
| VAE decoder depth | **12** | `dec_depth` |
| VAE decoder heads | **16** | head_dim = 64 |
| VAE MLP ratio | **4.0** | both halves |
| VAE LayerNorm eps | **1e-6** | both halves |
| VAE `use_variational` | **True** | Gaussian KL |
| VAE `logvar` clamp | **[-30, 20]** | bottleneck |
| VAE `qkv_bias` | **True** | inside `AttentionBlock` |
| VAE inner-attn rotary `dim` | **16** | = head_dim / 4 (NOT /2) |
| VAE inner-attn rotary `max_freq` | **576** | = H × W |
| VAE patch_dim | **1200** | = 3 × 20² |
| Latent scaling factor | **0.07843137255** | `generate.py` line ≈ 49 |
| DiT input H | **18** | matches VAE grid |
| DiT input W | **32** | matches VAE grid |
| DiT in_channels | **16** | matches VAE latent_dim |
| DiT patch size | **2** | latent-internal patchify → 9 × 16 = **144 tokens/frame** |
| DiT hidden_size | **1024** | model dim |
| DiT depth | **16** | layers |
| DiT num_heads | **16** | head_dim = **64** |
| DiT mlp_ratio | **4.0** | MLP hidden = 4096 |
| DiT timestep freq dim | **256** | `frequency_embedding_size` |
| DiT external_cond_dim | **25** | action vector dim |
| DiT max_frames | **32** | sliding window |
| Spatial RoPE dim | **32** | = head_dim / 2 |
| Spatial RoPE `freqs_for` | `"pixel"` | 2-D axial |
| Spatial RoPE max_freq | **256** | — |
| Temporal RoPE dim | **64** | = head_dim |
| Temporal attention causal | **True** | left-to-right time |
| Spatial attention causal | **False** | bidirectional in space |
| AdaLN modulation per block | `Linear(1024, 6 × 1024) × 2` | spatial + temporal halves |
| MLP activation | `GELU(approximate="tanh")` | DiT MLPs |
| LayerNorm eps (DiT) | **1e-6** | `elementwise_affine=False` |
| Action keys count | **25** | exact list above |
| Action camera normalization | `max_val=20, bin_size=0.5` | VPT-style |
| Action prepended zero frame | **1** | aligns action[i] ↔ frame[i] |
| Diffusion `T` (`max_noise_level`) | **1000** | sigmoid β schedule |
| β schedule `start, end, tau` | **-3, 3, 1** | sigmoid |
| DDIM steps | **10** | `--ddim-steps` |
| Initial-noise clamp | **±20** | `noise_abs_max` |
| Stabilization level | **15** | context-frame noise idx |
| Default `num_frames` | **32** | matches `max_frames` |
| Default `n_prompt_frames` | **1** | single-image prompt |
| Output FPS | **20** | quoted real-time on H100 |
| Per-frame latency (H100) | **47 ms** | Decart blog |
| Per-iter training cost | **150 ms** | Decart blog |
| Storage (DiT) | **2.43 GB** | safetensors FP32 |
| Storage (VAE) | **917 MB** | safetensors FP32 |
| Storage (total FP16) | **≈ 1.7 GB** | full pipeline cast to FP16 |
| License | **MIT** | code and weights |

## Data Layouts / Formats

### 9.1 HF repo `Etched/oasis-500m` file tree

```
Etched/oasis-500m/
├── .gitattributes                  115 B
├── LICENSE                        1.07 kB    MIT
├── README.md                       893 B
├── oasis500m.safetensors          2.43 GB    DiT-S/2 backbone, FP32
├── oasis500m.pt                   2.43 GB    Pickle version of same
├── vit-l-20.safetensors            917 MB    ViT-VAE encoder+decoder, FP32
├── vit-l-20.pt                     917 MB    Pickle version of same
└── media/                                    Demo gifs and PNGs (not weights)
```

**No `config.json`**. **No tokenizer.** Action input is a raw `(T, 25)` float tensor — there is no vocabulary BPE.

### 9.2 Expected (not yet dumped) tensor key prefixes in `oasis500m.safetensors`

Inferred from the PyTorch class hierarchy. Confirm by running locally:

```python
from safetensors import safe_open
with safe_open("oasis500m.safetensors", framework="pt") as f:
    for k in sorted(f.keys()): print(k)
```

Expected keys:

```
x_embedder.proj.weight                    # (1024, 16, 2, 2)
x_embedder.proj.bias                      # (1024,)
t_embedder.mlp.0.weight                   # (1024, 256)
t_embedder.mlp.0.bias                     # (1024,)
t_embedder.mlp.2.weight                   # (1024, 1024)
t_embedder.mlp.2.bias                     # (1024,)
external_cond.weight                      # (1024, 25)
external_cond.bias                        # (1024,)
blocks.{0..15}.s_norm1.{}                 # none — affine=False
blocks.{0..15}.s_attn.to_qkv.weight       # (3072, 1024) — no bias
blocks.{0..15}.s_attn.to_out.weight       # (1024, 1024)
blocks.{0..15}.s_attn.to_out.bias         # (1024,)
blocks.{0..15}.s_norm2.{}                 # none — affine=False
blocks.{0..15}.s_mlp.fc1.weight           # (4096, 1024)
blocks.{0..15}.s_mlp.fc1.bias             # (4096,)
blocks.{0..15}.s_mlp.fc2.weight           # (1024, 4096)
blocks.{0..15}.s_mlp.fc2.bias             # (1024,)
blocks.{0..15}.s_adaLN_modulation.1.weight # (6144, 1024)
blocks.{0..15}.s_adaLN_modulation.1.bias   # (6144,)
blocks.{0..15}.t_norm1.{}                  # none
blocks.{0..15}.t_attn.to_qkv.weight        # (3072, 1024) — no bias
blocks.{0..15}.t_attn.to_out.weight        # (1024, 1024)
blocks.{0..15}.t_attn.to_out.bias          # (1024,)
blocks.{0..15}.t_norm2.{}                  # none
blocks.{0..15}.t_mlp.fc1.weight            # (4096, 1024)
blocks.{0..15}.t_mlp.fc1.bias              # (4096,)
blocks.{0..15}.t_mlp.fc2.weight            # (1024, 4096)
blocks.{0..15}.t_mlp.fc2.bias              # (1024,)
blocks.{0..15}.t_adaLN_modulation.1.weight # (6144, 1024)
blocks.{0..15}.t_adaLN_modulation.1.bias   # (6144,)
final_layer.linear.weight                  # (64, 1024)        — patch_size² × out_channels = 4 × 16
final_layer.linear.bias                    # (64,)
final_layer.adaLN_modulation.1.weight      # (2048, 1024)
final_layer.adaLN_modulation.1.bias        # (2048,)
```

Total parameter count check (approximate):

- PatchEmbed: 16 × 1024 × 4 + 1024 ≈ 66 K
- TimestepEmbedder: 256 × 1024 + 1024 × 1024 + biases ≈ 1.31 M
- external_cond: 25 × 1024 + 1024 ≈ 26 K
- Per block: 2 × (QKV: 1024×3072 + Out: 1024×1024+1024 + MLP: 1024×4096+4096×1024+biases + adaLN: 1024×6144+6144) ≈ 2 × 14.9 M ≈ 29.8 M
- 16 blocks × 29.8 M ≈ 477 M
- FinalLayer: 1024 × 2048 + 2048 + 1024 × 64 + 64 ≈ 2.17 M
- **Total ≈ 481 M params** ≈ "500M" branding. ✓

### 9.3 Expected tensor key prefixes in `vit-l-20.safetensors`

```
patch_embed.proj.weight              # (1024, 3, 20, 20)
patch_embed.proj.bias                # (1024,)
encoder.{0..5}.norm1.{weight,bias}   # (1024,) each
encoder.{0..5}.attn.qkv.weight       # (3072, 1024)
encoder.{0..5}.attn.qkv.bias         # (3072,)
encoder.{0..5}.attn.proj.weight      # (1024, 1024)
encoder.{0..5}.attn.proj.bias        # (1024,)
encoder.{0..5}.norm2.{weight,bias}   # (1024,)
encoder.{0..5}.mlp.fc1.weight        # (4096, 1024)
encoder.{0..5}.mlp.fc1.bias          # (4096,)
encoder.{0..5}.mlp.fc2.weight        # (1024, 4096)
encoder.{0..5}.mlp.fc2.bias          # (1024,)
enc_norm.{weight,bias}               # (1024,)
quant_conv.weight                    # (32, 1024)
quant_conv.bias                      # (32,)
post_quant_conv.weight               # (1024, 16)
post_quant_conv.bias                 # (1024,)
decoder.{0..11}.norm1.{}             # (1024,)
decoder.{0..11}.attn.qkv.weight      # (3072, 1024)
decoder.{0..11}.attn.qkv.bias        # (3072,)
decoder.{0..11}.attn.proj.{}         # (1024, 1024)
decoder.{0..11}.norm2.{}             # (1024,)
decoder.{0..11}.mlp.fc1.{weight,bias} # (4096, 1024)
decoder.{0..11}.mlp.fc2.{weight,bias} # (1024, 4096)
dec_norm.{weight,bias}               # (1024,)
predictor.weight                     # (1200, 1024)
predictor.bias                       # (1200,)
```

`rotary_freqs` buffers are non-persistent — not in the safetensors file; HartsyInference computes them at load time from `H=18, W=32, head_dim=64, max_freq=576`.

### 9.4 Action stream format

Two on-disk formats, both PyTorch `.pt`:

- `*.actions.pt` — `List[Dict[str, int | List[int, int]]]`, one dict per frame. Each dict has all 25 keys; binary keys are `0`/`1`, the `camera` key is `[int_raw_x, int_raw_y]` in VPT bucket space. `one_hot_actions()` converts to `(T, 25)` float.
- `*.one_hot_actions.pt` — `Tensor(T, 25, float32)` already encoded.

HartsyInference's `OasisPipeline` should accept either a `float[T, 25]` array (one-hot, already encoded) or an `OasisAction[]` struct array with explicit fields. A small `OasisActionEncoder` helper handles the camera normalisation.

## Algorithm Steps

```
# === Generate `total_frames` frames given a prompt image and an action stream ===

INPUTS:
  prompt_rgb         : (1, n_prompt_frames, 3, 360, 640)  float in [0,1]
  actions            : (1, total_frames, 25)              float
  n_prompt_frames    : int (default 1)
  total_frames       : int (default 32)
  ddim_steps         : int (default 10)

CONSTANTS:
  max_noise_level    = 1000
  noise_abs_max      = 20
  stabilization_level= 15
  scaling_factor     = 0.07843137255
  max_frames         = 32

# --- 1. VAE encode the prompt ---
x_pixels = prompt_rgb * 2 - 1                                       # to [-1, 1]
x_pixels = reshape(x_pixels, (B*T, 3, 360, 640))
posterior = vae.encode(x_pixels)
latents   = posterior.mean * scaling_factor                          # (B*T, 576, 16)
latents   = reshape(latents, (B, T, 16, 18, 32))                    # (B, n_prompt, 16, 18, 32)

# --- 2. Pre-compute schedule ---
betas         = sigmoid_beta_schedule(1000, start=-3, end=3, tau=1)  # length 1000
alphas_cumprod = cumprod(1 - betas)                                   # length 1000
noise_range   = linspace(-1, 999, ddim_steps + 1)                    # 11 values

# --- 3. Autoregressive frame loop ---
FOR i FROM n_prompt_frames TO total_frames-1:
    # 3a. Append fresh noise as the (i+1)-th frame
    chunk = clamp(randn((B, 1, 16, 18, 32)), -20, +20)
    latents = concat([latents, chunk], dim=time)

    start_frame = max(0, i + 1 - max_frames)

    # 3b. DDIM denoise the new frame only
    FOR noise_idx FROM ddim_steps DOWN TO 1:
        t_ctx  = full((B, i),    stabilization_level - 1)            # = 14
        t      = full((B, 1),    noise_range[noise_idx])
        t_next = full((B, 1),    noise_range[noise_idx - 1])
        IF t_next < 0: t_next = t                                    # last step guard
        t      = concat([t_ctx, t],      dim=time)
        t_next = concat([t_ctx, t_next], dim=time)

        # Sliding window
        x_in    = latents[:, start_frame:]
        t_in    = t[:, start_frame:]
        a_in    = actions[:, start_frame : i + 1]

        v = dit_forward(x_in, t_in, a_in)                            # v-parameterisation

        a_t       = alphas_cumprod[t_in]                             # (B, T', 1, 1, 1)
        x_start   = sqrt(a_t) * x_in - sqrt(1 - a_t) * v
        x_noise   = (sqrt(1 / a_t) * x_in - x_start) / sqrt(1 / a_t - 1)

        a_next                       = alphas_cumprod[t_next_in]
        a_next[:, :-1]               = 1.0                           # freeze context
        IF noise_idx == 1: a_next[:, -1:] = 1.0                      # final step → clean
        x_pred = sqrt(a_next) * x_start + sqrt(1 - a_next) * x_noise

        latents[:, -1:] = x_pred[:, -1:]                             # write only the target frame

# --- 4. VAE decode ---
latents_flat = reshape(latents, (B*total_frames, 576, 16))
rgb_norm     = vae.decode(latents_flat / scaling_factor)             # (B*T, 3, 360, 640)
rgb          = (rgb_norm + 1) / 2
rgb          = clamp(rgb, 0, 1) * 255 → uint8
write_video("video.mp4", rgb, fps=20)
```

## Reference Implementations

**Primary** — [`github.com/etched-ai/open-oasis`](https://github.com/etched-ai/open-oasis) (MIT, ~6 files):

- [`generate.py`](https://github.com/etched-ai/open-oasis/blob/master/generate.py) — full inference loop. Quoted verbatim in § 2 / Algorithm Steps.
- [`dit.py`](https://github.com/etched-ai/open-oasis/blob/master/dit.py) — `PatchEmbed`, `TimestepEmbedder`, `FinalLayer`, `SpatioTemporalDiTBlock`, `DiT`, `DiT_S_2()` factory, `DiT_models` registry. The **only** released variant is `DiT-S/2`.
- [`vae.py`](https://github.com/etched-ai/open-oasis/blob/master/vae.py) — `DiagonalGaussianDistribution`, `Attention`, `AttentionBlock`, `AutoencoderKL`, `ViT_L_20_Shallow_Encoder()` factory, `VAE_models` registry.
- [`attention.py`](https://github.com/etched-ai/open-oasis/blob/master/attention.py) — `SpatialAxialAttention`, `TemporalAxialAttention`.
- [`utils.py`](https://github.com/etched-ai/open-oasis/blob/master/utils.py) — `sigmoid_beta_schedule`, `ACTION_KEYS`, `one_hot_actions`, `load_prompt`, `load_actions`.
- [`rotary_embedding_torch.py`](https://github.com/etched-ai/open-oasis/blob/master/rotary_embedding_torch.py) — vendored copy of `lucidrains/rotary-embedding-torch`. Provides `RotaryEmbedding`, `apply_rotary_emb`, `.get_axial_freqs()`, `.rotate_queries_or_keys()`.

**Lineage** (referenced in docstrings):

- [Facebook DiT](https://github.com/facebookresearch/DiT) — adaLN-zero modulation + sinusoidal timestep + FinalLayer. HartsyInference's existing DiT blocks match.
- [Diffusion Forcing](https://github.com/buoyancy99/diffusion-forcing) ([arXiv 2407.01392](https://arxiv.org/abs/2407.01392)) — the per-frame noise scheme and the spatial/temporal axial attention. Note the [Diffusion Forcing blog post](https://www.boyuan.space/diffusion-forcing/) explains the training paradigm; Oasis is the production-scale demonstration of it.
- [Latte](https://github.com/Vchitect/Latte) — spatio-temporal block interleaving.
- [Vision Pre-Training (VPT)](https://github.com/openai/Video-Pre-Training) — source of the 25-key Minecraft action space + camera bucketisation.

**Companion materials** (not code but reference quality):

- [Oasis blog](https://oasis-model.github.io/) — official architecture description.
- [Decart "A Universe in a Transformer"](https://decart.ai/publications/oasis-interactive-ai-video-game-model) — design philosophy + Sohu / inference cost claims.
- [Etched "Oasis: an interactive, explorable world model"](https://www.etched.com/blog-posts/oasis) — hardware perspective.
- [Chipstrat: "Etched's Oasis: Creating a Market For Sohu"](https://www.chipstrat.com/p/etcheds-oasis-creating-a-market-for) — third-party analysis.
- Press: [InfoQ](https://www.infoq.com/news/2024/11/decart-etched-oasis/), [The Decoder](https://the-decoder.com/ai-generated-game-oasis-now-turns-images-into-playable-3d-worlds/), [TechSpot](https://www.techspot.com/news/105436-oasis-interactive-ai-experience-turns-minecraft-nightmarish-hallucination.html).

**Community ports** (read-only, not source-of-truth):

- [PFWorld/open-oasis-PFWorld](https://github.com/PFWorld/open-oasis-PFWorld) — minor fork.
- [profhiggs/open-oasis-datacraft](https://github.com/profhiggs/open-oasis-datacraft) — fork wrapping Oasis around a Minecraft-mod token economy. Useful for action-stream examples.
- [XmYx/open-oasis](https://github.com/XmYx/open-oasis) — fork with QoL tweaks.
- [spikedoanz/realtime-oasis](https://github.com/spikedoanz/realtime-oasis) — real-time interactive harness; useful precedent for our `OasisLivePipeline`.

## Differences Between Implementations

There is one official reference (`etched-ai/open-oasis`) and several near-identical community forks. No serious behavioural divergence has been observed; the forks differ in:

- Real-time interactive harness vs. batch generate-and-save (spikedoanz/realtime-oasis).
- Minor tokenizer / action-loading conveniences (datacraft, PFWorld).
- Some forks add per-action sliders / UI; the model weights are untouched.

There is **no diffusers PR** for Oasis as of 2026-05-24. There is **no ComfyUI node** in the official repo. The architecture is simple enough that it has been re-implemented inside several game-engine harnesses (Unity, web demos) but always pointing back at the same safetensors.

There is also a related architectural lineage to be aware of, but they are **different models** and should NOT be confused with Oasis:

- [MineWorld (Microsoft, 2025-04)](https://arxiv.org/html/2504.08388v1) — also Minecraft, but uses a **discrete tokenizer** and a strict next-token AR transformer. Open-source.
- [Solaris (2025)](https://solaris-wm.github.io/) — multiplayer Minecraft world model, discrete.
- [DIAMOND (Alonso et al., NeurIPS 2024)](https://diamond-wm.github.io/) — Atari/CSGO, continuous, smaller scope.

These are the right precedents to study when HartsyInference's *second* world model arrives — but they are out of scope for the Oasis port itself.

## Open Questions

1. **Exact safetensors tensor key names.** The keys in § 9.2 and § 9.3 are inferred from the Python class hierarchy. Must be confirmed locally by dumping `oasis500m.safetensors` and `vit-l-20.safetensors` with the safetensors metadata reader. Particular concerns:
   - Whether `timm`'s `Mlp` uses `fc1`/`fc2` vs `0`/`2` naming in this version (should be `fc1`/`fc2`, but verify).
   - Whether the `to_qkv` linear is stored as a single fused tensor or pre-split (almost certainly fused).
   - Whether the `adaLN_modulation` `Sequential` is keyed as `.0`/`.1` (SiLU/Linear) — meaning weights live at `*.adaLN_modulation.1.weight`. We use the same convention in our existing DiT ports.
2. **Whether the FP32 weights are losslessly castable to FP16.** Likely yes (DiT activations regularly include values in ±5 range, well within FP16). Verify by per-tensor `abs.max()` check.
3. **Whether the safetensors carries any non-state-dict metadata** (e.g. embedded scaling factor). Probably not — `generate.py` hard-codes 0.07843137255 — but check the safetensors header anyway.
4. **Stability of the `stabilization_level = 15` choice** under different frame counts. The current value is hard-coded; for very long autoregressive runs (≥ 200 frames) it may need tuning. Out of scope for the initial port; document as a tunable.
5. **Action encoding for users who don't have raw VPT camera deltas.** Our public API should accept normalised camera floats in `[-1, 1]` directly. The "value = (raw - 40) / 40" mapping only matters for replaying VPT logs.
6. **Behaviour on prompt sizes other than 360 × 640.** The VAE's `Attention` registers `rotary_freqs` as a non-persistent buffer derived from `frame_height=18, frame_width=32` (hard-coded). Other resolutions will break unless you re-initialise. For the HartsyInference port: **enforce 360 × 640** at API level (the model card doesn't claim anything else works).
7. **Latent shape after VAE encode.** The code shows `vae.encode(x).mean` returning `(B, 576, 16)` — i.e. *token-major*, not channel-major. Then rearranged to `(B, T, C, H, W)` via `"(b t) (h w) c -> b t c h w"`. HartsyInference's tensor reshape utilities will need to follow this convention exactly when round-tripping.
8. **Whether the same `vit-l-20.safetensors` checkpoint can serve as a standalone image VAE** for other HartsyInference modalities (e.g. compress arbitrary 360 × 640 images for free). Likely yes — the VAE is task-agnostic — but its 16-channel latent space is **not** the same as SDXL's, so it's not interoperable.
9. **The "Oasis 7B" production model.** Whether Etched/Decart will release it under MIT later. If so, the same HartsyInference port should accept it as a config override (`hidden_size`, `depth`, possibly a discrete tokenizer).
10. **Real-time interactive surface in HartsyInference.** Phase 10 will need a streaming API: `OasisLivePipeline.PushAction(OasisAction)` → `Tensor frame` per call, with internal sliding-window state. This is non-trivial — the DDIM loop assumes batch generation; for live use we want pre-baked per-step kernels and a persistent latent buffer.
11. **KV-cache feasibility.** The temporal-axial attention is causal in time; in principle KV-cache could speed up the loop by ~16× (one new frame's Q against accumulated K/V). But the noise level on context frames changes between calls (well — actually it doesn't; `stabilization_level=15` is constant), so KV-cache may work. Worth investigating in the second pass for real-time mode.

## Implementation Notes

### How this maps to HartsyInference packages

A brand-new package, **`HartsyInference.World`**, is appropriate for world models (this is Phase 10 in `BUILD_ORDER.md`). Inside:

```
src/HartsyInference.World/
├── HartsyInference.World.csproj
├── Models/
│   ├── Tokenizers/
│   │   ├── IVideoTokenizer.cs                   # new abstraction (see § 6)
│   │   └── OasisVitVae.cs                        # the ViT-VAE class
│   ├── Denoisers/
│   │   ├── OasisDitConfig.cs
│   │   ├── OasisDit.cs                          # DiT-S/2 backbone
│   │   └── DitBlocks/
│   │       ├── SpatioTemporalDitBlock.cs
│   │       ├── SpatialAxialAttention.cs
│   │       └── TemporalAxialAttention.cs
│   └── Actions/
│       ├── OasisAction.cs                       # struct with 23 bools + 2 floats
│       ├── OasisActionEncoder.cs                # → float[25]
│       └── OasisActionKeys.cs                   # the constant list
├── Pipelines/
│   ├── OasisPipeline.cs                         # batch generate-N-frames API
│   └── OasisLivePipeline.cs                     # streaming real-time API (Phase 10 stretch)
├── Schedulers/
│   └── SigmoidBetaSchedule.cs                   # net-new — not in DIFFUSION_SCHEDULERS yet
└── Utilities/
    └── DiffusionForcingDdim.cs                  # v-param DDIM with per-frame noise levels
```

In **`HartsyInference.ModelAssets`**:

```
CheckpointConverters/
└── OasisCheckpointConverter.cs                  # loads oasis500m + vit-l-20 safetensors
```

### Net-new primitives required

Compared to existing HartsyInference infrastructure, Oasis introduces:

1. **`SigmoidBetaSchedule`** — not currently shipped. Easy CPU-side pre-compute (1000 floats); upload once.
2. **`DiffusionForcingDdim`** — extension of standard DDIM with **per-frame independent noise levels** in the timestep tensor (shape `(B, T)` rather than `(B,)`). The DiT forward already accepts this shape; the scheduler just needs to gather `alphas_cumprod[t]` with a per-frame index.
3. **Per-frame v-parameterisation update.** The math in § 7.3 is straight DDIM v-param; the only twist is the "freeze context frames" and "final-step force-clean" mask. Implement as a small vectorised helper.
4. **2-D axial RoPE.** The DiT's spatial attention applies RoPE separately to the H and W axes, sharing the head_dim into two halves of 32 each (`get_axial_freqs(H, W)` returns a tensor of shape `(H, W, 32)` and `apply_rotary_emb` broadcasts it correctly). HartsyInference's existing 1-D RoPE needs an axial variant. Reusable also for future video pipelines.
5. **No new attention kernels.** Both spatial and temporal use stock `F.scaled_dot_product_attention`. Map to our existing `IBackend.ScaledDotProductAttention` (with `is_causal` flag).
6. **No new conv kernels.** The VAE has only one `Conv2d` (the patch embed) — and stride == kernel, so it's a no-overlap patchify, well-served by the existing `Conv2D` path or even by a simple `(unfold + matmul)` substitute.
7. **`IVideoTokenizer` interface.** New abstraction. Continuous variant (Oasis) returns `Tensor latent` of shape `(B, T, C, H, W)`; future discrete variant returns `Tensor<int> indices` of shape `(B, T, H, W)`. The pipeline switches behaviour based on `tokenizer.IsDiscrete`. Even though Oasis is continuous, **define this abstraction now** so MagViT-v2 / MineWorld-style ports later don't force a refactor.
8. **Action-encoder utility.** Small CPU helper that maps `OasisAction` (struct) to `float[25]`. The VPT camera renormalisation lives here.

### What we already have

- `LayerNorm(eps=1e-6, elementwise_affine=False)` — used by DiT-style models everywhere.
- adaLN-zero modulation (`shift, scale, gate` from `SiLU(Linear(c))`) — `AdaLNModulation` already implemented in `DiTUtils`.
- Sinusoidal timestep embedding + 2-layer MLP — `SinusoidalTimestepEmbedding` in `DiTUtils`.
- `Mlp(in, hidden, act=GELU(tanh))` — covered by existing `GeluTanhMlp`.
- `nn.Linear` with no bias on `to_qkv`, with bias on `to_out` — standard.
- Continuous Gaussian VAE plumbing — `DiagonalGaussianDistribution` analogue exists; only the patch-embed / patch-predict heads are new shapes.
- DDIM v-param scheduler — partial; the per-frame index gather + freeze-context mask need to be added.

### Test-skipping discipline

> **Superseded 2026-08-06.** The per-model pipeline/generation tests this section specified were
> removed in the test-suite cleanup, and the rule is now the opposite: **do not add a test that
> proves a model works end to end** — a model that stops working is visible the moment anyone uses
> it. Test what breaks quietly instead (kernel numerics, cross-device equivalence, quantization and
> codec round-trips, padding/tiling geometry, format and key mapping), and put shared-component
> parity in `tests/<Project>/Parity/` with a `*ParityTests` name. See `docs/CODE_STYLE.md` §Testing.

### Layered diff harness for porting

Reuse the SD3.5 / Z-Image-style "dump each layer's input + output to JSON" harness:

1. **VAE encode** — feed sample image, dump `posterior.mean`. Compare C# against Python before touching the DiT.
2. **DiT layer 0 spatial attention** — feed a synthetic latent + zero action, dump output of `s_attn`. Iterate per layer.
3. **DiT layer 0 temporal attention** — same.
4. **DDIM step 0** — feed clean prompt latent + 1 noise frame, run 1 model forward, dump `v`, `x_start`, `x_pred`. Compare scalar-wise.
5. **Full one-step generation** — 1 DDIM step, dump final latent.
6. **Full 10-step single-frame** — full generation of frame index 1.
7. **VAE decode** — round-trip latent → pixels.
8. **Full 32-frame video** — byte-exact MP4.

### VRAM / viability per target GPU

| GPU | VRAM | Oasis-500m (FP16) |
|---|---|---|
| iGPU / 4 GB | 4 GB | Comfortable. Model ≈ 0.7 GB; activations for the default 32-frame window ≈ 1.5 GB. |
| 8 GB consumer | 8 GB | Trivial. Real-time at 20 FPS achievable with the streaming pipeline. |
| 12+ GB | — | Trivial. Headroom for batched generation. |

This is the **smallest production-quality DiT** in HartsyInference. It is the right model for: smoke-testing the world-model pipeline, validating new attention backends (since both spatial and temporal attention are exercised), demonstrating the C# real-time interactive surface, and onboarding new contributors.

### Ordering / dependencies for the build

1. **Land `OasisVitVae`** first — pure ViT, no novel primitives, validates `IVideoTokenizer` interface design.
2. **Land `SpatialAxialAttention` + `TemporalAxialAttention`** using existing `ScaledDotProductAttention` + new 2-D axial RoPE.
3. **Land `SpatioTemporalDitBlock` + `OasisDit`** — the 16-layer backbone.
4. **Land `SigmoidBetaSchedule` + `DiffusionForcingDdim`** — schedule + per-frame v-param DDIM.
5. **Land `OasisPipeline`** — batch generate-N-frames API.
6. **Validate end-to-end against Python reference (byte-exact at FP16).**
7. **Stretch (Phase 10b): `OasisLivePipeline`** — streaming surface for live demos (Unity, web). May involve KV-cache and / or pre-baked per-step kernel pipelines.

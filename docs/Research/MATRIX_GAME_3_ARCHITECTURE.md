# Matrix-Game 3.0 — Research Notes

> Status: Complete (model card + arXiv v2 paper + GitHub source code + Wan2.2 base config captured; only safetensors tensor-key dump remains as a local follow-up) | Last Updated: 2026-05-24 | Needed Before: HartsyInference.World (Matrix-Game 3.0 pipeline, Phase 10)
> Source of truth: [HF `Skywork/Matrix-Game-3.0`](https://huggingface.co/Skywork/Matrix-Game-3.0), [GitHub `SkyworkAI/Matrix-Game`](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-3), [arXiv 2604.08995 v2](https://arxiv.org/abs/2604.08995v2), [project page](https://matrix-game-v3.github.io/), [base `Wan-AI/Wan2.2-TI2V-5B`](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)
> License: Apache-2.0 (Matrix-Game 3.0 code + weights), Apache-2.0 (Wan2.2-TI2V-5B base), Apache-2.0 (UMT5-XXL encoder). No model-card-imposed use restrictions beyond standard Apache-2.0 terms.
> Related: [`LANCE_ARCHITECTURE.md`](LANCE_ARCHITECTURE.md) (Wan2.2 3D causal VAE — exact same `Wan2.2_VAE.pth`, same 48-channel latent, same mean/std), [`TEXT_ENCODERS.md`](TEXT_ENCODERS.md) (UMT5-XXL is also used by AuraFlow / Pile-T5-XL), [`FLOW_MATCHING_AUDIO.md`](FLOW_MATCHING_AUDIO.md) (rectified-flow background; Matrix-Game uses FlowUniPC).

## Summary

Matrix-Game 3.0 (Skywork AI Matrix-Game Team, arXiv:2604.08995, 2026-03 initial release) is a **memory-augmented interactive world model** that performs real-time streaming video generation at **720p (704×1280) @ 40 FPS** on multi-GPU A/H-series hardware. It is a finetune of **Wan2.2-TI2V-5B** with two structural additions: an **ActionModule** (keyboard via cross-attention, mouse via self-attention, attached to a subset of DiT blocks) and a **camera-aware long-horizon memory mechanism** that retrieves 5 past latent frames by camera-frustum overlap and injects them through the same joint-attention space as the noisy prediction. The underlying DiT is the **Wan2.2 5B dense** transformer (40 layers, dim 5120, 40 heads, head_dim 128, FFN 13824, RMSNorm with QK-norm and cross-attn-norm), patchified `(1,2,2)` over 48-channel VAE latents, with a 4× temporal / 16×16 spatial Wan2.2 3D causal VAE. The text branch is **UMT5-XXL** with 512-token context.

Two checkpoints ship under one HuggingFace repo: a **base model** (12.9 GB safetensors, 50-step FlowUniPC inference, sample_shift=5.0, CFG=5.0) and a **base_distilled_model** (25.9 GB safetensors — larger because it bundles student + critic / EMA — runs at **3 inference steps** via multi-segment Distribution Matching Distillation). Inference is autoregressive over **segments of latent length 15** (57 RGB frames for the first segment, 40 for every subsequent segment); the next segment conditions on the last 4 past latent frames plus 5 retrieved memory frames plus the new noisy prediction. A separate **MG-LightVAE** decoder (a pruned distillation of the Wan2.2 VAE decoder; 50 % or 75 % pruning shipping as `MG-LightVAE.pth` 2.74 GB and `MG-LightVAE_v2.pth` 841 MB) replaces the Wan2.2 decoder at inference time for a 2.6× / 5.2× decode speedup. There is also a paper-only **2×14B MoE** variant ("Coming Soon") that splits high-noise denoising between a first-person expert and a third-person expert; the 5B is what's actually downloadable today.

For HartsyInference this is a Phase-10 `HartsyInference.World` pipeline that reuses substantially all of the work needed for a Wan2.2 video pipeline (DiT backbone, 3D causal VAE, UMT5-XXL encoder, FlowUniPC scheduler) and adds three new pieces: (1) the **ActionModule** block (a small ~16-head dual-attention block with its own RoPE θ=256), (2) a **camera-pose + Plücker-embedding** preprocessor, and (3) a streaming **per-segment loop** that maintains the past-frame buffer, the 5-slot memory cache, and routes decoding to an async worker (or to a single in-process VAE call on a smaller install).

## Detailed Findings

### 1. Family / variants

| Variant | Params | Status | File on HF | Notes |
|---|---|---|---|---|
| `base_model` | 5 B | Released | `base_model/diffusion_pytorch_model.safetensors` (12.9 GB) | 50-step FlowUniPC sampler, CFG=5.0, sample_shift=5.0. |
| `base_distilled_model` | 5 B (active) + critic | Released | `base_distilled_model/diffusion_pytorch_model.safetensors` (25.9 GB) | DMD distilled. **3 inference steps** at runtime. Larger file because it contains both student and supporting weights. |
| **2×14B MoE** | 28 B total | "Coming Soon" (paper only) | n/a | Two 14B "high-noise" experts: first-person view + third-person view. Same DiT shape as 5B with `dim` and `num_layers` scaled. **Not downloadable as of 2026-05.** |

A separate **MG-LightVAE** (Matrix-Game LightVAE) ships as two distilled decoder variants:

| Variant | File | Size | Pruning | Speedup vs Wan2.2 VAE decode |
|---|---|---|---|---|
| Wan2.2 VAE (full) | `Wan2.2_VAE.pth` | 2.82 GB | 0 % | 1.0× (PSNR 33.79, SSIM 0.99, 0.76 s decode) |
| MG-LightVAE | `MG-LightVAE.pth` | 2.74 GB | **0.5** (50 %) | **2.6×** (PSNR 31.84, SSIM 0.99, 0.30 s decode) |
| MG-LightVAE v2 | `MG-LightVAE_v2.pth` | 841 MB | **0.75** (75 %) | **5.2×** (PSNR 31.14, SSIM 0.99, 0.13 s decode) |

All three share the Wan2.2 encoder; only the **decoder** is pruned (channel widths multiplied by `1.0 − pruning_rate`). The encoder is used only at session start to encode the input image; decoders run every segment.

### 2. Backbone: Wan2.2-TI2V-5B DiT, finetuned

From `Matrix-Game-3/wan/configs/config.py` (the inference-time loaded config; identical shape to `Wan-AI/Wan2.2-TI2V-5B/config.json`):

| Field | Value |
|---|---|
| `dim` (hidden size) | **5120** |
| `num_layers` | **40** |
| `num_heads` | **40** (head_dim = **128**) |
| `ffn_dim` | **13,824** |
| `in_dim` (latent channels in) | **48** |
| `out_dim` (latent channels out) | **48** |
| `freq_dim` (sinusoidal timestep dim) | **256** |
| `patch_size` (T, H, W) | **(1, 2, 2)** |
| `text_len` (max UMT5 tokens) | **512** |
| `text_dim` (UMT5 hidden) | **4096** |
| `window_size` | **(-1, -1)** (full attention; no sliding window) |
| `qk_norm` | **True** (RMSNorm on Q and K per head) |
| `cross_attn_norm` | **True** (RMSNorm on cross-attention input) |
| `eps` (RMSNorm) | **1e-6** |
| `model_type` | **`ti2v`** (text + image → video) |
| dtype | **bfloat16** |
| `use_memory` | **True** (Matrix-Game adds this) |
| `sigma_θ` (RoPE θ perturbation) | **0.0** at base, **0.8** during memory-aug training |
| `use_text_crossattn` | **True** |
| `_diffusers_version` (Wan2.2 base) | 0.33.0 |

Note that **Wan-AI's `config.json`** lists slightly different defaults (`dim=3072, ffn_dim=14336, num_heads=24, num_layers=30`). That is the *diffusers* config for the diffusers port (a smaller distilled variant). The **inference-time** shape used by Matrix-Game's `generate.py` is the larger 5120 / 40 / 40 / 13824 spec above — confirmed by the `Matrix-Game-3/wan/configs/config.py` constants and by the 12.9 GB safetensors size of `base_model/`.

**Block structure** (`Matrix-Game-3/wan/modules/model.py:WanModel`):

```
patch_embedding         = Conv3d(in_dim=48 → dim=5120, kernel=(1,2,2), stride=(1,2,2))
time_embedding          = MLP( sinusoidal(t, freq_dim=256) → dim → dim, act=SiLU )
time_projection         = Linear(dim → 6·dim)      # AdaLN modulation: shift/scale × (norm1, attn, ffn)
text_embedding          = Sequential( Linear(text_dim=4096 → dim), GELU, Linear(dim → dim) )

for block in 0..39:
    self_attn      = WanSelfAttention(dim=5120, heads=40, qk_norm=RMSNorm(head_dim=128, eps=1e-6))
                       + RoPE-3D over (T_lat, H_lat, W_lat) with optional per-head θ jitter (σ_θ)
    cross_attn     = WanCrossAttention(dim=5120, heads=40, qk_norm=True, cross_attn_norm=RMSNorm)
                       # Q from x, K/V from text context (UMT5)
    ffn            = Sequential( Linear(dim → ffn_dim=13824), GELU(approximate='tanh'), Linear(ffn_dim → dim) )
    norm3          = LayerNorm(dim, elementwise_affine=False)        # post-attn pre-ffn norm
    modulation     = Parameter(6·dim)                                 # per-block bias added to t-projection
    # Action conditioning injected here on a subset of blocks (see § 4)

head_norm    = LayerNorm(dim, elementwise_affine=False)
head         = Linear(dim → patch_T·patch_H·patch_W · out_dim) = Linear(5120 → 1·2·2·48=192)
```

**Activation:** SiLU in time MLP, GELU (`approximate='tanh'`) in FFN and in `text_embedding`.

**Attention:** flash-attention v3 preferred, v2 fallback, then `torch.nn.functional.scaled_dot_product_attention` last resort. Selection via `WAN_FA_VERSION` env or `--fa_version` CLI flag.

**Quantization (INT8):** `Int8Linear` (Triton kernel) replaces the QKV / O projections inside every `self_attn` block when `--use_int8` is set. The FFN, cross-attn, and modulation matrices stay bf16. This is exactly the "INT8 quantization to the attention projection layers in DiT" called out in the paper.

### 3. VAE — Wan2.2 3D Causal (encoder) + MG-LightVAE (decoder, optional)

`Matrix-Game-3/wan/modules/vae2_2.py` is a byte-for-byte port of the same `Wan2_2_VAE` used by Lance (see `LANCE_ARCHITECTURE.md` § 6). All constants identical:

```
z_dim         = 48
dim           = 160   (encoder base channels)
dec_dim       = 256   (decoder base channels — overridden by pruning_rate)
dim_mult      = [1, 2, 4, 4]
num_res_blocks= 2  (encoder), 3 (decoder)
patch_size    = 2
CACHE_T       = 2     (CausalConv3d frame cache)
RMSNorm everywhere (no GroupNorm)
```

**Same 48-channel mean / std** as Lance — reuse HartsyInference's existing `Wan22VaeNormalization` constants verbatim.

**MG-LightVAE pruning** = decoder channel widths multiplied by `(1.0 − pruning_rate)`. The encoder is **always** the full Wan2.2 encoder (used once per session to encode the input image). At load time the helper `infer_lightvae_pruning_rate_from_ckpt()` reads `decoder.conv1.weight.shape[0]` and derives the rate. The hybrid mode is "teacher encoder (unpruned) + student decoder (pruned)".

**VAE stride:** `(4, 16, 16)` over (T, H, W). With patch_size=(1,2,2), the **effective** compression from RGB to transformer tokens is `(4, 32, 32)` (T, H, W) → **token count for 1 segment ≈ 15 latent frames × (704/32) × (1280/32) = 15 × 22 × 40 = 13,200 tokens**.

### 4. ActionModule — Matrix-Game's only new transformer-side primitive

Source: `Matrix-Game-3/wan/modules/action_module.py`. Inserted as a per-block side branch on a *subset* of the 40 DiT blocks (the subset is passed in via the `blocks=[]` constructor arg from `WanModel.__init__`; the exact block indices are an Open Question — must be read from `model.py`'s block-construction loop, but a reasonable default observed in `action_module.py`'s usage pattern is every block).

```python
class ActionModule(nn.Module):
    def __init__(
        self,
        mouse_dim_in: int = 2,            # (mouse_x_delta, mouse_y_delta) per frame
        keyboard_dim_in: int = 6,         # 6 discrete movement actions (see § 5)
        hidden_size: int = 128,
        img_hidden_size: int = 1536,      # NOTE: not the DiT dim 5120; this is a *patch-projected* width
        keyboard_hidden_dim: int = 1024,
        mouse_hidden_dim: int = 1024,
        vae_time_compression_ratio: int = 4,
        windows_size: int = 3,            # temporal window over which actions attend
        heads_num: int = 16,
        patch_size: list = [1, 2, 2],
        qk_norm: bool = True,
        qkv_bias: bool = False,
        rope_dim_list: list = [8, 28, 28], # (t, h, w) RoPE axis split, sums to 64 = head_dim/2
        rope_theta = 256,
        mouse_qk_dim_list = [8, 28, 28],
        enable_mouse = True,
        enable_keyboard = True,
        blocks = [],                      # list of WanAttentionBlocks to attach to
        local_attn_size = 6,
    ): ...
```

**Two-stream design** (verbatim from the source summary):

1. **Mouse stream (continuous, self-attention):** mouse `(B, T, 2)` → MLP → `(B, T, mouse_hidden_dim=1024)` → projected up to `img_hidden_size=1536` → joined with image-feature tokens → **temporal self-attention with 3D RoPE** (rope_dim_list=[8,28,28], θ=256, applied separately to memory vs. prediction frames so the two segments don't share temporal positions). Local attention window: `local_attn_size=6` (latent frames).
2. **Keyboard stream (discrete, cross-attention):** keyboard one-hot `(B, T, 6)` → embedding/MLP → `(B, T, keyboard_hidden_dim=1024)` → split into K/V (`keyboard_hidden_dim * 2`) → cross-attended *into* the mouse-stream output as Q. Heads = 16 across this whole module.

Residual back into the DiT block: the ActionModule's output is added back to the block's hidden state after the standard self-attention but before cross-text-attention.

**RoPE in ActionModule:** completely separate from the main DiT's RoPE. θ=256 (vs the DiT's much larger θ — Open Question on exact DiT θ; the paper notes the perturbation σ_θ=0.8 is *added* to base θ during memory training). Per-head dims split (8, 28, 28) for (t, h, w) → total 64 = head_dim/2 = 128/2.

### 5. Action conditioning — exact input format

**Keyboard:** 6-dimensional vector per frame, indices 0..5 = **{W=forward, S=back, A=left, D=right, Q=down/none, ?=jump}** (the paper's data tuple `a_t ∈ {0,1}^6` and `Bench_actions_universal()` in `utils/conditions.py` confirm 6 dims; the W/A/S/D + 2 extras layout is inferred from the AAA-recording WSAD section). Encoded as binary one-hot per frame.

**Mouse:** 2-dimensional vector per frame `(Δx, Δy)` — these are **camera-rotation deltas** (yaw, pitch) in a *normalized* coordinate. The CLI defaults to magnitude **0.1 per discrete keystroke** for the canned action bench. In free-form interactive mode the user supplies arbitrary floats.

**Both** are sampled at the **video frame rate** (16 fps source / inference uses 24 fps render) and downsampled to the VAE latent rate (every 4 frames → 1 latent action). The ActionModule's `vae_time_compression_ratio=4` accounts for this — it gathers 4 consecutive frame-actions and feeds them as one latent action token (or windows them per `windows_size=3` latent frames).

**Camera pose:** in addition to raw mouse/keyboard the pipeline computes **per-frame extrinsics** (world-to-camera SE(3) matrices) from the integrated action stream via `utils/cam_utils.py:get_extrinsics()`. From extrinsics it builds **Plücker embeddings** (`(B, H_lat, W_lat, 16)` per frame, 6 ray-coord channels × ~3-context broadcasting — exact channel layout is Open Question 4). Plückers are passed as `plucker_emb` to `WanModel.forward()` and added/concatenated into the patch-embedded latents (the exact injection — addition vs concatenation vs side-conditioning — is in `WanModel.forward()` body, see Open Question 1).

**Action data tuple from training (paper Eq. § 4.1):**
```
D_t = ( I_t,     # RGB frame (H, W, 3)
        p_t,     # 3D position (x, y, z)
        r_t,     # rotation quaternion or Euler (4 or 3)
        c_t,     # camera intrinsics
        θ_t,     # rotation/yaw angle
        a_t )    # discrete action vector ∈ {0,1}^6
```

### 6. Memory mechanism — camera-aware long-horizon memory

(Paper § 3.2.) Per generated segment, the pipeline maintains:

- A **rolling buffer** of all past *predicted* latent frames since session start (the full session history).
- A **5-slot memory cache** of "relevant" past latents picked from that buffer based on camera-frustum overlap with the *current* prediction's camera trajectory.

**Selection.** `utils/cam_utils.py:select_memory_idx_fov()` runs each candidate past frame's camera frustum against the current segment's frustum (GPU-vectorized point-in-frustum test, sampling points uniformly in a sphere). The 5 frames with the highest visible-overlap fraction are selected. This is the "camera-aware" property — purely spatial overlap, no learned retrieval.

Concretely, in the released pipeline, the selection is observed running over a window of `range(1, 34, 8)` = indices [1,9,17,25,33] within the rolling buffer (stride 8 from frame 1 back to 33; this is a 5-tap subsample over the last ~33 frames). On longer sessions the window slides.

**Injection.** Memory frames are tensor-shape `(1, 48, 5, H_lat, W_lat)` (`x_memory`). They are run through the **same patch_embedding** as the current noisy latents, then **concatenated along the temporal/token axis** with the (a) past-frame latents (4 latents, clean) and (b) the current segment's noisy latents (10 latents). The DiT's self-attention then runs over the combined sequence (5+4+10 = 19 latent frames × spatial). RoPE-3D positions are assigned with the memory-frame slots receiving their **original** (historical) temporal indices, and the σ_θ=0.8 perturbation during memory training (frozen at inference σ_θ=0.0 after training) is what allows the model to handle non-contiguous t-axis gaps.

**Prediction-residual / frame re-injection (§ 3.1).** During *training* (not inference), each past frame `x^i` is perturbed by adding γ × the model's own prediction-residual `δ = x̂^i − x^i`:
```
x̃^i = x^i + γ · δ      with γ ~ uniform sampled
```
giving the trained model exposure to its own future inference-time errors. Memory frames get a separate γ_m. **At inference there is no residual injection** — past frames are passed in clean and memory frames are passed in clean. The benefit is fully baked into the trained weights.

**RoPE perturbation (§ 3.2):**
```
θ̂_h = θ_base × (1 + σ_θ × ε_h)
```
applied per-head with σ_θ=0.8 during training, σ_θ=0.0 (no perturbation) at inference. This is set via `WanModel(sigma_theta=0.0)` and lives in the self-attention RoPE.

### 7. Scheduler — FlowUniPCMultistep + DMD distillation

**Base model sampler:** **FlowUniPC** (`diffusers.FlowUniPCMultistepScheduler`), 50 inference steps, `sample_shift=5.0` (flow-matching timestep shift, applied as the SD3-style logit-normal warp), `sample_guide_scale=5.0` (classifier-free guidance), bf16. A fresh scheduler is instantiated **per segment** (the cached timesteps don't carry across segments — each segment is a fresh denoising trajectory conditioned on past latents + memory).

**Distilled model sampler:** **3 inference steps**, same FlowUniPC scheduler class. The distillation procedure is **Distribution Matching Distillation (DMD)** with a multi-segment rollout:

- **Cold-start phase:** 600 steps, single-segment, student LR 5e-7, critic LR 1e-7, student updates per iteration = 5.
- **Multi-segment phase:** 2,400 steps total, segment count `k` randomly sampled `1..6` per iteration, LR 1e-7, student updates per iteration = 3.

Both base and distilled use the **same bidirectional attention** architecture (no architectural change from base to distilled — the only difference is the weights and the step count).

**CFG.** Standard text-CFG with two forward passes per timestep:
```
v_full = DiT(x_t, t, context=cond_text,    plucker=plucker,    x_memory=x_memory, ...)
v_null = DiT(x_t, t, context=neg_prompt,   plucker=plucker_no_memory, ...)        # no memory on neg path
v      = v_null + sample_guide_scale · (v_full − v_null)
```
The neg path **explicitly disables memory** (passes a plücker variant with no memory tokens) — important parity point.

**Negative prompt** (default, baked into `wan/configs/config.py`):
```
"Vibrant colors, overexposure, static, blurred details, subtitles, style, artwork,
 painting, still image, overall grayness, worst quality, low quality, JPEG compression
 residue, ugly, mutilated, extra fingers, poorly drawn hands, poorly drawn faces,
 deformed, disfigured, malformed limbs, fused fingers, still image, cluttered background,
 three legs, crowded background, walking backwards"
```

### 8. Streaming session loop

(From `pipeline/inference_interactive_pipeline.py`.) Frame counts:

- **First segment:** 57 RGB frames → 15 latent frames (15 × 4 − 3 = 57 with the causal-VAE −3 boundary).
- **Each subsequent segment:** 40 RGB frames → 10 latent frames (10 × 4 = 40).
- **Past-latent overlap into next segment:** 4 latents (= 16 RGB frames).
- **Total frames after N iterations:** `57 + (N − 1) × 40`. At default `--num_iterations 12` → **497 frames** ≈ 20.7 s @ 24 fps.

Per-segment shapes:
- `latents`: `(1, 48, T_lat=15 first / 10 after, H_lat=44, W_lat=80)` for 704×1280.
- `x_memory`: `(1, 48, 5, 44, 80)`.
- `plucker_emb`: `(1, 44, 80, 16)` per current frame, broadcast across the 15-latent temporal extent.
- `mouse_cond`: `(1, T_RGB, 2)` (downsampled inside the ActionModule).
- `keyboard_cond`: `(1, T_RGB, 6)`.
- `context` (UMT5 text emb): `(1, ≤ 512, 4096)`.

### 9. Real-time @ 40 FPS

The "40 FPS" headline is for a **distilled 5B + INT8 attn + MG-LightVAE 75 % + async VAE decode worker + camera-FOV memory retrieval on GPU**, on a `NUM_GPUS = 9` system (paper § 3.4: "8 GPUs are used for DiT inference and 1 GPU is dedicated to VAE decoding"). Ablation (Table 1):

| Config | FPS | Drop |
|---|---|---|
| Full | ~40 | — |
| − INT8 | 27.38 | 12.62 |
| − VAE pruning | 25.79 | 14.21 |
| − GPU memory retrieval | 6.60 | 33.40 |

So **GPU-side FOV retrieval is the single biggest perf knob**, then VAE pruning, then INT8. A single-GPU install will be far below 40 FPS and that's OK — this is acceptable for an offline-render workflow.

**Sequence-parallel.** Within the 8-GPU DiT chunk, attention is split across GPUs via **Ulysses-style sequence parallelism** (DiT activations split along the sequence-length axis). `max_seq_len` is the per-rank slice = `total_seq_len / sp_size`.

**Async VAE worker.** A dedicated process on GPU 8 owns the (MG-Light)VAE decoder. Main DiT process queues a finished latent segment via a multiprocessing queue and continues with the next segment; the worker decodes and emits frames to an `ack_queue`. Warmup iterations: `--async_vae_warmup_iters 1`.

### 10. 2×14B MoE variant (paper-only)

(§ 3.5.) The 28B MoE consists of **two 14B "high-noise" experts** specialized by **viewpoint**: one trained on first-person data only, one trained on third-person data only. A routing decision (deterministic, not learned) at sample-start picks the expert based on the input image's view type. This is **not** a Mixtral-style per-token MoE — it's a coarse model-level switch. Each expert is itself a Wan2.2-shape DiT scaled up to 14B (specific shape TBD; not yet released).

For HartsyInference, the 5B is the only viable target until the MoE actually drops. When/if it does, the implementation is "load expert A xor expert B at session start" — no per-token routing code needed.

## Key Numbers / Constants

| Constant | Value | Where used |
|---|---|---|
| `dim` (DiT hidden) | **5120** | WanModel backbone |
| `num_layers` | **40** | DiT depth |
| `num_heads` | **40** | head_dim = 128 |
| `ffn_dim` | **13,824** | FFN inner |
| `in_dim` / `out_dim` | **48** / **48** | VAE latent channels in / out |
| `patch_size` (T, H, W) | **(1, 2, 2)** | Conv3d patchify |
| `freq_dim` | **256** | sinusoidal timestep embed |
| `text_len` | **512** | max UMT5 tokens |
| `text_dim` | **4096** | UMT5-XXL hidden |
| `eps` (RMSNorm) | **1e-6** | all RMSNorms |
| `qk_norm` | True | per-head RMSNorm on Q/K |
| `cross_attn_norm` | True | RMSNorm on cross-attn input |
| `window_size` | (-1, -1) | full attention |
| `use_memory` | True | enables x_memory path |
| `sigma_θ` (RoPE perturbation) | 0.8 (train) / **0.0 (inference)** | self-attn RoPE jitter |
| VAE `z_dim` | **48** | latent channels |
| VAE stride (T, H, W) | **(4, 16, 16)** | RGB→latent compression |
| VAE `dim`, `dec_dim` | 160, 256 | base channels |
| VAE `dim_mult` | [1, 2, 4, 4] | — |
| VAE `num_res_blocks` | 2 (enc) / 3 (dec) | — |
| VAE `patch_size` | 2 | RGB→12-channel pre-conv |
| VAE `CACHE_T` | 2 | CausalConv3d frame cache |
| MG-LightVAE pruning rates | 0.5, 0.75 | decoder channel multiplier `(1−p)` |
| ActionModule `mouse_dim_in` | **2** | (Δx, Δy) per frame |
| ActionModule `keyboard_dim_in` | **6** | discrete action one-hot |
| ActionModule `hidden_size` | 128 | internal |
| ActionModule `img_hidden_size` | 1536 | patch-projected DiT side |
| ActionModule `keyboard_hidden_dim` | 1024 | — |
| ActionModule `mouse_hidden_dim` | 1024 | — |
| ActionModule `vae_time_compression_ratio` | 4 | matches VAE temporal stride |
| ActionModule `windows_size` | 3 | temporal local-attn window (latent frames) |
| ActionModule `heads_num` | 16 | — |
| ActionModule `rope_dim_list` | [8, 28, 28] | (t, h, w) — sums to 64 = head_dim/2 |
| ActionModule `rope_theta` | 256 | RoPE base θ |
| ActionModule `local_attn_size` | 6 | local-attn span (latent frames) |
| Frames per first segment | **57** | RGB |
| Frames per subsequent segment | **40** | RGB |
| Past-latent overlap | **4** (= 16 RGB) | continuity |
| Memory slots | **5** | `x_memory` temporal dim |
| Memory candidate window | range(1, 34, 8) → [1,9,17,25,33] | 5-tap stride-8 over last 33 |
| `sample_shift` | **5.0** | flow-matching timestep shift |
| `sample_guide_scale` | **5.0** | CFG scale |
| `num_inference_steps` (base) | **50** | FlowUniPC |
| `num_inference_steps` (distilled) | **3** | DMD-distilled |
| Default resolution | **704 × 1280** | (H × W) — 720p portrait of 1280 wide |
| Latent grid for 704×1280 | **44 × 80** | (H/16, W/16) |
| Sequence tokens per segment (first) | 15 × 22 × 40 = 13,200 | post-patchify |
| FPS render rate | 24 (sampling 16) | — |
| Real-time target FPS | **40** | with 9-GPU cluster + INT8 + MG-LightVAE 75 % |
| Async VAE warmup | 1 iteration | `--async_vae_warmup_iters 1` |
| Max action magnitude (canned) | 0.1 | mouse Δ per discrete keystroke |

## Data Layouts / Formats

### HuggingFace repo file tree — `Skywork/Matrix-Game-3.0` (total ≈ 56.6 GB)

```
Skywork/Matrix-Game-3.0/
├── .gitattributes                       1.7   kB
├── README.md                            5.46  kB
├── model_index.json                     46    B     # {"_class_name": "MatrixGame3I2VPipeline"}
├── framework.png                        1.92  MB    # architecture diagram
├── Wan2.2_VAE.pth                       2.82  GB    # full Wan2.2 3D causal VAE (encoder + decoder)
├── MG-LightVAE.pth                      2.74  GB    # 50%-pruned decoder + full encoder
├── MG-LightVAE_v2.pth                   841   MB    # 75%-pruned decoder + full encoder
├── models_t5_umt5-xxl-enc-bf16.pth     11.4   GB    # UMT5-XXL encoder weights (bf16, encoder-only)
├── base_model/
│   ├── config.json                      1.06  kB    # diffusers WanModel config
│   └── diffusion_pytorch_model.safetensors  12.9 GB # 5B DiT, bf16, base (50-step)
├── base_distilled_model/
│   └── diffusion_pytorch_model.safetensors  25.9 GB # student + critic / EMA bundle (3-step)
│   └── (config.json present)            ~1   kB
└── google/
    └── umt5-xxl/                        21.5  MB    # tokenizer files (spm.model, tokenizer_config.json, etc.)
```

### Expected tensor key prefixes (Wan2.2 native naming, not diffusers)

Confirmed from the `Wan-AI/Wan2.2-TI2V-5B` `diffusion_pytorch_model.safetensors.index.json`:

```
patch_embedding.{weight, bias}
time_embedding.0.{weight, bias}                  # Linear
time_embedding.2.{weight, bias}                  # Linear (after SiLU)
time_projection.1.{weight, bias}                 # Linear → 6·dim
text_embedding.0.{weight, bias}                  # Linear text_dim→dim
text_embedding.2.{weight, bias}                  # Linear dim→dim (after GELU)
blocks.{0..39}.self_attn.{q, k, v}.{weight, bias}
blocks.{0..39}.self_attn.norm_{q, k}.weight       # RMSNorm
blocks.{0..39}.self_attn.o.{weight, bias}
blocks.{0..39}.cross_attn.{q, k, v}.{weight, bias}
blocks.{0..39}.cross_attn.norm_{q, k}.weight
blocks.{0..39}.cross_attn.o.{weight, bias}
blocks.{0..39}.ffn.0.{weight, bias}               # Linear dim→ffn_dim
blocks.{0..39}.ffn.2.{weight, bias}               # Linear ffn_dim→dim
blocks.{0..39}.norm3.{weight, bias}               # LayerNorm (post-attn)
blocks.{0..39}.modulation                          # Parameter(6·dim)
head.head.{weight, bias}                           # Linear dim → patch_T·patch_H·patch_W·out_dim
head.modulation                                    # Parameter(2·dim) for final AdaLN
```

**Matrix-Game-specific additions** (live inside the same `model.safetensors`):

```
blocks.{i}.action_module.mouse_proj.*              # mouse stream MLP & attn
blocks.{i}.action_module.keyboard_proj.*           # keyboard embed
blocks.{i}.action_module.{q, k, v, o}.*            # per-stream
blocks.{i}.action_module.cross_attn.*              # mouse→keyboard cross-attn
blocks.{i}.action_module.norm_*.weight             # RMSNorms
plucker_proj.*                                     # plücker embedding projection (Conv or Linear)
```

**These names need to be confirmed by a one-time `safetensors safe_open` dump** — see Open Question 1.

### `base_model/config.json` (diffusers WanModel format)

Inferred shape (1.06 kB file):
```json
{
  "_class_name": "WanModel",
  "_diffusers_version": "0.33.0",
  "dim": 5120, "ffn_dim": 13824,
  "num_heads": 40, "num_layers": 40,
  "in_dim": 48, "out_dim": 48,
  "freq_dim": 256, "text_len": 512,
  "eps": 1e-06,
  "model_type": "ti2v"
}
```
(Open Question 2 — need to verify; this differs from Wan-AI's `Wan2.2-TI2V-5B/config.json` which lists `dim=3072 / num_heads=24 / num_layers=30`. Either Matrix-Game ships a *different* config.json reflecting the actual finetuned shape, or the 12.9 GB safetensors really does pack a 30-layer 3072-dim DiT — verify before implementation.)

### Tokenization

Standard UMT5 SentencePiece (`google/umt5-xxl`). Used only as a frozen encoder (no decoder). Output: `(1, ≤ 512, 4096)` bf16 tensor passed as `context` to the DiT. The Matrix-Game pipeline does **not** apply a special chat template; raw prompt text is encoded directly.

## Algorithm Steps

### One denoising step (per segment)

```
inputs:
  latents       : (B=1, 48, T_lat, H_lat, W_lat)   # noisy current segment
  past_latents  : (B=1, 48,    4,  H_lat, W_lat)   # clean, from previous segment
  memory_latents: (B=1, 48,    5,  H_lat, W_lat)   # clean, FOV-retrieved
  mouse_cond    : (B=1, T_RGB, 2)
  kbd_cond      : (B=1, T_RGB, 6)
  plucker       : (B=1, H_lat, W_lat, 16)          # current camera trajectory
  text_emb      : (1, 512, 4096)                   # UMT5
  neg_text_emb  : (1, 512, 4096)
  t             : scalar in [0, 1]

1.  # ---- patchify all three sets of latents through the same Conv3d ----
    x_curr   = patch_embedding(latents)             # (1, dim, T_lat, H_lat/2, W_lat/2)
    x_past   = patch_embedding(past_latents)        # (1, dim, 4,     H_lat/2, W_lat/2)
    x_mem    = patch_embedding(memory_latents)      # (1, dim, 5,     H_lat/2, W_lat/2)

2.  # ---- concatenate along temporal axis ----
    x        = concat([x_mem, x_past, x_curr], dim=2)   # T_total = 5 + 4 + T_lat
    flatten to sequence: (1, T_total · h · w, dim)

3.  # ---- 3D RoPE indices: memory frames get *original historical* t-indices,
    #      past+curr get contiguous t-indices, with σ_θ=0 jitter ----
    rope = build_3d_rope_indices(...)

4.  # ---- timestep & text embeddings ----
    t_emb   = time_embedding(sinusoidal(t, freq_dim=256))
    shift_scale = time_projection(t_emb).view(6, dim)    # AdaLN params

5.  # ---- 40 DiT blocks ----
    for b in 0..39:
        x = AdaLN(x, shift_scale[0:2]); x = self_attn(x, rope=rope) ; x_attn = x
        if b ∈ action_blocks:
            x_attn = x_attn + action_module(x_attn, mouse_cond, kbd_cond)
        x = x + AdaLN(x_attn, shift_scale[2])
        x = AdaLN(norm3(x), shift_scale[3:5]); x = cross_attn(x, text_emb=context)
        x = x + AdaLN(ffn(x), shift_scale[5])

6.  # ---- read out only the current-frame tokens ----
    x_curr = x[:, mem+past_token_count:, :]     # drop memory + past tokens
    pred   = head(head_norm(x_curr))            # Linear dim → 192
    pred   = unpatchify(pred)                   # (1, 48, T_lat, H_lat, W_lat)

7.  return pred                                  # = velocity prediction (flow-matching)
```

### Streaming session loop

```
# session start
img             = load_input_image()                         # (1, 3, H, W)
img_latent      = vae.encode(img)                            # (1, 48, 1, H/16, W/16)
text_emb        = umt5.encode(prompt)                        # (1, 512, 4096)
neg_text_emb    = umt5.encode(neg_prompt)
history_buf     = [ ]                                        # list of (latent, camera_pose) tuples
cam_pose        = identity()                                 # initial extrinsics

# segment 0 — bootstrap
T_lat = 15  ;  T_rgb = 57
actions   = get_user_actions_for_T_rgb_frames()              # mouse+kbd buffers
cam_poses = integrate_actions_to_camera_poses(cam_pose, actions, T_rgb)
plucker   = compute_plucker(cam_poses, intrinsics)           # (1, H_lat, W_lat, 16)

past_lat   = repeat(img_latent, 4, dim=2)                    # bootstrap: 4 copies of seed image
mem_lat    = repeat(img_latent, 5, dim=2)                    # bootstrap memory

latents    = torch.randn(1, 48, T_lat, H_lat, W_lat)
sched      = FlowUniPCMultistepScheduler(num_steps=3 or 50, shift=5.0)
for t in sched.timesteps:
    v_full = WanModel(latents, t, text_emb,    past_lat, mem_lat, mouse, kbd, plucker)
    v_null = WanModel(latents, t, neg_text_emb, past_lat, mem_lat_zero, mouse, kbd, plucker_no_mem)
    v      = v_null + 5.0 · (v_full − v_null)
    latents = sched.step(v, t, latents)

rgb_segment0 = vae.decode(latents)                          # 57 RGB frames
emit(rgb_segment0)
history_buf += [(latents[:,:,k], cam_poses[k]) for k in 0..T_lat]
cam_pose    = cam_poses[-1]

# segment 1..N − 1 — streaming
for seg in 1..N−1:
    T_lat = 10  ;  T_rgb = 40
    actions   = get_user_actions_for_T_rgb_frames()
    cam_poses = integrate_actions_to_camera_poses(cam_pose, actions, T_rgb)
    plucker   = compute_plucker(cam_poses, intrinsics)

    past_lat  = stack([history_buf[i].latent for i in last_4_indices], dim=2)
    mem_idx   = select_memory_idx_fov(history_buf, cam_poses, k=5)
    mem_lat   = stack([history_buf[i].latent for i in mem_idx], dim=2)

    latents   = torch.randn(1, 48, T_lat, H_lat, W_lat)
    sched     = FlowUniPCMultistepScheduler(num_steps=3 or 50, shift=5.0)
    for t in sched.timesteps:
        v_full = WanModel(latents, t, text_emb,    past_lat, mem_lat, mouse, kbd, plucker)
        v_null = WanModel(latents, t, neg_text_emb, past_lat, mem_lat_zero, mouse, kbd, plucker_no_mem)
        v      = v_null + 5.0 · (v_full − v_null)
        latents = sched.step(v, t, latents)

    rgb_segment = vae.decode(latents)                        # async-queue if multi-GPU
    emit(rgb_segment)
    history_buf += [(latents[:,:,k], cam_poses[k]) for k in 0..T_lat]
    cam_pose    = cam_poses[-1]
```

## Reference Implementations

**Primary:** [`SkyworkAI/Matrix-Game` — `Matrix-Game-3/`](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-3)

Source-of-truth files:
- [`Matrix-Game-3/generate.py`](https://github.com/SkyworkAI/Matrix-Game/blob/main/Matrix-Game-3/generate.py) — CLI entry; arg parsing, dist init, pipeline selection.
- [`Matrix-Game-3/pipeline/inference_pipeline.py`](https://github.com/SkyworkAI/Matrix-Game/blob/main/Matrix-Game-3/pipeline/inference_pipeline.py) — `MatrixGame3Pipeline` (standard / canned-action mode).
- [`Matrix-Game-3/pipeline/inference_interactive_pipeline.py`](https://github.com/SkyworkAI/Matrix-Game/blob/main/Matrix-Game-3/pipeline/inference_interactive_pipeline.py) — streaming interactive variant; per-segment loop, memory selection, async-VAE queue.
- [`Matrix-Game-3/pipeline/vae_config.py`](https://github.com/SkyworkAI/Matrix-Game/blob/main/Matrix-Game-3/pipeline/vae_config.py) — VAE selection (`wan2.2` vs `mg_lightvae` vs `mg_lightvae_v2`).
- [`Matrix-Game-3/pipeline/vae_worker.py`](https://github.com/SkyworkAI/Matrix-Game/blob/main/Matrix-Game-3/pipeline/vae_worker.py) — async decode worker process.
- [`Matrix-Game-3/wan/modules/model.py`](https://github.com/SkyworkAI/Matrix-Game/blob/main/Matrix-Game-3/wan/modules/model.py) — `WanModel`, `WanAttentionBlock`, `WanSelfAttention`, `WanCrossAttention`, `Int8Linear`.
- [`Matrix-Game-3/wan/modules/action_module.py`](https://github.com/SkyworkAI/Matrix-Game/blob/main/Matrix-Game-3/wan/modules/action_module.py) — `ActionModule` (dual-stream mouse+keyboard).
- [`Matrix-Game-3/wan/modules/attention.py`](https://github.com/SkyworkAI/Matrix-Game/blob/main/Matrix-Game-3/wan/modules/attention.py) — FlashAttention v3/v2 dispatcher + SDPA fallback.
- [`Matrix-Game-3/wan/modules/posemb_layers.py`](https://github.com/SkyworkAI/Matrix-Game/blob/main/Matrix-Game-3/wan/modules/posemb_layers.py) — RoPE 1D / n-D, real & complex variants, NTK theta rescale.
- [`Matrix-Game-3/wan/modules/vae2_2.py`](https://github.com/SkyworkAI/Matrix-Game/blob/main/Matrix-Game-3/wan/modules/vae2_2.py) — Wan2.2 + MG-LightVAE 3D causal VAE.
- [`Matrix-Game-3/wan/modules/t5.py`](https://github.com/SkyworkAI/Matrix-Game/blob/main/Matrix-Game-3/wan/modules/t5.py) + `tokenizers.py` — UMT5-XXL encoder + SentencePiece tokenizer wrapper.
- [`Matrix-Game-3/wan/configs/config.py`](https://github.com/SkyworkAI/Matrix-Game/blob/main/Matrix-Game-3/wan/configs/config.py) — `matrix_game3` EasyDict with all shape constants and inference defaults.
- [`Matrix-Game-3/wan/configs/shared_config.py`](https://github.com/SkyworkAI/Matrix-Game/blob/main/Matrix-Game-3/wan/configs/shared_config.py) — UMT5 dtype (bf16), 1000-timestep flow base, 16 fps sample rate, 481-frame max, base negative prompt.
- [`Matrix-Game-3/wan/triton_kernels.py`](https://github.com/SkyworkAI/Matrix-Game/blob/main/Matrix-Game-3/wan/triton_kernels.py) — INT8 W8A8 GEMM triton kernel (port to CUDA PTX for HartsyInference).
- [`Matrix-Game-3/utils/cam_utils.py`](https://github.com/SkyworkAI/Matrix-Game/blob/main/Matrix-Game-3/utils/cam_utils.py) — SE(3) inverse, SLERP, Plücker embeddings, `select_memory_idx`, `select_memory_idx_fov` (GPU frustum overlap).
- [`Matrix-Game-3/utils/conditions.py`](https://github.com/SkyworkAI/Matrix-Game/blob/main/Matrix-Game-3/utils/conditions.py) — canned action bench (`Bench_actions_universal`), action→tensor packing, `combine_data` (asserts `num_frames % 4 == 1`).

**Paper:** [arXiv:2604.08995 v2](https://arxiv.org/abs/2604.08995v2) (HTML at [`arxiv.org/html/2604.08995v2`](https://arxiv.org/html/2604.08995v2)). Title: *"Matrix-Game 3.0: Real-Time and Streaming Interactive World Model with Long-Horizon Memory"*. Authors: Zile Wang + 22 co-authors, Skywork AI Matrix-Game Team, 2026. Includes the architecture diagram referenced as `framework.png` in the HF repo.

**Project page:** [matrix-game-v3.github.io](https://matrix-game-v3.github.io/) — demo videos showing 720p @ 40 FPS streaming.

**HuggingFace:** [`Skywork/Matrix-Game-3.0`](https://huggingface.co/Skywork/Matrix-Game-3.0).

**Base model:** [`Wan-AI/Wan2.2-TI2V-5B`](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B), [`Wan-Video/Wan2.2` GitHub](https://github.com/Wan-Video/Wan2.2). Wan2.2 itself derives from Wan2.1.

**ComfyUI port:** [`Yuan-ManX/ComfyUI-Matrix-Game`](https://github.com/Yuan-ManX/ComfyUI-Matrix-Game) — but as of 2026-05-24 this targets the **older 17B Matrix-Game 1.0**, not 3.0. No 3.0 ComfyUI integration exists yet.

**Acknowledged influences** (from README):
- [Self-Forcing](https://github.com/guandeh17/Self-Forcing) — autoregressive video distillation training scheme (parent of DMD-segment).
- [GameFactory](https://github.com/KwaiVGI/GameFactory) — action-control module pattern (where the dual-stream mouse/keyboard design comes from).
- [LightX2V](https://github.com/ModelTC/lightx2v) — INT8 quantization framework (where `Int8Linear` + the Triton kernel come from).
- [lingbot-world](https://github.com/Robbyant/lingbot-world) — context-parallel framework (Ulysses-style SP).

## Differences Between Implementations

Only one reference implementation exists today (the ByteDance-style "single-codebase" pattern — Skywork ships `Matrix-Game-3/` and that's it). No diffusers PR exists upstream. The ComfyUI fork targets v1.0 only. **No disagreements to reconcile.**

The only notable inconsistency is between the *Wan-AI base config* (`dim=3072, num_heads=24, num_layers=30`) and the *Matrix-Game inference config* (`dim=5120, num_heads=40, num_layers=40`). The latter is what's loaded at inference by Matrix-Game's `generate.py`, and the 12.9 GB safetensors size (5B × 2 bytes = 10 GB + overheads) is consistent with the 5120/40/40 shape. The smaller Wan-AI `config.json` likely describes a *diffusers-distilled* version that lives in the same HF repo but isn't what Matrix-Game uses. **Always trust `Matrix-Game-3/wan/configs/config.py` for the actual inference shape.**

## Open Questions

1. **Exact tensor key names in `base_model/diffusion_pytorch_model.safetensors`** — particularly: are the ActionModule weights stored as `blocks.{i}.action_module.*` (nested), or as a separate top-level `action_modules.{i}.*` table? And: which subset of the 40 blocks have an ActionModule attached? Dump locally with:
   ```python
   from safetensors import safe_open
   with safe_open("base_model/diffusion_pytorch_model.safetensors", framework="pt") as f:
       for k in sorted(f.keys()):
           print(k, f.get_tensor(k).shape)
   ```

2. **`base_model/config.json` exact contents.** The 1.06 kB file may be the diffusers WanModel config with the inflated 5120/40/40 shape, or may match the 3072/24/30 Wan-AI diffusers shape (which would imply Matrix-Game is loading from the `.safetensors` directly via its own `WanModel.__init__` and ignoring `config.json`). Confirm by reading the file after download.

3. **DiT RoPE θ base value.** The ActionModule uses θ=256; the main DiT's self-attention RoPE θ is **not** in `config.py` and must be read from `model.py`'s `WanSelfAttention.__init__` (likely `1_000_000` or `10_000`, matching Wan2.2; cannot extract from web summary).

4. **Plücker embedding exact channel layout.** The pipeline summary says `(1, H_lat, W_lat, 16)` per frame. Standard Plücker is 6 floats (ray origin × ray direction); 16 channels suggests something like 6 (Plücker) + 9 (3×3 relative rotation) + 1 (relative translation magnitude) or 4 sets of (ray_o, ray_d) for corner pixels. Read `utils/cam_utils.py:get_plucker_embeddings()` locally.

5. **Which blocks have an ActionModule?** `ActionModule.__init__(blocks=[])` takes a list of `WanAttentionBlock` references. The construction in `WanModel.__init__` decides this — likely all 40, but could be a subset. Confirm in source.

6. **Exact AdaLN modulation scheme.** `time_projection(t_emb).view(2, 6, dim)` — paper / source has 6 modulation params per block but the split (which goes to pre-attn norm, which to post-attn, which to ffn) needs to be read from `WanAttentionBlock.forward()`. Standard Wan pattern is `[norm1_shift, norm1_scale, attn_gate, norm2_shift, norm2_scale, ffn_gate]`.

7. **FlowUniPC scheduler details for HartsyInference port.** Need to implement `FlowUniPCMultistepScheduler` (or the FlowMatch-Euler equivalent with `shift=5.0` — verify whether UniPC vs Euler gives matching outputs at 50 steps and matching outputs at 3 distilled steps).

8. **DMD-distilled student-only weight layout.** The 25.9 GB `base_distilled_model/` is roughly 2× the 12.9 GB base. Likely contains student + critic (DMD needs both) + maybe an EMA copy. For inference only the student is needed; figure out which key prefix is the student so HartsyInference can load only that ~13 GB slice. Probably keys like `student.*`, `critic.*`, `ema.*` — confirm by key dump.

9. **`MG-LightVAE.pth` / `MG-LightVAE_v2.pth` tensor key naming.** Are they drop-in replacements (same keys as `Wan2.2_VAE.pth` but smaller shapes), or do they have a `student.` / `decoder_pruned.` prefix? Affects whether the existing Wan2.2-VAE loader can load them with just a shape-tolerant mode.

10. **INT8 quantization scale layout.** `Int8Linear` packs scales as a side tensor (probably `*.weight_scale` of shape `[out_features]`). Confirm exact name and dtype (FP32? BF16?) for HartsyInference INT8 kernel parity.

11. **The `--interactive` mode CLI contract.** How does the streaming pipeline read live keyboard/mouse inputs from the user? Stdin? A socket? A file watcher? Read the top of `inference_interactive_pipeline.py:__init__` and the `generate.py` interactive branch.

12. **Frame-rate clarification.** Source is 16 fps (`wan_shared_cfg.sample_fps = 16`), CLI generates at 24 fps, paper claims 40 fps real-time. The 40 fps figure is *output rate after distillation and async pipelining* — not the inference clock. Document for HartsyInference users so they don't expect 40 fps on single-GPU.

## Implementation Notes

### How this maps to HartsyInference packages

- **New package `HartsyInference.World`** (Phase 10) — entirely new. Contains:
  - `MatrixGame3Pipeline.cs` (streaming session manager: rolling buffer, memory cache, per-segment denoise loop, async VAE worker).
  - `Pipelines/MatrixGame3StandardPipeline.cs` (canned-action one-shot; matches `inference_pipeline.py`).
  - `Pipelines/MatrixGame3InteractivePipeline.cs` (live actions; matches `inference_interactive_pipeline.py`).
  - `CameraUtils.cs` — SE(3) inverse, SLERP quaternion interpolation, integrate-actions-to-poses, `GetPluckerEmbeddings`.
  - `MemoryRetrieval.cs` — `SelectMemoryByFovOverlap` (port `select_memory_idx_fov`), sphere-point sampling, frustum test (GPU kernel).
  - `ActionEncoder.cs` — packs raw (mouse_dx, mouse_dy, keyboard[6]) per-frame buffers into the tensors the DiT expects.
- **`HartsyInference.Video`** — shared with Wan2.2 / future Wan-family pipelines:
  - `Models/Denoisers/WanDit.cs` (the 40-layer Wan2.2 DiT — also needed for any Wan2.2 video pipeline).
  - `Models/Denoisers/DiTBlocks/WanAttentionBlock.cs`, `WanSelfAttention.cs`, `WanCrossAttention.cs`.
  - `Models/Denoisers/DiTBlocks/MatrixGameActionModule.cs` (new — only Matrix-Game uses it, but it lives next to the WanDit blocks).
  - `Models/Vae/Wan22VaeDecoder.cs` (full Wan2.2 VAE — likely already needed by other pipelines).
  - `Models/Vae/MgLightVaeDecoder.cs` (pruned variant — same code path with shape-tolerant loader and `decoder_channels = dim * (1 − pruning_rate)`).
  - `Schedulers/FlowUniPCMultistepScheduler.cs` (new — not in HartsyInference yet; the FlowMatchEulerDiscreteScheduler may give acceptable results at 50 steps but a real UniPC is needed for 3-step distilled).
- **`HartsyInference.Diffusion`** — provides reusable bits:
  - `Models/TextEncoders/Umt5XxlEncoder.cs` (already needed for AuraFlow / Pile-T5; identical model here).
  - `Utilities/SinusoidalTimestepEmbedding.cs` (already exists in `DiTUtils`).
  - `Utilities/RotaryPositionEmbeddingNd.cs` (extend existing 1D RoPE helper to N-D with per-axis `rope_dim_list`).
- **`HartsyInference.ModelAssets`** — new converter:
  - `CheckpointConverters/MatrixGame3CheckpointConverter.cs` — splits `model.safetensors` into `dit.*` (Wan core), `action.*` (per-block ActionModule), `plucker_proj.*`. Handles the optional `base_distilled_model/` student-only extraction.
  - Existing `Wan22VaeConverter` (if it exists from Lance work) handles `Wan2.2_VAE.pth`; add an `MgLightVaeConverter` that reads pruning rate from `decoder.conv1.weight.shape[0]` per `infer_lightvae_pruning_rate_from_ckpt()`.
- **`HartsyInference.ModelAssets.Tokenizers`** — UMT5 SentencePiece. The Lance/AuraFlow text-encoder work should already cover this; ensure `google/umt5-xxl/spm.model` loads.

### Net-new backend / kernel work required

1. **3D Conv input patchify (Conv3d 1×2×2 stride 1×2×2).** Trivial — degenerate to 2D Conv per frame. Already covered by the Wan2.2 VAE pipeline if landed.
2. **3D RoPE with split-axis `rope_dim_list = [8, 28, 28]` / per-axis θ.** ActionModule uses this; the main DiT uses M-RoPE-like 3D too. Extend the existing `RotaryPositionEmbedding` to take per-axis dim splits and per-axis θ overrides. **Reuse for any Wan/LTX video pipeline.**
3. **CausalConv3d streaming decode (`CACHE_T = 2`).** Already required by Lance Phase 9 video work; same shape, same constants — purely reuse.
4. **MG-LightVAE.** Same architecture as Wan2.2 VAE with `dim_mult` scaled by `(1 − pruning_rate)` only on the decoder side. Reuse the encoder code, just allow shape-flexible loading of the decoder side.
5. **ActionModule dual-stream attention.** Small (~128/1024/1536 dims, 16 heads). Plain bf16 SDPA is fine for v1 — the small dims won't bottleneck. No new kernel.
6. **GPU camera-frustum overlap test.** Naively: sample ~2k random 3D points in a sphere, project each candidate frustum onto each, count overlaps. A 1-shot CUDA kernel; falls back to CPU on single-GPU installs (slow path, but still correct).
7. **INT8 W8A8 GEMM (attention QKV/O only).** The reference uses a Triton kernel. HartsyInference will need a PTX-compiled equivalent (one CUDA C++ source → PTX, called via the existing CUDA driver API path). Scale per output channel, dtype FP32 or BF16 (TBD per Open Question 10). **First INT8 path in the project — design carefully.** Alternatively, ship without INT8 in v1 and accept the ~30 % slowdown.
8. **Async VAE worker.** Multi-GPU only. v1 can ignore (single-GPU runs serially); when multi-GPU lands, wire a background `Task` that owns the second GPU's CUDA context and pops latents off a queue.
9. **Ulysses sequence-parallel attention.** Multi-GPU only. Same comment — defer to a later phase.
10. **FlowUniPC scheduler.** This is the most novel scheduler in HartsyInference's lineup. The simplest correct port is the diffusers `FlowUniPCMultistepScheduler` algorithm with `shift=5.0` applied to the timestep grid in the SD3 logit-normal style. At 3 distilled steps a precomputed timestep table is fine; at 50 steps the standard UniPC corrector loop is needed.

### VRAM and viability per target GPU

| GPU | VRAM | Single-GPU MG3 base (50-step) | Single-GPU MG3 distilled (3-step) | 9-GPU 40 FPS cluster |
|---|---|---|---|---|
| RTX 3060 12 GB | 12 GB | DiT bf16 ≈ 12.4 GB — won't fit. FP8 cast (~6.2 GB) + UMT5 (~5.7 GB FP8) + VAE (~1.4 GB FP16) ≈ 13 GB — tight, will need text encoder offload. | Same as base (same DiT). | n/a |
| RTX 4090 24 GB | 24 GB | Comfortable FP16 (~12.4 GB DiT + 5.7 GB UMT5 + 1.4 GB VAE ≈ 19.5 GB; +activations). Short clips OK. | Comfortable. ~3-step inference is ~17× faster than 50-step. | n/a |
| A100 / H100 40+ GB | 40+ | Comfortable. | Comfortable. | n/a |
| 9× H100 cluster | 9 × 80 GB | n/a | n/a | **Target config for 40 FPS** (paper §3.4). |

Recommended `QualityProfileApplier` presets:
- **Low** = distilled 3-step + INT8 attn + MG-LightVAE v2 (75 %) — fits 12 GB, ~5 fps on RTX 3060.
- **Medium** = distilled 3-step + bf16 + MG-LightVAE (50 %) — fits 24 GB, ~10 fps on RTX 4090.
- **High** = base 50-step + bf16 + Wan2.2 VAE — fits 24 GB, ~30 s per 57-frame segment on RTX 4090.

Q4_K GGUF dumps do not yet exist for Matrix-Game 3.0. When/if a community port appears under `lightx2v/` or similar, a Q4_K DiT (~3 GB) makes the 12 GB single-GPU streaming path actually viable.

### Ordering / dependencies for the build

1. **Land the Wan2.2 DiT first** as a generic dependency (`HartsyInference.Video`). The 40-layer 5120-dim block is reusable for any Wan2.2 pipeline. Validate on Wan2.2-TI2V-5B's own demo (text → 5-sec video) before tackling Matrix-Game.
2. **Land the Wan2.2 3D causal VAE** (`HartsyInference.Video.Models.Vae.Wan22VaeDecoder`). This is the same VAE Lance uses — if Lance lands first, this is purely reuse.
3. **Land UMT5-XXL encoder** in `HartsyInference.Diffusion.Models.TextEncoders`. Shared with AuraFlow / Pile-T5.
4. **Land `FlowUniPCMultistepScheduler`** in `HartsyInference.Video.Schedulers`.
5. **Then** add the ActionModule block and `WanModel.forward()` overrides for action conditioning + memory tokens.
6. **Then** add the streaming `MatrixGame3Pipeline` with rolling buffer + memory selection + per-segment denoise.
7. **Optional follow-ups:** MG-LightVAE, INT8 attention path, async VAE worker, Ulysses SP. These are the perf knobs that get the cluster from ~6 fps to 40 fps; correctness lands without them.

### Test-skipping discipline

Following project convention:

- `MatrixGame3StandardPipelineTests` should require: `MATRIX_GAME_3_BASE_PATH` (or `_DISTILLED_PATH`), `WAN22_VAE_PATH`, `UMT5_XXL_PATH`; VRAM probe ≥ 20 GB for bf16, ≥ 12 GB for FP8.
- `MatrixGame3InteractivePipelineTests` should additionally require: a synthetic action sequence; skip on machines without GPU; longer timeout (multi-segment).
- Reference validation: dump latents at segment 0 from Python reference; compare in C# to within `1e-2` (bf16 + scheduler accumulation tolerates this).

### Reuse opportunities (the bulk of this is *not* net-new)

- **Wan2.2 DiT backbone** = reusable for any Wan2.2 / Wan-family pipeline (TI2V, T2V, I2V, V2V).
- **Wan2.2 VAE + MG-LightVAE** = shared with Lance, Wan2.2 standalone, future Wan2.5.
- **UMT5-XXL encoder** = shared with AuraFlow.
- **3D RoPE with split-axis dims** = applicable to Wan, LTX-Video, any 3D-position DiT.
- **`FlowUniPCMultistepScheduler`** = applicable to Wan2.1, Wan2.2, future flow-matching video models.
- **Camera-pose + Plücker utilities** = applicable to any camera-controlled video model (CameraCtrl, MotionCtrl, GameFactory).

**Genuinely net-new for Matrix-Game 3.0 alone:** the ActionModule block, the `select_memory_idx_fov` GPU kernel, and the streaming session loop with rolling buffer + 5-slot memory cache. Everything else is shared infrastructure.

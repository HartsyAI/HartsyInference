# Matrix-Game 2.0 — Research Notes

> Status: Complete (HF model card + GitHub source + paper + per-task configs captured; safetensors key dump still required) | Last Updated: 2026-05-24 | Needed Before: HartsyInference.World (Matrix-Game 2.0 pipeline, Phase 10)
> Source of truth: [SkyworkAI/Matrix-Game GitHub](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-2), [HF `Skywork/Matrix-Game-2.0`](https://huggingface.co/Skywork/Matrix-Game-2.0), [arXiv 2508.13009](https://arxiv.org/abs/2508.13009)
> License: **MIT** (confirmed on HF model card and on the Matrix-Game-2 GitHub README)
> Related: future `MATRIX_GAME_3_ARCHITECTURE.md` (the 5B sibling), [`FLOW_MATCHING_AUDIO.md`](FLOW_MATCHING_AUDIO.md) (flow-matching scheduler background), [`VAE_ARCHITECTURE.md`](VAE_ARCHITECTURE.md), [`FLUX_ARCHITECTURE.md`](FLUX_ARCHITECTURE.md) (AdaLN modulation lineage)

## Summary

**Matrix-Game 2.0** is Skywork AI's open-source 1.8B-parameter **interactive world model** — given a single reference image and a stream of keyboard + mouse actions, it autoregressively synthesizes a controllable video at **~25 FPS @ 352 × 640** on a single H100. It is the **entry-level / low-VRAM sibling** of the 5B Matrix-Game 3.0 (40 FPS @ 720p) and a successor to the 17B Matrix-Game 1.0 (offline only).

Architecturally, Matrix-Game 2.0 is **SkyReels-V2-I2V-1.3B-540P** (a Wan2.1 family 1.3B I2V DiT) with the text branch removed and per-block **ActionModules** added to the **first 15 of its 30 DiT layers** (the foundation model puts ActionModules in all 30 — the distilled checkpoints prune to layers 0–14). The "+0.5B" of new parameters is exactly those ActionModules (mouse MLP + self-attention + projection + keyboard MLP + cross-attention + projection per inserted block, totalling ~500M new weights).

Real-time streaming is achieved by **(a)** converting the bidirectional DiT into a **block-wise causal** transformer (each frame attends only to its own frame and a sliding window of the last 6 frames), **(b)** **DMD + Self-Forcing few-step distillation** of the original ~1000-step teacher into a **3- or 4-step student** (`denoising_step_list = [1000, 666, 333]` for universal/gta, `[1000, 750, 500, 250]` for templerun), and **(c)** **rolling KV cache** that evicts the oldest frame tokens once the local-attention window of 6 frames is exceeded.

The VAE is the **Wan2.1 3D causal VAE** (16 latent channels, **8 × 8 spatial / 4 × temporal** compression, 508 MB). Input image conditioning is dual: the input image is **VAE-encoded** (concatenated with the noisy latent along the channel dim → `in_dim = 36`) **and** passed through a frozen **CLIP-ViT-H/14 XLM-RoBERTa** encoder (4.77 GB) for "visual context" that is consumed by the I2V cross-attention. Action conditioning is **deterministic** (no learned router): every other denoising step, current-frame mouse deltas pass through an MLP and self-attention (acting like RoPE'd cross-attention over the spatial token grid) and keyboard one-hots pass through MLP + cross-attention.

For HartsyInference, Matrix-Game 2.0 is best treated as a **new `HartsyInference.World` package built on top of a Wan2.1-family DiT backbone** that will be shared with Matrix-Game 3.0 once we add it. The same DiT, RoPE, AdaLN, action-module, KV-cache, and Wan VAE primitives are reusable across both — only the backbone weights (1.3B vs 5B), action-block coverage (15 vs 30 blocks), and resolution (352×640 vs 1280×720) change.

## Detailed Findings

### 1. Family / variants

| Variant | Backbone | Params | Action coverage | Resolution | Steps | License | Use case |
|---|---|---|---|---|---|---|---|
| `Matrix-Game-1.0` | Custom 17B DiT | 17 B | all blocks | 720p offline | many | MIT | Original release, *not* real-time. arXiv 2506.18701, May 2025. |
| **`Matrix-Game-2.0`** | **SkyReels-V2-I2V-1.3B-540P (Wan2.1 I2V)** | **1.8 B** | **first 15 of 30 blocks** (foundation: all 30) | **352 × 640** | **3 or 4** | **MIT** | Real-time @ 25 FPS H100. |
| `Matrix-Game-3.0` | Wan2.2-TI2V-5B | 5 B | first 15 of N blocks | 720p | distilled (DMD) | MIT | 40 FPS @ 720p, adds long-horizon memory + frame self-correction. arXiv 2604.08995. |

Inside `Skywork/Matrix-Game-2.0` on HuggingFace, **four DiT checkpoints** ship, each tied to a config in the GitHub repo's `configs/distilled_model/`:

| Checkpoint folder | File | Size | Used for | Action shape |
|---|---|---|---|---|
| `base_model/` | `diffusion_pytorch_model.safetensors` | **3.65 GB** | The original 1.8B foundation model, action modules in *all 30* blocks. Used for fine-tuning or as a teacher. | 4-dim keyboard + 2-dim mouse |
| `base_distilled_model/` | `base_distill.safetensors` | **6.48 GB** | Distilled universal (post-DMD); inserts action modules in blocks 0–14 only; 3-step student. **This is the default `inference.py --config inference_universal.yaml` target.** | 4-dim keyboard + 2-dim mouse |
| `gta_distilled_model/` | `gta_keyboard2dim.safetensors` | **6.48 GB** | GTA-driving-specialized distillation; 3-step student. | 2-dim keyboard + 2-dim mouse |
| `templerun_distilled_model/` | `templerun_7dim_onlykey.safetensors` | **6.03 GB** | TempleRun-specialized; 4-step student; **keyboard-only** (mouse disabled). | 7-dim keyboard, no mouse |

> The distilled checkpoints are ~2× larger than the foundation model on disk because they keep the original Wan2.1 backbone in FP32 (`tensor_type: F32` per the SkyReels card) plus the action modules and the new distillation deltas; the foundation `base_model/` was apparently saved in FP16 (3.65 GB ≈ 1.8B × 2 B). Confirm with safetensors metadata at integration time — see Open Q 1.

Plus the always-present shared sub-models in the HF repo:

| File | Size | Role |
|---|---|---|
| `Wan2.1_VAE.pth` | **508 MB** | Wan2.1 3D causal VAE (PyTorch `.pth`, not safetensors). |
| `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` | **4.77 GB** | Frozen CLIP-ViT-H/14 with XLM-RoBERTa text tower; only the **image** encoder is used (image → 257-token visual context). |
| `xlm-roberta-large/` | — | XLM-RoBERTa tokenizer/weights (vestigial — text branch is *not* used by Matrix-Game 2.0; this is residual SkyReels-V2 baggage). |
| `architecture.png` | 414 kB | The pipeline diagram. |
| `model_index.json` | 46 B | `{"_class_name": "MatrixGame2I2VPipeline"}` — the diffusers `from_pretrained` entry point. |

**Repository total = 27.9 GB**. The `xlm-roberta-large/` weights and the CLIP text tower can be skipped on disk for a minimal HartsyInference install — we only need `Wan2.1_VAE.pth`, the CLIP image tower (~1.0 GB partial), and one distilled checkpoint.

### 2. Backbone — `CausalWanModel` (Wan2.1 1.3B I2V, text branch removed)

The exact `config.json` for the universal distilled checkpoint:

```json
{
  "_class_name": "CausalWanModel",
  "_diffusers_version": "0.35.0.dev0",
  "dim": 1536,
  "ffn_dim": 8960,
  "freq_dim": 256,
  "in_dim": 36,
  "out_dim": 16,
  "num_heads": 12,
  "num_layers": 30,
  "eps": 1e-06,
  "model_type": "i2v",
  "text_len": 512,
  "patch_size": [1, 2, 2],
  "local_attn_size": 6,
  "sink_size": 0,
  "inject_sample_info": false,
  "action_config": { ... see § 3 ... }
}
```

| Field | Value | Meaning |
|---|---|---|
| `dim` | **1536** | Transformer hidden dim. |
| `num_layers` | **30** | DiT blocks. |
| `num_heads` | **12** | Multi-head attention; `head_dim = 1536 / 12 = 128`. |
| `ffn_dim` | **8960** | FFN inner dim (~5.83× hidden — standard Wan ratio). |
| `freq_dim` | **256** | Sinusoidal frequency dim for timestep embedding (before MLP → `dim*6` for AdaLN). |
| `in_dim` | **36** | Patch-embed input channels = `16 (noisy latent) + 16 (img_cond latent) + 4 (mask)` (concat-condition I2V). |
| `out_dim` | **16** | Output channels = VAE latent channel count. |
| `patch_size` | **(1, 2, 2)** | (T, H, W) patchify; 3D `Conv3d(36, 1536, kernel=(1,2,2), stride=(1,2,2))`. |
| `text_len` | **512** | Vestigial Wan2.1 field — text is removed in Matrix-Game-2 so no real effect; cross-attention now consumes the 257-token CLIP visual context. |
| `local_attn_size` | **6** (universal/templerun), **4** (gta_drive), **-1** (foundation) | Sliding causal-attention window measured in **latent frames**. -1 means full causal. |
| `sink_size` | **0** | No StreamingLLM-style attention sink. |
| `eps` | **1e-6** | RMSNorm / LayerNorm epsilon. |
| `inject_sample_info` | **false** | Don't add per-sample auxiliary embeddings. |
| `model_type` | `"i2v"` | Selects `WanI2VCrossAttention` (image-conditioned) rather than the T2V cross-attn. |

**Resulting block:**
- `WanLayerNorm(1536)` → `WanSelfAttention(1536, heads=12, qk_norm=True, RoPE)` → residual
- `WanLayerNorm(1536)` (when `cross_attn_norm=True`) → `WanI2VCrossAttention(q=hidden, kv=257-token CLIP visual_context)` → residual
- **`ActionModule(...)`** (only present in blocks ∈ `action_config["blocks"]`) → adds **two** more residuals: mouse self-attention residual + keyboard cross-attention residual (see § 3)
- `WanLayerNorm(1536)` → `FFN(Linear 1536→8960, GELU, Linear 8960→1536)` → residual
- AdaLN modulation: a learnable parameter `modulation` of shape **`[1, 6, dim]`** is added to a timestep-derived 6-way (scale-shift × 3 ops: self-attn pre, cross-attn pre, FFN pre) modulation. The timestep MLP is `Linear(256→1536) → SiLU → Linear(1536→1536*6)`.

**Final head:** `Head(dim=1536, out_dim=16, patch_size=(1,2,2))` = `WanLayerNorm + Linear(1536 → 1*2*2*16 = 64)` with its own 2-way `modulation[1, 2, dim]` (final scale-shift). Output is unpatchified to `(B, 16, T, H_lat, W_lat)`.

**Self-attention RoPE.** `causal_rope_apply()` uses the standard Wan 3D RoPE: head_dim is split into `[temporal, height, width]` rotary slices. From `WanModel`: `rope_params(1024, d - 4 * (d // 6))` is precomputed for the temporal axis; the remainder is split between H and W. The exact split for `head_dim=128` is **(t=44, h=42, w=42)** — to verify with a key dump (Open Q 3).

**QK-norm.** `qk_norm=True` ⇒ `WanRMSNorm(head_dim=128, eps=1e-6)` on both Q and K before flash attention. **Required** — the released weights include these norms.

### 3. ActionModule — mouse + keyboard injection

Per-block ActionModule constructor and forward live in `wan/modules/action_module.py`. Released `action_config` (universal distilled):

```json
{
  "blocks": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
  "enable_keyboard": true,
  "enable_mouse": true,
  "heads_num": 16,
  "hidden_size": 128,
  "img_hidden_size": 1536,
  "keyboard_dim_in": 4,
  "keyboard_hidden_dim": 1024,
  "mouse_dim_in": 2,
  "mouse_hidden_dim": 1024,
  "mouse_qk_dim_list": [8, 28, 28],
  "patch_size": [1, 2, 2],
  "qk_norm": true,
  "qkv_bias": false,
  "rope_dim_list": [8, 28, 28],
  "rope_theta": 256,
  "vae_time_compression_ratio": 4,
  "windows_size": 3
}
```

Per-variant action-config diffs:

| Field | universal | gta_drive | templerun | foundation |
|---|---|---|---|---|
| `blocks` | 0..14 | 0..14 | 0..14 | **0..29 (all)** |
| `enable_mouse` | true | true | **false** | true |
| `keyboard_dim_in` | **4** (fwd, back, left, right) | **2** (fwd, back) | **7** (nomove, jump, slide, turnleft, turnright, leftside, rightside) | 4 |
| `mouse_dim_in` | 2 | 2 | n/a | 2 |
| `local_attn_size` (top-level, not in action_config) | 6 | **4** | 6 | -1 |

Constants identical across variants:
- `heads_num = 16` (action module heads — **not** the main 12-head DiT)
- `hidden_size = 128` (keyboard embed dim before MLP expansion to 1024)
- `mouse_hidden_dim = keyboard_hidden_dim = 1024` (action stream hidden); `head_dim = 1024 / 16 = 64`
- `img_hidden_size = 1536` (matches DiT `dim`)
- `windows_size = 3` (number of past **latent** frames concatenated to current mouse/keyboard for context)
- `vae_time_compression_ratio = 4` (Wan VAE temporal stride — so `windows_size=3` latent frames = 12 input frames of context)
- `rope_dim_list = mouse_qk_dim_list = [8, 28, 28]` — split of the **64-dim action head** into (T, H, W) RoPE axes. Sum = 64. ✓
- `rope_theta = 256` — note: **much smaller** than the Wan main `rope_theta` (effectively the standard 10000); chosen so the action RoPE fits the relatively short temporal axis.
- `patch_size = (1, 2, 2)` — matches DiT patchify.
- `qk_norm = True`, `qkv_bias = False`.

#### Mouse path

Input: `mouse_condition` of shape `(B, N_frames, 2)` — continuous **(pitch_delta, yaw_delta)** floats. `N_frames = (T_latent − 1) × 4 + 1` where `T_latent` = number of latent frames. Universal/GTA range is `±0.1` per the `CAM_VALUE = 0.1` constant in `Bench_actions_universal()`.

Per ActionModule block:

```text
1. Left-pad mouse_condition with the first frame (pad_t = vae_time_compression_ratio * windows_size = 12 frames).
2. For each latent frame i in [0, T_latent): group_mouse[i] = mouse_condition[ 4*(i - windows_size) + pad_t : 4*i + pad_t ]   # window of 4*3 = 12 raw mouse vectors
   → shape (B, T_latent, 12, 2)
3. Broadcast over spatial tokens S = H_lat × W_lat = 44 × 80 = 3520, but actually the code uses the post-patch grid 22 × 40 = 880.
   group_mouse → reshape to (B*S, T_latent, pad_t*C) = (B*880, T_latent, 24)
4. Concatenate with the patchified hidden states (per spatial token, per latent frame, the current 1536-dim feature):
   group_mouse = cat([hidden_states (1536), group_mouse (24)], dim=-1)  →  (B*880, T_latent, 1560)
5. mouse_mlp = Linear(1560 → 1024) → GELU(tanh) → Linear(1024 → 1024) → LayerNorm(1024)
6. t_qkv = Linear(1024 → 1024*3) → split → QKV
7. qk_norm = WanRMSNorm(head_dim=64) on Q and K
8. apply_rotary_emb(Q, K, freqs_cis from rope_dim_list=[8,28,28], theta=256, start_offset=start_frame)
9. flash_attn_func(Q, K, V) over the *temporal* axis (per spatial token; "self-attention" across latent frames)
10. Output projection: proj_mouse = Linear(1024 → 1536)
11. hidden_states = hidden_states + attn   # residual into the main DiT stream
```

So **mouse acts as a per-spatial-token temporal self-attention** that mixes the last `windows_size=3` latent frames of mouse signal into each token's representation. Note this is **NOT** classic cross-attention — Q/K/V all come from the fused (hidden_state + mouse) representation. The mouse signal is "injected" by concatenation into the MLP input before QKV.

#### Keyboard path

Input: `keyboard_condition` of shape `(B, N_frames, keyboard_dim_in)` — one-hot encoding of the keys held that frame.

Per ActionModule block:

```text
1. Left-pad with first-frame copy (pad_t = 12 frames).
2. keyboard_embed = nn.Sequential(Linear(K → 128), SiLU, Linear(128 → 128))   # K = 4/2/7
3. For each latent frame i: group_keyboard[i] = keyboard_embedded[ 4*(i-windows_size)+pad_t : 4*i+pad_t ]
   → shape (B, T_latent, 12, 128) → flatten last two: (B, T_latent, 1536)   # 12*128 = 1536
4. mouse_attn_q = Linear(1536 → 1024)       # Q from current hidden_states (full spatial sequence)
5. keyboard_attn_kv = Linear(1536 → 2048)   # K,V from grouped keyboard embedding (split → K, V each 1024)
6. q.view(B, L, heads=16, D=64), k.view(B, T_latent, heads=16, D=64), v same.
7. qk_norm = WanRMSNorm(64).
8. (use_rope_keyboard=True branch ALWAYS active.) Reshape Q to per-spatial-token sequence (B*880, T_latent, 16, 64); broadcast K, V across S.
9. apply_rotary_emb(Q, K, freqs_cis, start_offset=start_frame).
10. flash_attn_func(Q, K, V) — cross-attention from per-token current frame to *time* axis of grouped keyboard embedding.
11. proj_keyboard = Linear(1024 → 1536)
12. hidden_states = hidden_states + attn   # residual
```

Keyboard is genuine **cross-attention**: Q comes from the DiT hidden state, K/V come from the keyboard MLP output. RoPE is applied on the temporal axis on both sides so the keyboard context's frame index aligns with the DiT's current frame index.

**Net per-action-block cost.** Each ActionModule block adds ~(2 × mlp + 4 × linear + 2 × flash-attn) work. Roughly ~17M params per block × 15 blocks ≈ **~255M params** for the universal distilled action stack (rough). Combined with the unchanged 1.3B Wan backbone → the marketed 1.8B total. (Verify exact via key dump — Open Q 1.)

### 4. Wan2.1 VAE — 16-channel 3D causal VAE

Reference: `wan/vae/wanx_vae.py` and `wan/vae/wanx_vae_src/` (mirror of Wan2.1's official VAE).

| Field | Value |
|---|---|
| Latent channels (`z_channels`) | **16** |
| Spatial downsample | **8×** (each axis) |
| Temporal downsample | **4×** (`vae_time_compression_ratio = 4`) |
| Conv kernel type | `CausalConv3d` (Wan family) — frame cache for streaming decode |
| Norm | RMSNorm |
| Storage | `Wan2.1_VAE.pth` = **508 MB** (FP32) |
| Tiling defaults | `tile_size=[44, 80]`, `tile_stride=[23, 38]` (in **latent** spatial cells, so 44×8=352 px and 80×8=640 px tiles) |
| Per-channel mean/std normalization | Yes, 16-vector each (from `WanVAEWrapper`). Exact values to be dumped — see Open Q 4. |

**Resolution example (the universal mode):**
- Output video size: **352 × 640 px** (H × W). 25 FPS.
- Latent spatial: `352/8 × 640/8 = 44 × 80 = 3520`. After DiT patchify with stride 2 in H,W: `22 × 40 = 880 = "S"`. That's the **"880" magic number** all over the action module code (`assert S == 880`).
- 1 latent frame ↔ 4 raw frames. Distilled checkpoint defaults to `image_or_video_shape = [1, 16, 15, 44, 80]` → **15 latent frames = 57 raw frames** (the "57 = 15*4 - 3" arithmetic is actually `(15-1)*4 + 1 = 57` because the first latent encodes a single static frame, and each subsequent latent encodes 4 new frames).
- For long video synthesis: the pipeline does **autoregressive block generation**; each call to `pipeline.inference(...)` generates `num_frame_per_block = 3` new latent frames, then the rolling KV cache shifts.

**VAE pre-processing:** input image at the original resolution → `Resize(352, 640) → ToTensor → Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])`. Then VAE-encoded with tiling.

**Image conditioning latent build** (from `inference.py`):
```python
image = preprocess(input_image)                                # (1, 3, 1, 352, 640)
padding_video = zeros_like(image).repeat(1, 1, 4*(N_lat-1), 1, 1)   # rest are black
img_cond_pixels = cat([image, padding_video], dim=2)           # (1, 3, 1 + 4*(N_lat-1), 352, 640)
img_cond_latent = vae.encode(img_cond_pixels, tile=True)       # (1, 16, N_lat, 44, 80)
mask_cond = ones_like(img_cond_latent); mask_cond[:, :, 1:] = 0  # mask=1 for frame 0, mask=0 for the rest
cond_concat = cat([mask_cond[:, :4], img_cond_latent], dim=1)  # (1, 4+16=20, N_lat, 44, 80)
```
So the **20-channel** condition tensor concat with the 16-channel noisy latent → `in_dim = 36` for the patch embed. Mask channel uses only the first 4 of the broadcast 16-channel mask (presumably arbitrary — they just need ≥1 channel to flag "this is frame 0").

**CLIP visual context:** `visual_context = self.vae.clip.encode_video(image)` — runs the input image through CLIP-ViT-H/14, returns a `(1, 257, 1280)` token sequence (CLS + 16×16 = 256 patches) projected to 1280-dim. This is the K/V source for `WanI2VCrossAttention` and is **cached once** per generation (held in `crossattn_cache[layer]['k_cache' / 'v_cache']` of shape `(1, 257, 12, 128)`).

### 5. Scheduler — FlowMatch + DMD-Distilled Few-Step

**Teacher** (the un-distilled Wan2.1 base): standard `FlowMatchScheduler` with logit-normal time shift, **~1000 inference steps**.

**Student** (the distilled checkpoints): the same `FlowMatchScheduler` but with a fixed discrete `denoising_step_list`. Inference loop:

```python
for step_idx, denoise_t in enumerate(denoising_step_list):
    if warp_denoising_step:
        t = scheduler.timesteps[1000 - denoise_t]   # warp into the canonical 1000-step schedule
    else:
        t = denoise_t
    pred = generator(noisy_latents, t, conditional_dict, ...)
    noisy_latents = scheduler.add_noise(pred, noise, next_t)   # for non-final steps
return pred  # last step's clean prediction
```

Exact `denoising_step_list`:

| Mode | Step list | # steps |
|---|---|---|
| **universal** | `[1000, 666, 333]` | **3** |
| **gta_drive** | `[1000, 666, 333]` | **3** |
| **templerun** | `[1000, 750, 500, 250]` | **4** |

Common scheduler settings (from `inference_yaml/*.yaml`):

| Key | Value |
|---|---|
| `warp_denoising_step` | `true` |
| `ts_schedule` | `false` |
| `mixed_precision` | `true` (bfloat16 weights + fp16 VAE) |
| `seed` | `42` |
| `image_or_video_shape` | `[1, 16, 15, 44, 80]` — `(B, C_lat, T_lat, H_lat, W_lat)` |
| `num_frame_per_block` | **3** — autoregressive block size, in latent frames. |
| `context_noise` | `0` |
| `causal` | `true` |
| **`timestep_shift`** | **5.0** (`model_kwargs.timestep_shift`) — applied to the logit-normal time shift in the FlowMatchScheduler. |

**Distillation pedigree (from the paper):**
- **Phase 1 — ODE init.** 40k (t_high, t_low) pairs along the teacher's ODE trajectory; 6k fine-tune steps at `lr=6e-6` to teach the student to do a single ODE leap.
- **Phase 2 — DMD with Self-Forcing.** 4k steps. The student is wrapped in a Self-Forcing loop (it conditions on its own previously generated frames inside the same training rollout) and minimizes Distribution-Matching Distillation loss (Eq. 5 in the paper):
  ```
  L_student = E[ || G_phi({x_t^i}, {c^i}, {t^i}) − {x_0^i} ||² ]
  ```
  where the inner generator outputs are scored against the teacher's distribution via a separately trained critic.
- Step-count progression in the paper: distilled first to **4 steps**, then further down to **3 steps** in deployment.

**Implication for HartsyInference:** we do NOT need to implement DMD training. We only need the discrete few-step Flow-Match inference loop with the warp-step remapping. This is ~50 lines of pipeline code.

### 6. Causal attention + rolling KV cache

Key design ideas (from `wan/modules/causal_model.py` and `pipeline/causal_inference.py`):

**Block-causal mask.** Frames are divided into blocks of `num_frame_per_block = 3` latent frames. Inside a block: bidirectional attention. Across blocks: causal — block k attends to blocks 0..k. Implemented via `flex_attention` with a precomputed `block_mask`.

**Local attention window.** `local_attn_size = 6` latent frames (universal/templerun) or `4` (gta_drive). Beyond the window, frames cannot attend. Combined with `num_frame_per_block = 3`, this means **a new frame block sees the previous 6/4 latent frames of context**, equivalent to ~24/16 raw frames at 25 FPS = ~1.0 / ~0.65 seconds of past history.

**KV cache layout** (per layer, per modality):

| Cache | Shape | Heads × Dim | Notes |
|---|---|---|---|
| `kv_cache1[layer]` (main self-attn) | `(B, local_attn_size * frame_seq_length, 12, 128)` = `(1, 6 × 880, 12, 128) = (1, 5280, 12, 128)` | 12 × 128 | The main DiT self-attention K and V buffer. `frame_seq_length = H_lat/2 × W_lat/2 = 22 × 40 = 880`. |
| `kv_cache_mouse[layer]` (mouse self-attn) | `(B * 880, kv_cache_size, 16, 64)` | 16 × 64 | Per-spatial-token mouse temporal cache; scales with full sequence length. |
| `kv_cache_keyboard[layer]` (keyboard cross-attn) | `(B, kv_cache_size, 16, 64)` | 16 × 64 | Single batch keyboard cache; broadcast over spatial tokens at attention time. |
| `crossattn_cache[layer]` (CLIP I2V) | `(B, 257, 12, 128)` | 12 × 128 | Fixed-size; populated once on the first call and reused. |

Each cache tracks two indices: `global_end_index` (logical absolute position in the full video timeline) and `local_end_index` (physical position inside the ring buffer).

**Eviction logic** (literal code in `action_module.py` and equivalent in `causal_model.py`):
```python
if current_end > global_end_index and num_new + local_end_index > kv_cache_size:
    num_evicted = num_new + local_end_index - kv_cache_size
    num_rolled  = local_end_index - num_evicted - sink_tokens
    cache[k/v][:, sink:sink+num_rolled] = cache[k/v][:, sink+num_evicted:sink+num_evicted+num_rolled].clone()
    local_end_index = local_end_index + (current_end - global_end_index) - num_evicted
local_start = local_end_index - num_new
cache[k/v][:, local_start:local_end_index] = k_or_v_new
```
This is a **rolling buffer** — once full, the oldest non-sink frames are memcpy'd forward and the new frames overwrite the tail. `sink_size = 0` in all released configs, so no StreamingLLM-style sink is used. Attention always looks at `cache[k/v][:, max(0, local_end - local_attn_size):local_end]` — a sliding window of exactly `local_attn_size` latent frames.

### 7. Inference pipeline at a glance

From `pipeline/causal_inference.py` (`CausalInferencePipeline` and `CausalInferenceStreamingPipeline`):

```text
1. Setup: load distilled DiT, Wan VAE decoder (fp16, torch.compile), Wan VAE encoder + CLIP, scheduler.
2. Preprocess first frame: resize→352×640, normalize→[-1,1], vae.encode → 16-ch latent at (1, 1, 44, 80).
3. Build cond_concat = [mask(4-ch) | image_cond_latent_broadcast(16-ch)] → 20 channels, length N_lat.
4. Compute visual_context = CLIP(first_frame) → (1, 257, 1280). Cache it.
5. Sample noise: shape (1, 16, N_lat, 44, 80) at bfloat16.
6. Build the per-frame mouse/keyboard tensors from user actions (or Bench_actions_* for benchmarks).
7. Initialize all KV caches to zeros, indices to 0.
8. For each output block of 3 latent frames:
     for step_t in denoising_step_list:                   # 3 or 4 steps
         pred, x0 = generator(noisy_block, t=step_t, cond_concat_block, visual_context,
                              mouse_cond, keyboard_cond,
                              kv_cache=kv_cache1, kv_cache_mouse=..., kv_cache_keyboard=...,
                              crossattn_cache=crossattn_cache,
                              current_start=block_start_token,
                              cache_start=...)
         if not final step:
             noisy_block = scheduler.add_noise(x0, fresh_noise, next_t)
     latents_collected.append(x0)
     advance KV cache indices by num_frame_per_block * frame_seq_length
9. videos = vae.decode(cat(latents_collected, dim=2))    # tiled, fp16, torch.compile'd
10. Convert to uint8, save MP4 with action overlay.
```

The **streaming** variant (`CausalInferenceStreamingPipeline`) interleaves step 8 with `get_current_action()` calls — between blocks, it polls user input (or reads from a queue) and re-builds `mouse_cond` / `keyboard_cond` slices for the next block. The KV caches are preserved across blocks; only the DiT forward + VAE decode of the new block are computed each tick.

**Real-time budget @ 25 FPS:**
- 25 FPS × 4× temporal compression = **6.25 latent frames/s** generated.
- `num_frame_per_block = 3` latent frames per inference call ⇒ **2.08 inference calls/s** ⇒ **~480 ms per block**.
- Per block: 3 forward passes (universal) of the 1.8B model × ~150 ms each ≈ 450 ms, + VAE decode for `3×4=12` raw frames ≈ 30 ms. On H100 SXM, these numbers fit comfortably.

### 8. Differences from Matrix-Game 1.0 and 3.0

| Aspect | Matrix-Game 1.0 | **Matrix-Game 2.0** | Matrix-Game 3.0 |
|---|---|---|---|
| Backbone | Custom 17B DiT | **Wan2.1 1.3B I2V (SkyReels-V2-I2V-1.3B-540P)** | Wan2.2 5B TI2V |
| Total params | 17 B | **1.8 B** | 5 B |
| Real-time | No | **Yes — 25 FPS @ 352×640 on H100** | Yes — 40 FPS @ 720p on H100 |
| Inference steps | Many | **3 (or 4)** via DMD + Self-Forcing | Few-step DMD + multi-segment AR distillation |
| Action coverage in DiT | All blocks | **First 15 of 30 blocks** (distilled); all 30 (foundation) | First 15 blocks (same scheme as 2.0) |
| Long-horizon | Limited | **Local 6-frame KV window only** | **Frame self-correction during training + camera-aware memory retrieval** |
| Action shape | 6-dim keyboard | **4 / 2 / 7-dim per variant** | Similar to 2.0 + camera-pose conditioning |
| VAE | Wan2.1 | **Wan2.1** (508 MB) | Wan2.2 (different latent stats; 3D causal) |
| Image cond | Image + text | **Image-only** (text branch removed); CLIP-ViT-H/14 visual context | Image + camera + (optional text) |
| Paper | arXiv 2506.18701 | **arXiv 2508.13009** (Aug 2025) | arXiv 2604.08995 (Apr 2026) |
| License | MIT | **MIT** | MIT |

**Reusable components between 2.0 and 3.0 in HartsyInference:**
- ActionModule (identical class — only `keyboard_dim_in`, `mouse_dim_in` and `blocks` change).
- `CausalWanModel` core: AdaLN, RoPE, qk-norm, flash-attn forward, FFN.
- Rolling KV-cache primitive.
- DMD few-step FlowMatch scheduler with `warp_denoising_step` and `timestep_shift`.
- CLIP-ViT-H/14 visual context encoder.
- Wan2.1 VAE (for 2.0) vs Wan2.2 VAE (for 3.0) — different latent channels (16 vs ~48); design `IWanVae` as a polymorphic interface.

**Diverging in 3.0 (do NOT plan for in the 2.0 build):**
- Long-horizon memory retrieval mechanism (camera-aware).
- Self-correcting prediction-residual training.
- Different action dim for camera pose.

## Key Numbers / Constants

| Constant | Value | Where used |
|---|---|---|
| `dim` | **1536** | DiT hidden |
| `num_layers` | **30** | DiT depth |
| `num_heads` | **12** | head_dim = 128 |
| `ffn_dim` | **8960** | FFN inner |
| `in_dim` | **36** | patch_embed in_channels (16 noisy + 16 img_cond + 4 mask) |
| `out_dim` | **16** | VAE latent channels |
| `patch_size` | **(1, 2, 2)** | TxHxW patchify |
| `freq_dim` | **256** | timestep sin freq |
| `eps` | **1e-6** | norms |
| `text_len` | **512** | vestigial (text branch removed) |
| `local_attn_size` | **6** (uni/templerun), **4** (gta), **-1** (foundation) | sliding window in latent frames |
| `sink_size` | **0** | no attention sink |
| `cross_attn_norm` | **true** | LayerNorm before CLIP cross-attn |
| `qk_norm` | **true** | WanRMSNorm(head_dim) on Q,K |
| `num_frame_per_block` | **3** | AR block size, latent frames |
| `vae_time_compression_ratio` | **4** | Wan VAE temporal stride |
| `windows_size` (action) | **3** | past latent frames mixed into action MLP |
| `pad_t` (action) | **12** | = 4 × 3 = pre-window pad |
| Action `heads_num` | **16** | action stream attention heads |
| Action `head_dim` | **64** | = 1024 / 16 |
| Action `hidden_size` | **128** | keyboard embed intermediate |
| Action `mouse_hidden_dim` | **1024** | mouse stream hidden |
| Action `keyboard_hidden_dim` | **1024** | keyboard stream hidden |
| `img_hidden_size` | **1536** | matches DiT dim |
| `rope_dim_list` (action) | **[8, 28, 28]** | action RoPE T/H/W split, sum=64 |
| `rope_theta` (action) | **256** | action RoPE base |
| `mouse_dim_in` | **2** (universal/gta) | (pitch_delta, yaw_delta) |
| `keyboard_dim_in` | **4 / 2 / 7** | universal / gta / templerun |
| `CAM_VALUE` | **0.1** | mouse delta magnitude in benchmark configs |
| **VAE z_channels** | **16** | latent feature dim |
| **VAE spatial downsample** | **8×** | each of H, W |
| **VAE temporal downsample** | **4×** | T |
| VAE tile_size | **[44, 80]** | latent cells = 352×640 px |
| VAE tile_stride | **[23, 38]** | latent cells |
| CLIP visual context tokens | **257** | CLS + 16×16 |
| CLIP visual context dim | **1280** | ViT-H/14 |
| `frame_seq_length` | **880** | post-patch spatial tokens (22×40) |
| `image_or_video_shape` | **[1, 16, 15, 44, 80]** | default video latent shape |
| `denoising_step_list` (uni/gta) | **[1000, 666, 333]** | 3 steps |
| `denoising_step_list` (templerun) | **[1000, 750, 500, 250]** | 4 steps |
| `warp_denoising_step` | **true** | remap into 1000-step scheduler |
| `timestep_shift` | **5.0** | logit-normal shift in FlowMatchScheduler |
| Output resolution | **352 × 640** | H × W px |
| Output FPS | **25** | on H100 |
| Per-block budget | **~480 ms** | for 25 FPS @ block_size=3 latent frames |
| Min VRAM (24 GB) | per README | NVIDIA card requirement |
| Repo total size | **27.9 GB** | full HF download |
| Base teacher size | **3.65 GB** | `base_model/diffusion_pytorch_model.safetensors` |
| Distilled checkpoint size | **6.48 GB** (universal/gta) / **6.03 GB** (templerun) | safetensors |
| Wan2.1 VAE | **508 MB** | `Wan2.1_VAE.pth` |
| CLIP-ViT-H/14 | **4.77 GB** | `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` |
| Training (distillation) batch size | **256** | from paper |
| Training (distillation) lr | **2e-5** primary, **6e-6** ODE init | from paper |
| Distillation steps | **120 k** primary | from paper |
| Interactive video data | **~1200 hours** | UE + GTA5 from paper |

## Data Layouts / Formats

### HuggingFace `Skywork/Matrix-Game-2.0` file tree

```
Skywork/Matrix-Game-2.0/                                                          27.9 GB total
├── .gitattributes                                                                1.64 kB
├── README.md                                                                     5.63 kB
├── architecture.png                                                              414 kB
├── model_index.json                                                                46 B    # {"_class_name":"MatrixGame2I2VPipeline"}
├── Wan2.1_VAE.pth                                                                508 MB    # PyTorch .pth (NOT safetensors)
├── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth                      4.77 GB    # CLIP image+text tower
├── xlm-roberta-large/                                                               —      # vestigial text tokenizer; can be ignored for inference
├── base_model/                                                                   3.65 GB    # original 1.8B foundation (action_blocks=0..29)
│   ├── base_config.json                                                          972 B
│   └── diffusion_pytorch_model.safetensors                                      3.65 GB
├── base_distilled_model/                                                         6.48 GB    # universal distilled student (3 steps, action_blocks=0..14)
│   ├── config.json                                                               912 B
│   └── base_distill.safetensors                                                 6.48 GB
├── gta_distilled_model/                                                          6.48 GB    # GTA-driving distilled student
│   ├── config.json                                                               912 B
│   └── gta_keyboard2dim.safetensors                                             6.48 GB
└── templerun_distilled_model/                                                    6.03 GB    # TempleRun distilled student (4 steps, mouse=off)
    ├── config.json                                                               851 B
    └── templerun_7dim_onlykey.safetensors                                       6.03 GB
```

### GitHub `SkyworkAI/Matrix-Game/Matrix-Game-2/` code layout

```
Matrix-Game-2/
├── README.md
├── inference.py                       # Bench_actions_* driver
├── inference_streaming.py             # interactive driver
├── setup.py
├── requirements.txt
├── assets/
├── configs/
│   ├── distilled_model/
│   │   ├── universal/config.json
│   │   ├── gta_drive/config.json
│   │   └── templerun/config.json
│   ├── foundation_model/
│   │   └── config.json                # action_blocks 0..29, local_attn_size=-1
│   └── inference_yaml/
│       ├── inference_universal.yaml
│       ├── inference_gta_drive.yaml
│       └── inference_templerun.yaml
├── demo_images/{universal,gta_drive,templerun}/*.png
├── demo_utils/
│   └── vae_block3.py                  # VAEDecoderWrapper
├── pipeline/
│   ├── __init__.py
│   └── causal_inference.py            # CausalInferencePipeline + CausalInferenceStreamingPipeline
├── utils/
│   ├── conditions.py                  # Bench_actions_universal / _gta_drive / _templerun + combine_data
│   ├── visualize.py
│   ├── misc.py
│   └── wan_wrapper.py                 # WanDiffusionWrapper, WanVAEWrapper
└── wan/                                # forked from Wan-Video/Wan2.1 (Apache 2.0)
    ├── __init__.py
    ├── image2video.py
    ├── text2video.py
    ├── configs/                       # incl. wan_i2v_14B.py (the 14B unused here; 1.3B variant lives in SkyReels)
    ├── distributed/
    ├── modules/
    │   ├── action_module.py           # ★ ActionModule
    │   ├── attention.py
    │   ├── causal_model.py            # ★ CausalWanModel, CausalWanAttentionBlock, CausalWanSelfAttention
    │   ├── clip.py                    # CLIP-ViT-H/14 + XLM-RoBERTa wrapper
    │   ├── model.py                   # original WanModel (non-causal)
    │   ├── posemb_layers.py           # get_nd_rotary_pos_embed, apply_rotary_emb
    │   ├── t5.py                      # vestigial
    │   ├── tokenizers.py
    │   ├── vae.py
    │   └── xlm_roberta.py
    ├── utils/
    └── vae/
        ├── wanx_vae.py                # WanxVAEWrapper, get_wanx_vae_wrapper
        ├── wrapper.py
        └── wanx_vae_src/              # Wan2.1 VAE internals
```

### Expected tensor key prefixes inside the distilled `.safetensors`

Based on `CausalWanModel` module hierarchy (must be confirmed via key-dump — Open Q 1):

```
patch_embedding.weight  patch_embedding.bias                  # Conv3d(36, 1536, (1,2,2))
time_embedding.0.weight / .1.weight / .bias                   # Linear(256→1536) → SiLU → Linear(1536→1536)
time_projection.weight / .bias                                # Linear(1536→1536*6)
blocks.{0..29}.modulation                                     # [1, 6, 1536]
blocks.{0..29}.norm1.weight  norm1.bias                        # WanLayerNorm
blocks.{0..29}.self_attn.{q,k,v,o}.weight                      # 1536→1536
blocks.{0..29}.self_attn.{norm_q,norm_k}.weight                # WanRMSNorm(128)
blocks.{0..29}.norm3.weight  norm3.bias                        # cross_attn pre-norm
blocks.{0..29}.cross_attn.{q,k,v,o}.weight                     # I2V cross-attn
blocks.{0..29}.cross_attn.{norm_q,norm_k}.weight               # qk-norm
blocks.{0..29}.norm2.weight  norm2.bias                        # FFN pre-norm
blocks.{0..29}.ffn.0.weight  ffn.0.bias                        # Linear(1536→8960)
blocks.{0..29}.ffn.2.weight  ffn.2.bias                        # Linear(8960→1536)
blocks.{0..14}.action_model.keyboard_embed.0.{weight,bias}     # Linear(K→128)
blocks.{0..14}.action_model.keyboard_embed.2.{weight,bias}     # Linear(128→128)
blocks.{0..14}.action_model.mouse_mlp.0.{weight,bias}          # Linear(1560→1024)
blocks.{0..14}.action_model.mouse_mlp.2.{weight,bias}          # Linear(1024→1024)
blocks.{0..14}.action_model.mouse_mlp.3.{weight,bias}          # LayerNorm(1024)
blocks.{0..14}.action_model.t_qkv.weight                       # Linear(1024→3072)
blocks.{0..14}.action_model.{img_attn_q_norm,img_attn_k_norm}.weight  # WanRMSNorm(64)
blocks.{0..14}.action_model.proj_mouse.weight                  # Linear(1024→1536)
blocks.{0..14}.action_model.mouse_attn_q.weight                # Linear(1536→1024)  (used by keyboard path)
blocks.{0..14}.action_model.keyboard_attn_kv.weight            # Linear(1536→2048)
blocks.{0..14}.action_model.{key_attn_q_norm,key_attn_k_norm}.weight  # WanRMSNorm(64)
blocks.{0..14}.action_model.proj_keyboard.weight               # Linear(1024→1536)
head.norm.weight  head.norm.bias                                # WanLayerNorm
head.head.weight  head.head.bias                                # Linear(1536, 1*2*2*16=64)
head.modulation                                                 # [1, 2, 1536]
```

(For `base_model/` / foundation, `blocks.{0..29}.action_model.*` for ALL 30 blocks.)

### Action tensor shapes

| Tensor | Shape | dtype |
|---|---|---|
| `mouse_condition` (universal) | `(1, N_frames, 2)` | bfloat16 |
| `keyboard_condition` (universal) | `(1, N_frames, 4)` | bfloat16 |
| `keyboard_condition` (gta_drive) | `(1, N_frames, 2)` | bfloat16 |
| `mouse_condition` (gta_drive) | `(1, N_frames, 2)` | bfloat16 |
| `keyboard_condition` (templerun) | `(1, N_frames, 7)` | bfloat16 |
| (templerun mouse_condition) | n/a | — |

`N_frames = (T_latent - 1) * 4 + 1`. For default `T_latent = 15`, `N_frames = 57`. **`N_frames % 4 == 1`** is asserted.

### Keyboard index maps

```python
# universal
KEYBOARD_IDX = { "forward": 0, "back": 1, "left": 2, "right": 3 }
# gta_drive
KEYBOARD_IDX = { "forward": 0, "back": 1 }
# templerun
KEYBOARD_IDX = { "nomove": 0, "jump": 1, "slide": 2,
                 "turnleft": 3, "turnright": 4,
                 "leftside": 5, "rightside": 6 }
```

### Mouse camera-delta map (universal/gta)

```python
CAM_VALUE = 0.1
CAMERA_VALUE_MAP = {
    "camera_up":   [ CAM_VALUE, 0           ],
    "camera_down": [-CAM_VALUE, 0           ],
    "camera_l":    [ 0,        -CAM_VALUE   ],
    "camera_r":    [ 0,         CAM_VALUE   ],
    "camera_ur":   [ CAM_VALUE, CAM_VALUE   ],
    "camera_ul":   [ CAM_VALUE,-CAM_VALUE   ],
    "camera_dr":   [-CAM_VALUE, CAM_VALUE   ],
    "camera_dl":   [-CAM_VALUE,-CAM_VALUE   ],
}
```

Convention: mouse channel-0 = pitch delta (vertical), channel-1 = yaw delta (horizontal). Sign convention: positive yaw = right, positive pitch = up. Magnitude `0.1` per latent frame is "comfortable speed" — the model accepts continuous floats so user UIs can scale this to taste, but training data is centered around `±0.1`.

## Algorithm Steps

### Image-to-controllable-video (one block of `num_frame_per_block = 3` latent frames)

```text
Inputs:
    noisy_latents  : (1, 16, 3, 44, 80)  bfloat16
    cond_concat    : (1, 20, 3, 44, 80)  bfloat16   (4-ch mask + 16-ch img-cond, sliced to current block)
    visual_context : (1, 257, 1280)      bfloat16   (CLIP, cached)
    mouse_cond     : (1, 13, 2)          bfloat16   (per-frame mouse, full block range, pre-padded)
    keyboard_cond  : (1, 13, K)          bfloat16   (per-frame keyboard one-hot)
    kv_cache1[layer]         : (1, 5280, 12, 128)   (main self-attn ring buffer)
    kv_cache_mouse[layer]    : (880, kv_cache_size, 16, 64)
    kv_cache_keyboard[layer] : (1, kv_cache_size, 16, 64)
    crossattn_cache[layer]   : (1, 257, 12, 128)
    current_start  : int — token offset into the rolling buffer
    denoising_step_list : [1000, 666, 333]   or   [1000, 750, 500, 250]
    timestep_shift : 5.0

Step 1: patchify
    x = Conv3d(36→1536, kernel=(1,2,2), stride=(1,2,2))( cat([noisy_latents, cond_concat], dim=1) )
    x = x.flatten(2).transpose(1,2)       # (1, 3*22*40=2640, 1536)

Step 2: timestep embed
    for t_idx, t in enumerate(denoising_step_list):
        t_emb = sin_emb(t, freq_dim=256)
        t_emb = SiLU( Linear(256→1536)(t_emb) )
        t_mod = Linear(1536→1536*6)(t_emb).chunk(6, dim=-1)  # (shift1, scale1, gate1, shift2, scale2, gate2) for self-attn / FFN
        ... AdaLN modulation applied per block.

Step 3: 30 transformer blocks
    for layer in range(30):
        # AdaLN scale/shift
        h = x
        h = norm1(h) * (1 + scale1) + shift1
        # Self-attention (causal, with rolling KV cache)
        q,k,v = self_attn.qkv(h)
        q,k = qknorm(q), qknorm(k)
        q,k = causal_rope_apply(q,k, freqs_3d, current_start)
        roll_kv_cache(kv_cache1[layer], k, v, local_attn_size=6)
        attn = flex_attention(q, kv_cache1[layer].k[:, -local_attn_size*880:], ..., block_mask=causal_block_mask)
        x = x + gate1 * self_attn.o(attn)

        # I2V cross-attention to CLIP visual_context (one-shot cache fill)
        h = norm3(x)
        if crossattn_cache[layer] is empty: populate from cross_attn.k(visual_context), cross_attn.v(visual_context)
        cross = flash_attn(cross_attn.q(h), crossattn_cache[layer].k, crossattn_cache[layer].v)
        x = x + cross_attn.o(cross)

        # Action module (only blocks 0..14 in distilled checkpoints)
        if layer < 15:
            x = action_module.forward(x, tt=3, th=22, tw=40,
                                      mouse_condition=mouse_cond,
                                      keyboard_condition=keyboard_cond,
                                      is_causal=True,
                                      kv_cache_mouse=kv_cache_mouse[layer],
                                      kv_cache_keyboard=kv_cache_keyboard[layer],
                                      start_frame=current_start//880,
                                      num_frame_per_block=3)
            # internally: mouse self-attn residual + keyboard cross-attn residual

        # FFN
        h = norm2(x) * (1 + scale2) + shift2
        x = x + gate2 * Linear(8960→1536)( GELU( Linear(1536→8960)(h) ) )

    # Head
    x = norm_head(x) * (1 + shift_head) + scale_head
    out = Linear(1536, 64)(x)                       # 64 = patch_t * patch_h * patch_w * out_dim = 1*2*2*16
    out = unpatchify(out)                            # (1, 16, 3, 44, 80)

Step 4: convert flow prediction to x0 estimate
    x0 = noisy_latents + (1 - sigma_t) * out        # FlowMatch formula
    if t_idx < len(denoising_step_list) - 1:
        noisy_latents = scheduler.add_noise(x0, fresh_noise, t_next)
    else:
        return x0                                   # final clean latent for this block
```

After the block returns, advance `current_start += 3 * 880 = 2640` tokens, append `x0` to the running latent buffer, and (in streaming mode) ask for the next 12 frames of action input before repeating.

### Streaming decode

```text
collected_latents = [first_frame_latent]
while True:
    new_actions = get_current_action()             # poll keyboard/mouse 12 frames worth
    mouse_cond  = build_mouse_cond(new_actions)    # (1, 13, 2)  (1 frame overlap with previous block for padding)
    key_cond    = build_key_cond(new_actions)      # (1, 13, K)
    block_latents = denoise_block(noise=randn(1,16,3,44,80), mouse_cond, key_cond)
    collected_latents.append(block_latents)
    frames = vae.decode(block_latents)             # (1, 3, 12, 352, 640) uint8 after rescale
    yield frames                                    # to display / encoder
```

## Reference Implementations

**Primary code:** [github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-2](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-2)

Source-of-truth files (canonical paths inside that subtree):

- [`inference.py`](https://github.com/SkyworkAI/Matrix-Game/blob/main/Matrix-Game-2/inference.py) — entry point; image preprocessing, VAE encode, condition build, calls `CausalInferencePipeline.inference`.
- [`inference_streaming.py`](https://github.com/SkyworkAI/Matrix-Game/blob/main/Matrix-Game-2/inference_streaming.py) — interactive variant; calls `CausalInferenceStreamingPipeline`.
- [`pipeline/causal_inference.py`](https://github.com/SkyworkAI/Matrix-Game/blob/main/Matrix-Game-2/pipeline/causal_inference.py) — **`CausalInferencePipeline`** and **`CausalInferenceStreamingPipeline`**. Builds KV caches, drives the denoising loop, applies `warp_denoising_step`, manages `current_start` / `cache_start`.
- [`utils/wan_wrapper.py`](https://github.com/SkyworkAI/Matrix-Game/blob/main/Matrix-Game-2/utils/wan_wrapper.py) — `WanDiffusionWrapper` (wraps `CausalWanModel`, owns `FlowMatchScheduler`, converts flow ↔ x0); `WanVAEWrapper` (loads `Wan2.1_VAE.pth`, per-channel mean/std normalization, encode/decode).
- [`utils/conditions.py`](https://github.com/SkyworkAI/Matrix-Game/blob/main/Matrix-Game-2/utils/conditions.py) — `Bench_actions_universal / _gta_drive / _templerun`, `combine_data`, `CAM_VALUE = 0.1`, `KEYBOARD_IDX` maps.
- [`wan/modules/causal_model.py`](https://github.com/SkyworkAI/Matrix-Game/blob/main/Matrix-Game-2/wan/modules/causal_model.py) — **`CausalWanModel`**, **`CausalWanAttentionBlock`**, **`CausalWanSelfAttention`** with `causal_rope_apply`, KV-cache, flex_attention block masks.
- [`wan/modules/action_module.py`](https://github.com/SkyworkAI/Matrix-Game/blob/main/Matrix-Game-2/wan/modules/action_module.py) — **`ActionModule`** (mouse path + keyboard path), `WanRMSNorm`.
- [`wan/modules/model.py`](https://github.com/SkyworkAI/Matrix-Game/blob/main/Matrix-Game-2/wan/modules/model.py) — original `WanModel` (non-causal); `WanAttentionBlock`, `WanI2VCrossAttention`, `Head`, `WanLayerNorm`.
- [`wan/modules/posemb_layers.py`](https://github.com/SkyworkAI/Matrix-Game/blob/main/Matrix-Game-2/wan/modules/posemb_layers.py) — `get_nd_rotary_pos_embed`, `apply_rotary_emb`.
- [`wan/modules/clip.py`](https://github.com/SkyworkAI/Matrix-Game/blob/main/Matrix-Game-2/wan/modules/clip.py) — CLIP-ViT-H/14 wrapper; only the `encode_video` method is used for visual context.
- [`wan/vae/wanx_vae.py`](https://github.com/SkyworkAI/Matrix-Game/blob/main/Matrix-Game-2/wan/vae/wanx_vae.py) and [`wan/vae/wanx_vae_src/`](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-2/wan/vae/wanx_vae_src) — Wan2.1 VAE encoder + decoder + per-channel normalization.
- [`demo_utils/vae_block3.py`](https://github.com/SkyworkAI/Matrix-Game/blob/main/Matrix-Game-2/demo_utils/vae_block3.py) — `VAEDecoderWrapper` (fp16 decode-only path used in inference; `torch.compile(mode="max-autotune-no-cudagraphs")`).
- Per-task configs: [`configs/distilled_model/{universal,gta_drive,templerun}/config.json`](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-2/configs/distilled_model), [`configs/foundation_model/config.json`](https://github.com/SkyworkAI/Matrix-Game/blob/main/Matrix-Game-2/configs/foundation_model/config.json), [`configs/inference_yaml/inference_*.yaml`](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-2/configs/inference_yaml).

**Paper:** [arXiv:2508.13009](https://arxiv.org/abs/2508.13009) — "Matrix-Game 2.0: An Open-Source, Real-Time, and Streaming Interactive World Model" (Aug 18, 2025).

**Project page:** [matrix-game-v2.github.io](https://matrix-game-v2.github.io/) — demo videos + interactive UI demo.

**HuggingFace:** [`Skywork/Matrix-Game-2.0`](https://huggingface.co/Skywork/Matrix-Game-2.0) — weights.

**Base model:** [`Skywork/SkyReels-V2-I2V-1.3B-540P`](https://huggingface.co/Skywork/SkyReels-V2-I2V-1.3B-540P) — the un-actionified parent; useful to diff Matrix-Game-2's checkpoint against.

**Wan2.1 lineage:** [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) — original `WanModel`, VAE, CLIP wrapper.

**Sibling (Matrix-Game 3.0):** [arXiv:2604.08995](https://arxiv.org/abs/2604.08995), [matrix-game-v3.github.io](https://matrix-game-v3.github.io/), [`Matrix-Game-3/`](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-3) — to be researched separately.

**Predecessor (Matrix-Game 1.0):** [arXiv:2506.18701](https://arxiv.org/abs/2506.18701), [`Skywork/Matrix-Game`](https://huggingface.co/Skywork/Matrix-Game) — 17B, not real-time, different architecture.

**Community ports:** [stevenenriquez/Matrix](https://github.com/stevenenriquez/Matrix) and [cappuch/Matrix-Game-HF](https://github.com/cappuch/Matrix-Game-HF) — diffusers integrations; useful as cross-checks for the `MatrixGame2I2VPipeline` Python wrapper.

## Differences Between Implementations

**There is only one reference implementation** (the SkyworkAI/Matrix-Game codebase). A diffusers-style `MatrixGame2I2VPipeline` ships in the HF repo's `model_index.json` but its class body is in the same codebase (`pipeline/causal_inference.py`). Community ports (`stevenenriquez/Matrix`, `cappuch/Matrix-Game-HF`) appear to just re-wrap the same code under a diffusers `DiffusionPipeline.from_pretrained` API.

**Internal divergences across the four checkpoints:**
- Foundation vs distilled — different `action_config.blocks` (30 vs 15) and different `local_attn_size` (-1 vs 6 vs 4).
- Universal vs GTA — different `keyboard_dim_in` (4 vs 2) and different `local_attn_size` (6 vs 4).
- TempleRun — `enable_mouse=False`, `keyboard_dim_in=7`, 4-step rather than 3-step denoising. Otherwise identical.

**Vestigial cleanup notes:**
- `text_len = 512` is still in every config but the text branch is gone; can be ignored.
- `xlm-roberta-large/` is in the HF repo but never loaded.
- The CLIP weight file contains both image and text towers; only the image tower is used.
- `wan/modules/t5.py` is dead code in Matrix-Game-2 (used only by upstream Wan2.1 T2V).
- `inject_sample_info = false` everywhere; the related code path is dead.

## Open Questions

1. **Exact tensor key names in the four `.safetensors`.** I inferred the key prefixes above from the Python module hierarchy. The `MatrixGame2CheckpointConverter` should be written **after** dumping actual keys:
   ```python
   from safetensors import safe_open
   with safe_open("base_distilled_model/base_distill.safetensors", framework="pt") as f:
       for k in sorted(f.keys()): print(k, f.get_slice(k).get_shape())
   ```
   In particular: confirm whether `action_model.*` lives **per block** (my guess) or as a top-level `action_model.{0..14}.*` list.

2. **`base_distill.safetensors` storage dtype.** 6.48 GB for ~1.8B params suggests **FP32 weights** (8 GB unrealistic; 4 GB would be FP16; 6.5 GB is between). Best guess: most weights FP16, action-module weights and norms FP32 = mixed precision. Confirm via key dump (look at `dtype` per tensor).

3. **Wan self-attention RoPE channel split.** The code uses `rope_params(1024, d - 4*(d//6))` for the temporal axis. For `head_dim=128`: `d//6 = 21`, so temporal = `128 - 84 = 44`, leaving `84/2 = 42` for height and width. Confirm by inspection or by printing freq tensor shapes at runtime.

4. **Wan2.1 VAE per-channel mean/std.** The `WanVAEWrapper.__init__` "initializes normalization parameters (mean/std values for 16 channels)" — the actual 16-element vectors must be read from `Wan2.1_VAE.pth` or from the `wanx_vae_src/` source. (Wan2.1 publishes these in its repo; copy them verbatim.)

5. **CLIP `encode_video` exact output shape.** I'm assuming `(1, 257, 1280)` based on standard ViT-H/14 with 224² input and 14² patches. The Matrix-Game-2 wrapper may pool / project to a different shape. Confirm by running `print(visual_context.shape)` once on a test image.

6. **AdaLN modulation expansion convention.** The `modulation` parameter is shape `[1, 6, dim]` and the `time_projection` outputs `dim*6`. I noted "(shift1, scale1, gate1, shift2, scale2, gate2)" but the actual chunk order could be `(shift1, shift2, scale1, scale2, gate1, gate2)` or similar. Read the exact assignment lines in `WanAttentionBlock.forward`.

7. **FlowMatchScheduler `timestep_shift` semantics.** `model_kwargs.timestep_shift = 5.0` is applied somewhere in the scheduler. Confirm whether it's the standard logit-normal shift (`t' = shift·t / (1 + (shift-1)·t)`, same as SD3 / Z-Image / Flux) or a different formula. The code lives in `WanDiffusionWrapper._init_scheduler`.

8. **`warp_denoising_step` exact arithmetic.** From the YAML and pipeline code, `t = scheduler.timesteps[1000 - denoise_t]`. This assumes the scheduler has been pre-built with `num_train_timesteps=1000`. Confirm `1000 - 0 = 1000` (max-noise) is a valid index (it is in a 1001-length array, otherwise off-by-one).

9. **Number of CLIP visual context tokens after pooling.** "257" appears once in the KV cache shape comment of `causal_inference.py`. Confirm — could also be a derived constant from the CLIP image resolution Matrix-Game-2 uses (might be downsampled to e.g. 16×16 + 1 = 257 explicitly).

10. **Action module's `enable_mouse=False` keyboard-only branch.** TempleRun config sets `enable_mouse=False`, but the code path for `mouse_attn_q` (which is defined in the keyboard block) is still allocated. Confirm whether TempleRun's `.safetensors` actually contains zero-valued `mouse_attn_q`/`keyboard_attn_kv`/`proj_keyboard` weights or whether those keys are absent.

11. **`use_rope_keyboard` always True?** Asserted `True` in `ActionModule.forward`, but the `else` branch (no-RoPE-on-keyboard) is fully implemented. Look for a config flag that toggles it. (Best guess: legacy code, always True now.)

12. **Streaming pipeline's exact action-polling cadence.** I described "12 frames worth = 1 block" but the actual `get_current_action()` might be called per-latent-frame (3× per block). The YAML's `num_frame_per_block: 3` suggests per-block. Confirm by reading the streaming loop.

13. **Output noise re-injection schedule between denoising steps.** `pred + sigma * fresh_noise` vs `scheduler.add_noise(x0, fresh_noise, t_next)` — the exact formula matters for the 3-step distillation to match the teacher's distribution. Read `WanDiffusionWrapper.flow_to_x0` and the loop in `CausalInferencePipeline`.

14. **`base_model/base_config.json` exact content (404'd via raw URL).** It's 972 B — likely the same shape as the foundation_model `config.json` (action blocks 0..29) but should be confirmed by HF-download.

## Implementation Notes

### How this maps to HartsyInference packages

Matrix-Game 2.0 motivates a **new `HartsyInference.World` NuGet package**. Suggested layout:

- **`HartsyInference.World`** (new) — adds:
  - `Models/MatrixGame2/MatrixGame2Config.cs`, `Models/MatrixGame2/CausalWanModel.cs`, `Models/MatrixGame2/CausalWanBlock.cs`, `Models/MatrixGame2/ActionModule.cs`.
  - `Pipelines/MatrixGame2Pipeline.cs` (offline `Bench_actions_*` driver).
  - `Pipelines/MatrixGame2StreamingPipeline.cs` (interactive driver with `IActionStream` interface).
  - `Streaming/IActionStream.cs` — abstract input source (keyboard+mouse), pluggable for game-engine integration.
  - `Streaming/RollingKvCache.cs` — generic ring-buffer KV cache reusable for Matrix-Game 3.0 and any other causal-DiT model.

- **`HartsyInference.Diffusion`** (existing) — adds:
  - `Models/Vae/Wan21VaeConfig.cs`, `Models/Vae/Wan21VaeEncoder.cs`, `Models/Vae/Wan21VaeDecoder.cs` (3D causal VAE, 16-channel, 8×8×4 compression). Design as `IWanVae` so Wan2.2 (Matrix-Game 3.0) can implement the same interface.
  - `Models/TextEncoders/OpenClipViTH14Image.cs` (image-only forward — text tower NOT needed).
  - `Schedulers/FlowMatchDmdScheduler.cs` — discrete-step Flow-Match with `warp_denoising_step` + `timestep_shift=5.0`. Reusable for any DMD-distilled flow model.

- **`HartsyInference.ModelAssets`** (existing) — adds:
  - `CheckpointConverters/MatrixGame2CheckpointConverter.cs` — loads one of `base_distill / gta_keyboard2dim / templerun_7dim_onlykey .safetensors`, splits into per-block buckets, separates `action_model.*` from `self_attn.*` / `cross_attn.*` / `ffn.*`, handles the dual mouse-path/keyboard-path projections.
  - `CheckpointConverters/Wan21VaePthConverter.cs` (or recommend offline conversion to `.safetensors`).
  - `CheckpointConverters/OpenClipViTH14CheckpointConverter.cs` — parse the 4.77 GB `.pth`, take only the visual tower (~1.0 GB FP16).

### Net-new backend / kernel work required

1. **3D Causal Convolution.** The Wan2.1 VAE uses `CausalConv3d` with a 2-frame cache (same family as Wan2.2 / Lance). Requires `IBackend.Conv3D(input, weight, bias, stride, padding, dilation)`. Naive `im2col + GEMM` is fine for v1; specialized streaming kernels follow. **Shared with future Wan / LTX video pipelines.**

2. **Flex-attention block mask.** The `flex_attention(block_mask=causal_block_mask_per_block_of_3)` call needs an equivalent in HartsyInference. First-pass: emit a precomputed dense bool mask and do padded dense attention; later, a true sparse / block-sparse attention kernel.

3. **Rolling KV cache primitive.** Reusable `RollingKvCache<T>` data structure that supports `Insert(newK, newV)` with eviction-and-memcpy. The pattern in `action_module.py` (clone-shift the survivors forward then overwrite the tail) is straightforward — implement once as a generic kernel, reuse across all three caches (main self-attn, mouse self-attn, keyboard cross-attn).

4. **CLIP cross-attention K/V one-shot fill.** `crossattn_cache[layer]` is filled on the first call and reused. Pipeline-level optimization, not a new kernel — just gate `cross_attn.k(visual_context)` and `cross_attn.v(visual_context)` behind a `null check`.

5. **bf16 + fp16 mixed precision.** Weights/activations are bf16; the VAE decoder runs fp16 with `torch.compile(max-autotune-no-cudagraphs)`. HartsyInference already supports both; ensure the autocast policy in the pipeline mirrors this (DiT → bf16, VAE decode → fp16).

6. **3D RoPE (already partially in DiT primitives).** The main DiT's RoPE splits the 128-dim head into `(44, 42, 42)` for `(T, H, W)`. The action module's RoPE splits 64-dim into `(8, 28, 28)`. Both reduce to the same `apply_rotary_emb` primitive; only the freq-cache precompute differs.

7. **GELU(tanh approximation).** Standard PyTorch `gelu(approximate="tanh")`. Already implemented in `ActivationKernels`. Used by `mouse_mlp` and the main DiT's `ffn` (depending on Wan version — confirm).

8. **AdaLN modulation.** The 6-way and 2-way splits are simple element-wise ops. Existing `AdaLnModulation` in `DiTBlocks/` should handle this.

9. **No new tokenizer work.** No text path. (Skip `xlm-roberta-large/`, `wan/modules/t5.py`, the CLIP text tower.)

### VRAM and viability per target GPU

| GPU | VRAM | Universal (1.8B distilled, bf16) | TempleRun (4 steps, mouse off) |
|---|---|---|---|
| RTX 3060 12 GB | 12 | Tight. DiT bf16 ≈ 3.6 GB + VAE 16-bit ≈ 0.25 GB + CLIP image tower ≈ 0.7 GB + activations + KV caches (~1.5 GB for `local_attn_size=6` × 880 × 12 × 128 × 4 B × 30 layers ≈ 1.1 GB). Total ≈ 6.5 GB resident; should fit. Probably 10-15 FPS on a 3060 due to compute, not memory. |
| RTX 4060 16 GB | 16 | Comfortable. Expect 15-20 FPS. |
| RTX 4090 24 GB | 24 | Comfortable. Should achieve ~25 FPS at the official preset. |
| H100 SXM 80 GB | 80 | Reference target. 25 FPS as marketed. |

The README's "Nvidia GPU with **24 GB+** memory" is conservative — likely required because of the 4.77 GB CLIP load + uncompiled VAE; with our pure-C# stack, we can probably hit 12 GB cards.

Recommended quality presets (per `QualityProfileApplier`):
- `Standard` = bf16 DiT + fp16 VAE + fp16 CLIP image tower.
- `Low` (12 GB target) = FP8 DiT (Q8_K via existing GGUF backend) + fp16 VAE.

Q4_K GGUF dumps are not yet available for Matrix-Game 2.0. When `unsloth/Matrix-Game-2.0-GGUF` or similar appears, a Q4_K DiT (~1 GB) makes the 12 GB path trivial.

### Ordering / dependencies for the build

1. **Land the Wan2.1 VAE first.** This is the lowest-risk piece and is reusable across the whole Wan / SkyReels / Matrix-Game family. Implement encode + decode + tiling + per-channel normalization; validate against `Wan2.1_VAE.pth` outputs on a static image.
2. **Land the CLIP-ViT-H/14 image encoder.** Frozen forward, no training. Verify the (1, 257, 1280) output on a test image.
3. **Land the `CausalWanModel` backbone WITHOUT action modules** (load the foundation `base_model` checkpoint with `enable_keyboard=False, enable_mouse=False`). Validate against the un-actionified SkyReels-V2-I2V-1.3B-540P on a single-block generation. This is the high-risk net-new infra.
4. **Add `ActionModule` and rolling KV cache.** Validate on the universal distilled checkpoint against `Bench_actions_universal` outputs.
5. **Add few-step Flow-Match scheduler with `warp_denoising_step`.** Verify the 3-step deterministic loop matches the Python reference numerically.
6. **Build the offline pipeline + the streaming pipeline.** Add the `IActionStream` abstraction for game-engine integration.
7. **Phase 11 (post-2.0): bring up Matrix-Game 3.0** by swapping the backbone (Wan2.2 5B) and VAE (Wan2.2 48-ch), keeping the ActionModule and KV-cache primitives.

### Test-skipping discipline

> **Superseded 2026-08-06.** The per-model pipeline/generation tests this section specified were
> removed in the test-suite cleanup, and the rule is now the opposite: **do not add a test that
> proves a model works end to end** — a model that stops working is visible the moment anyone uses
> it. Test what breaks quietly instead (kernel numerics, cross-device equivalence, quantization and
> codec round-trips, padding/tiling geometry, format and key mapping), and put shared-component
> parity in `tests/<Project>/Parity/` with a `*ParityTests` name. See `docs/CODE_STYLE.md` §Testing.

### Reuse opportunities

- `Qwen3Tokenizer` infrastructure — NOT reused (no text).
- `FlowMatchEulerDiscreteScheduler` — partially reusable; we need a *discrete few-step* variant. Factor out a `FlowMatchDmdScheduler` that the existing Flux / Z-Image pipelines can also use.
- `SinusoidalTimestepEmbedding`, `WanLayerNorm` (≡ standard LayerNorm), `WanRMSNorm` (≡ existing RmsNorm) — direct reuse.
- `OpenClipViTH14` — the image tower; useful for any future image-conditioned diffusion pipeline (LTX, Wan-i2v, AnimateDiff successors).
- Wan2.1 VAE — foundation for Matrix-Game 3.0 (after swap to Wan2.2) and any future Wan-family image / video pipelines.
- `RollingKvCache` — reusable for Matrix-Game 3.0, future causal-DiT, and possibly StreamingLLM-style language model work.

### Critical things to NOT skip in v1

- **Per-channel VAE mean/std normalization** — silent off-by-one here yields a usable but color-wrong output; lift the 16 values verbatim from Wan2.1.
- **`warp_denoising_step = true`** — the distilled checkpoints are trained against the warp; running them without it makes the 3-step inference collapse.
- **KV-cache eviction memcpy** — not optional; a fully-static cache will silently produce a wrong output once `local_end_index > kv_cache_size`.
- **`is_causal=True` flag on every flash-attn call** in the action module — this controls whether the kv_cache_* branches are taken.

### Long-term: how this dovetails with Matrix-Game 3.0

Once Matrix-Game 3.0 lands:
- Replace `Wan21VaeDecoder` with `Wan22VaeDecoder` (~48 latent channels, different mean/std, otherwise same `CausalConv3d` + RMSNorm).
- Replace the 1.3B `CausalWanModel` with the 5B version (dim ≈ 3072, layers ≈ 30-ish, heads ≈ 24).
- Action modules: same class, more blocks possibly.
- Add: long-horizon memory retrieval module (camera-pose query → frozen memory bank K/V), frame self-correction loop (synthetic-frame re-injection during inference for drift correction).
- Both Matrix-Game 2.0 and 3.0 can ship from the same `HartsyInference.World` package with a model-card discriminator.

---

Sources:
- [Skywork/Matrix-Game-2.0 on HuggingFace](https://huggingface.co/Skywork/Matrix-Game-2.0)
- [Matrix-Game GitHub (SkyworkAI)](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-2)
- [arXiv 2508.13009 — Matrix-Game 2.0 paper](https://arxiv.org/abs/2508.13009)
- [arXiv 2604.08995 — Matrix-Game 3.0 paper](https://arxiv.org/abs/2604.08995)
- [arXiv 2506.18701 — Matrix-Game 1.0 paper](https://arxiv.org/abs/2506.18701)
- [Skywork/SkyReels-V2-I2V-1.3B-540P (base model)](https://huggingface.co/Skywork/SkyReels-V2-I2V-1.3B-540P)
- [matrix-game-v2.github.io project page](https://matrix-game-v2.github.io/)
- [matrix-game-v3.github.io project page](https://matrix-game-v3.github.io/)
- [Wan-Video/Wan2.1 lineage](https://github.com/Wan-Video/Wan2.1)

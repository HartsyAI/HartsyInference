# Lance — Research Notes

> Status: Complete — **image T2I + video T2V IMPLEMENTED 2026-06-08** (both structurally verified end-to-end on CPU; numeric validation vs checkpoint pending). Last Updated: 2026-07-16 | Video: `LanceVideoPipeline` + Wan2.2 VAE streaming decode (`feat_cache`) + frame-streaming/encoders all built — see PHASE_9 § 6.
>
> **REAL-CHECKPOINT RECONCILIATION (2026-07-16, first released `Lance_3B/model.safetensors`, 11.8 GB BF16 — NOT the 24.7 GB from the pre-release table).** Facts that override guesses below (source: real key dump + `inference_lance.sh` + the released GitHub code):
> - **Latent patch is `(1,1,1)`**, NOT (1,2,2): `vae2llm = Linear(48→2048)`, `llm2vae = Linear(2048→48)` — one token per 48-channel latent pixel; total spatial downscale is **16×** (not 32×). Head keys are `vae2llm.*` / `llm2vae.*` (not `vae_in`/`vae_out`).
> - **QK-RMSNorm is ON** (OQ#3 resolved): per-head `q_norm`/`k_norm` [128] with `_moe_gen` siblings in every layer (`--llm_qk_norm true` und+gen).
> - **`max_latent_size = 64`** (not 32); the frozen `latent_pos_embed.pos_embed` [4096, 2048] table (1 frame × 64×64) SHIPS in the checkpoint and is indexed `t·64² + h·64 + w` (`get_flattened_position_ids_extrapolate_video`). Load it, don't recompute.
> - **Positions:** released inference runs `--apply_qwen_2_5_vl_pos_emb true` → Qwen2.5-VL `get_rope_index` M-RoPE (text 1-D, vision block 3-D anchored at the first pad; video temporal step = `tokens_per_second` 2). **MaPE (OQ#2) does NOT apply to pure T2I/T2V** — `shift_position_ids` only shifts `full_noise`/`full` splits (editing refs), and the T2I target block is mode `"noise"`.
> - **Sequence:** ChatML template (`text_template=true`) with a fixed per-task system prompt; the vision block is `<|vision_start|><|video_pad|>×N<|vision_end|>` (video_pad 151656 even for images!) followed by `<|im_end|>`. The **noise split includes both sentinels** (bidirectional inside, invisible from outside). Uncond = same sequence with the caption tokens (modality 0) removed.
> - **Sampling (OQ#4/#5 resolved):** 2-way text CFG only for T2I, `cfg_interval=[0.4, 1.0]` (cond-only below t=0.4), `cfg_renorm_type="global"` with `cfg_renorm_min=0` (`v *= clamp(‖v_cond‖/‖v_cfg‖, 0, 1)`); timestep fed to the embedder is the raw shifted t∈[0,1].
> - **No ViT/connector/task/modality embeds in this checkpoint** — T2I-only release; `lm_head` present but unused by generation.
> - Engine: `LanceCheckpointConverter`/`LanceTransformer`/`LancePipelineCommon`/`LanceImagePipeline` reconciled; parity via `tests/python-reference/dump_lance_reference.py` + `LanceRealWeightParityTests` + `diff_lance_layers.py`.
>
> **BUILD STATUS (2026-06-08):** Lance image T2I is built and runs end-to-end (see PHASE_4 § Lance). Several open questions below were resolved while building from the verbatim upstream source (pulled raw):
> - **OQ#2 (MaPE offsets):** NOT in `get_rope_index` (that's stock Qwen2.5-VL M-RoPE). They live in `data/common.py` `shift_position_ids` — `pos_shift=1000`; modality type-4 (gen/noisy) temporal rebased to the 1000 range, type-3 (clean-VAE) to 2000. Spatial axes unchanged.
> - **OQ#11 (CausalConv3d / `CACHE_T`):** the VAE decode driver processes ONE latent frame per call; frame 0 uses `first_chunk=True` with a fresh all-None cache, so **for a single image (T=1) the whole decoder is stateless** — the feat_cache streaming machine is video-only. Also: `feat_cache=None` is INVALID mid-decode (Resample skips `time_conv` → temporal mismatch vs the `DupUp3D` shortcut).
> - **Conv3D:** no net-new `IBackend.Conv3D` needed — `CausalConv3d` decomposes into `Conv2D` over temporal taps (all backends).
> - **OQ#12 (MoT `_moe_gen` layout):** confirmed — gen-path weights are sibling keys (`*_moe_gen`), not fused; routing is deterministic by modality role (gen mode: text→und weights, vae→`_moe_gen` weights), one joint attention. Latent handoff einops: `(t pt)(h ph)(w pw) c → (t h w)(pt ph pw c)` channel-last.
> - **Still pending (validation-gated):** exact `model.safetensors` key names (OQ#1), `llm_qk_norm` default (OQ#3), `cfg_vision_scale` (OQ#4, editing-only), sparse attn mask exactness (OQ#8 — currently full attn for B=1).
> Source of truth: [ByteDance/Lance GitHub](https://github.com/bytedance/Lance), [arXiv 2605.18678](https://arxiv.org/abs/2605.18678), [HF repo `bytedance-research/Lance`](https://huggingface.co/bytedance-research/Lance)
> License: Apache 2.0
> Related: [`Z_IMAGE_ARCHITECTURE.md`](Z_IMAGE_ARCHITECTURE.md) (M-RoPE / NextDiT lineage of the LLM backbone), [`FLOW_MATCHING_AUDIO.md`](FLOW_MATCHING_AUDIO.md) (rectified-flow background), [`TEXT_ENCODERS.md`](TEXT_ENCODERS.md), [`VAE_ARCHITECTURE.md`](VAE_ARCHITECTURE.md)

## Summary

Lance ("Unified Multimodal Modeling by Multi-Task Synergy", ByteDance Research, May 2026) is a **single 3B-active-parameter** native unified multimodal model. One checkpoint covers six tasks: text-to-image (T2I), text-to-video (T2V), image editing, video editing, image understanding (VQA / captioning), and video understanding. There are **two release variants** sharing the same architecture: `Lance_3B` (image-only specialist, 24.7 GB safetensors) and `Lance_3B_Video` (general image+video, 28.4 GB safetensors).

The backbone is a **modified Qwen2.5-VL 3B decoder** (36 layers / hidden 2048 / 16 heads / 2 KV heads / FFN 11008 / RMSNorm / SwiGLU / GQA factor 8). The novel piece is a per-layer **dual-stream "MoT" (Mixture-of-Tokens) routing**: every token is *also* annotated with a modality role (text-und, ViT-semantic, clean-VAE, noisy-VAE) and is sent through one of two parallel sets of (Q, K, V, O, gate_proj, up_proj, down_proj, input_layernorm, post_attention_layernorm) — the standard set for **understanding** tokens and a parallel `*_moe_gen` set for **generation** tokens. Both streams attend to each other through a single shared joint attention. On top of this Lance applies **MaPE** (Modality-Aware Positional Encoding), which shifts only the *temporal* axis of the model's 3D M-RoPE by a per-role offset Δ_m so the four token roles don't collide in the shared position space.

Generation is **rectified-flow velocity prediction** with logit-normal timestep shifting (3.5 for image, 4.0 for video), explicit Euler integration over a default 30 steps, and **three-way CFG** (text + vision conditional, default text scale 4.0, optional renorm). Understanding is autoregressive next-token over the LLM's text vocab. The frozen Qwen2.5-VL ViT (32 layers / 1280 dim / 14-px patches / 4-frame temporal patches / windowed attention with full-attention at indices 7/15/23/31) provides semantic vision tokens; the frozen **Wan2.2 3D causal VAE** (48 latent channels, 16× spatial / 4× temporal downscale, RMSNorm, CausalConv3d with 2-frame cache) provides generation latents. Latents are patchified `(t,h,w)=(1,2,2)` → 192-dim tokens → `Linear(192→2048)` and joined back via `Linear(2048→192)` after the transformer.

For HartsyInference, Lance is best treated as **two related pipelines sharing one transformer backbone**: a Phase 4 (image breadth) `LanceImagePipeline` using `Lance_3B`, and a Phase 9 (video) `LanceVideoPipeline` using `Lance_3B_Video`. The image path can ship without 3D temporal infra; the video path requires the Wan2.2 3D causal VAE and CausalConv3d streaming-decode plumbing. Either path requires net-new backend work for **packed/var-length attention** (FlashAttention's `flash_attn_varlen_func`) and **MoT routing** (dispatching different weights for different token slices in the same attention call).

## Detailed Findings

### 1. Family / variants

| Variant | Params (active) | Modalities | Safetensors | Notes |
|---|---|---|---|---|
| `Lance_3B` | 3 B | T2I + image edit + image understanding | 24.7 GB | Image-only specialist. Same LLM config as Video. |
| `Lance_3B_Video` | 3 B | T2I + T2V + image/video edit + image/video understanding | 28.4 GB | Larger because more generation streams trained. |
| `Qwen2.5-VL-ViT` | (frozen) | shared | 1.34 GB | Vision tokenizer used by both. |
| `Wan2.2_VAE.pth` | (frozen) | shared | 2.82 GB | 3D causal VAE, used by both. |

Both `Lance_3B` and `Lance_3B_Video` ship with identical `llm_config.json`, `tokenizer.json`, `generation_config.json`, `vocab.json`, `merges.txt` — only `model.safetensors` differs.

### 2. Backbone: modified Qwen2.5-VL 3B

From `Lance_3B/llm_config.json`:

| Field | Value |
|---|---|
| `architectures` | `Qwen2_5_VLForConditionalGeneration` |
| `hidden_size` | **2048** |
| `num_hidden_layers` | **36** |
| `num_attention_heads` | **16** (head_dim = 128) |
| `num_key_value_heads` | **2** (**GQA factor 8**) |
| `intermediate_size` | **11008** |
| `hidden_act` | `silu` → SwiGLU FFN (`down(silu(gate(x)) * up(x))`) |
| `rms_norm_eps` | **1e-6** |
| `rope_theta` | **1,000,000.0** |
| `rope_scaling.type` | `mrope` |
| `rope_scaling.mrope_section` | **[16, 24, 24]** (t, h, w) — sums to 64 = head_dim / 2 |
| `max_position_embeddings` | 128,000 |
| `sliding_window` | 32768 (disabled — `use_sliding_window=false`) |
| `vocab_size` | **151,936** |
| `tie_word_embeddings` | true |
| `torch_dtype` | `bfloat16` |
| `bos_token_id` | 151643 |
| `eos_token_id` | 151645 |
| Vision sentinel IDs | `vision_start=151652`, `vision_end=151653`, `vision_token=151654`, `image_token=151655`, `video_token=151656` |

The "modified" piece is **MoT** (see § 3). Otherwise the LLM is a vanilla Qwen2.5-VL decoder: causal mask, RMSNorm-pre, RMSNorm-post, SwiGLU, GQA, M-RoPE, tied embeddings.

### 3. MoT (Mixture-of-Tokens) — dual-stream routing

Reference: `modeling/lance/qwen2_navit.py`. Three decoder-layer classes are defined:

- `Qwen2DecoderLayer` — vanilla (single stream).
- `Qwen2MoEDecoderLayer` — dual MLPs only.
- `Qwen2MoTDecoderLayer` — **dual QKV + dual norms + dual MLPs** ← Lance uses this.

Per layer, the MoT variant stores **two complete parameter sets**:

| Slot | "Und" path (understanding) | "Gen" path (generation, `_moe_gen` suffix) |
|---|---|---|
| Attention | `q_proj`, `k_proj`, `v_proj`, `o_proj` | `q_proj_moe_gen`, `k_proj_moe_gen`, `v_proj_moe_gen`, `o_proj_moe_gen` |
| FFN | `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj` | `mlp_moe_gen.gate_proj`, `mlp_moe_gen.up_proj`, `mlp_moe_gen.down_proj` |
| Norms | `input_layernorm`, `post_attention_layernorm` | `input_layernorm_moe_gen`, `post_attention_layernorm_moe_gen` |

**Routing key:** token's *modality role*, not a learned router. Roles:

- `text-und` → "und" path. (Text tokens used for understanding output / prompt.)
- `vit-semantic` → "und" path. (Qwen2.5-VL ViT features projected into LLM hidden via MLPconnector.)
- `clean-vae-cond` → "gen" path. (VAE latents of cond images/videos in editing tasks.)
- `noisy-vae-target` → "gen" path. (The denoising target during flow-matching forward.)

**Joint attention:** a single `PackedAttentionMoT.forward` does both QKV computations (one per route), concatenates the projected Q/K/V buffers back into a single packed sequence in original order, then runs **one** `flash_attn_varlen_func` call. Outputs are similarly demuxed back to each route's `o_proj`. This is *not* a "gating" MoE — there is no router probability; the route is determined deterministically by the modality role.

**Active parameter count = 3 B** because for any given token only one of the two parameter sets is touched. Total stored parameters are roughly 2× a vanilla Qwen2.5-VL 3B.

**Helper functions in `qwen2_navit.py`:**
- `freeze_und_params()` — used during the gen-only init phase.
- `init_moe()` — copies und-path weights into gen-path slots as initialization.
- `untie_lm_head()` — splits the tied embedding/LM head if the LM head needs to diverge.
- `get_rope_index(...)` — builds the 3D (t,h,w) position index tensor for M-RoPE over an interleaved sequence, applying the MaPE Δ_m offsets per role.
- `create_sparse_mask(sample_lens, split_lens, attn_modes, device)` — builds the FlashAttention block mask with `BLOCK_SIZE=128`. Text → causal; visual blocks → bidirectional within block; cross-block edges → only "later attends earlier clean" (3D causal).

**Optional QK-norm:** controlled by `llm_qk_norm` flag (per-head RMSNorm on Q and K). Default in released checkpoints not verified by web inspection — must be confirmed against `model.safetensors` keys (presence of `q_norm`/`k_norm` weights).

### 4. M-RoPE + MaPE (Modality-Aware Positional Encoding)

The LLM uses Qwen2.5-VL's **M-RoPE** (multimodal RoPE): the 128-dim head is split into three axis groups with `mrope_section = [16, 24, 24]` (×2 for cos/sin → 32/48/48 floats per token per axis). Axes are `(t, h, w)`.

**MaPE** is Lance's add-on. For modality role m ∈ {text-und, vit-semantic, clean-vae, noisy-vae} the **temporal** axis is shifted by Δ_m before RoPE is applied:

```
p_{t,h,w}^{(m)} = [ t̂_{t,h,w}^{(m)} + Δ_m ,  ĥ_{t,h,w}^{(m)} ,  ŵ_{t,h,w}^{(m)} ]
```

so identical (h, w) locations across roles still get distinct rotations. Spatial axes are unchanged, preserving locality. The specific Δ_m integer offsets are implemented inside `get_rope_index()` and could not be extracted from the web — **must be read locally** once the repo is cloned. The paper ablation (Table — removing MaPE) drops GEdit-Bench image-editing avg from 6.86 → 6.30, so MaPE is load-bearing for the editing tasks.

### 5. Vision encoder (frozen, understanding only) — Qwen2.5-VL ViT

From `Qwen2.5-VL-ViT/config.json`:

| Field | Value |
|---|---|
| `depth` | 32 |
| `hidden_size` | 1280 |
| `intermediate_size` | 3420 |
| `num_heads` | 16 (head_dim = 80) |
| `hidden_act` | `silu` |
| `in_channels` / `in_chans` | 3 |
| `patch_size` / `spatial_patch_size` | **14** |
| `temporal_patch_size` | **2** (consumes 2 frames per token) |
| `spatial_merge_size` | **2** (post-encoder 2×2 token merge → 4× reduction) |
| `window_size` | **112** (window attention) |
| `fullatt_block_indexes` | **[7, 15, 23, 31]** (full-attn layers, others windowed) |
| `tokens_per_second` | 2 (temporal token rate) |
| `out_hidden_size` | **2048** (already projected to LLM dim inside ViT) |
| `torch_dtype` | bfloat16 |
| `_attn_implementation` | flash_attention_2 |

Used **only on the understanding path**. Generation paths get their visual conditioning from VAE latents (not from ViT). Output is consumed by `MLPconnector(vit_hidden=1280 → 2048, act="gelu_pytorch_tanh")` and then injected into the shared sequence as "und-stream" tokens.

### 6. Generation VAE — Wan2.2 3D Causal VAE

Reference: `modeling/vae/wan/vae2_2.py` and `model.py`. Wrapper class `Wan2_2_VAE` owns the actual `WanVAE_`.

**`AutoEncoderParams` (set in `LanceConfig`):**

```python
downsample_spatial = 16
downsample_temporal = 4
z_channels = 48
```

**`WanVAE_.__init__` defaults (note: `z_dim` is **overridden to 48** at the use site):**

```python
dim = 160                      # base channels
dec_dim = 256                  # decoder base channels
z_dim = 16                     # OVERRIDDEN → 48
dim_mult = [1, 2, 4, 4]
num_res_blocks = 2
attn_scales = []
temperal_downsample = [True, True, True]  # vae2_2 instantiation
dropout = 0.0
```

**Pipeline:**
- Encoder3d input is patchified with `patch_size=2` ⇒ 12 channels in. `CausalConv3d(12 → 160)`, 4 stages with channel mults [1,2,4,4] and 2 residual blocks each, mid-attention block, head → 96 (μ and logvar of 48 each).
- Decoder3d mirrors the encoder, ending at 12 channels → unpatchified back to RGB.
- All convolutions are `CausalConv3d` with `CACHE_T = 2` (2-frame cache) so streaming/chunked decode is possible without temporal leakage from the future.
- **RMSNorm** replaces GroupNorm throughout (Wan2.2 design).
- Total VAE params ≈ 127 M; weight file `Wan2.2_VAE.pth` is 2.82 GB FP32.

**Latent normalization (per-channel; constants embedded in the module):**

Mean (48 values):
```
[-0.2289, -0.0052, -0.1323, -0.2339, -0.2799,  0.0174,  0.1838,  0.1557,
 -0.1382,  0.0542,  0.2813,  0.0891,  0.1570, -0.0098,  0.0375, -0.1825,
 -0.2246, -0.1207, -0.0698,  0.5109,  0.2665, -0.2108, -0.2158,  0.2502,
 -0.2055, -0.0322,  0.1109,  0.1567, -0.0729,  0.0899, -0.2799, -0.1230,
 -0.0313, -0.1649,  0.0117,  0.0723, -0.2839, -0.2083, -0.0520,  0.3748,
  0.0152,  0.1957,  0.1433, -0.2944,  0.3573, -0.0548, -0.1681, -0.0667]
```

Std (48 values):
```
[0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
 0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
 0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
 0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
 0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
 0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744]
```

**Encode side:** `z = (μ − mean) / std`. **Decode side:** `μ̃ = z·std + mean`.

**Tokenization handoff to the transformer:**
- VAE latent of shape `(B, 48, T, H, W)` is patched with `latent_patch_size = (1, 2, 2)` (T, H, W).
- Token feature dim = `48 × 1 × 2 × 2 = 192`.
- Projected by `nn.Linear(192 → 2048)` (the `vae_in` head).
- After the transformer, the noisy-target slice is read back with `nn.Linear(2048 → 192)` (the `vae_out` head) and unpatched.

### 7. Lance top-level module

Reference: `modeling/lance/lance.py`.

**`LanceConfig` defaults:**

```python
visual_gen = True
visual_und = True
llm_config = None             # Qwen2.5-VL 3B (above)
vit_config = None             # Qwen2.5-VL ViT (above)
vae_config = AutoEncoderParams(downsample_spatial=16, downsample_temporal=4, z_channels=48)
latent_patch_size = (1, 2, 2)
max_latent_size = 32          # max H/W in latent grid units
vit_max_num_patch_per_side = 70
connector_act = "gelu_pytorch_tanh"
interpolate_pos = False
timestep_shift = 1.0          # logit-normal shift; overridden at inference (3.5 / 4.0)
```

**Submodules instantiated by `Lance(PreTrainedModel)`:**

- `language_model` — modified `Qwen2ForCausalLM` (the MoT-augmented backbone).
- `vit` — `Qwen2_5_VisionTransformerPretrainedModel`, optional (off if `visual_und=False`).
- `MLPconnector(vit_hidden=1280 → 2048, act="gelu_pytorch_tanh")` — ViT → LLM projection.
- `vae_in: nn.Linear(192 → 2048)` — patched VAE latent → hidden.
- `vae_out: nn.Linear(2048 → 192)` — hidden → patched VAE latent.
- `TimestepEmbedder` — sinusoidal embed (dim 256) → MLP → 2048; conditions noisy stream on flow-matching timestep.
- `PositionEmbedding3D(max_latent_num_frames, max_latent_size=32, hidden_size=2048)` — frozen 3D sin-cos position embed added to VAE patches.
- Optional `nn.Embedding(10, 1280)` for task id and `nn.Embedding(10, 1280)` for modality id (added to ViT features). Usage in released weights TBD.

**Sequence packing example** (for an editing prompt with one source image and one noisy target):

```
[ ... text(B_text(T)) ... <|vision_start|> ViT(V_vit_semantic) clean_vae(V_clean) noisy_vae(V_noisy) <|vision_end|> ... text(B_text(T')) ... ]
```

Visual blocks are always wrapped in `<|vision_start|>` (151652) and `<|vision_end|>` (151653). The block mask permits bidirectional attention inside a visual block and 3D-causal cross-block attention.

### 8. Flow-matching forward + sampler

**Training-time forward (rectified flow):**

```
t ~ U(0, 1)
x_t = t · noise + (1 − t) · clean_latent
target = noise − clean_latent              # velocity
v_θ    = Lance(x_t, text, [vit], [clean_cond_latent], t)
loss   = MSE(v_θ, target)                  # MSE branch
```

**Logit-normal timestep shift** (Stable Diffusion 3 style):

```
t' = shift · t / (1 + (shift − 1) · t)
```

- `timestep_shift = 1.0` at train init (no shift).
- **3.5 for image inference**, **4.0 for video inference** (CT / SFT / RL phases).

**Inference-time loop (Euler):**

```
steps = torch.linspace(1, 0, num_timesteps + 1)        # default num_timesteps = 30
steps = apply_logit_normal_shift(steps, shift)
for k in range(num_timesteps):
    t      = steps[k]
    t_next = steps[k+1]
    dt     = t - t_next
    v_t    = forward(latents, ...)
    latents = latents - v_t * dt
```

**Three-way CFG** (text + vision conditional; see `validation_gen`):

```
v_cond                    = Lance(x_t, text=cond,   vision=cond)
v_text_uncond             = Lance(x_t, text=uncond, vision=cond)
v_text_vision_uncond      = Lance(x_t, text=uncond, vision=uncond)

v_final = v_text_vision_uncond
        + cfg_text_scale   · (v_cond           − v_text_uncond)
        + cfg_vision_scale · (v_text_uncond    − v_text_vision_uncond)
```

- Default `cfg_text_scale = 4.0` (shell script).
- Default `cfg_vision_scale` not in `inference_lance.sh` — must read `config/examples/*.json` per-task. **Open question.**

**Optional CFG renorm** — `scale = clip(||v_t|| / (||v_final|| + 1e-8), cfg_renorm_min, 1.0)`. Modes: `global`, `channel`, or off. Defaults unknown without local config inspection.

**KV-cache path** (`validation_gen_KVcache`): a `NaiveCache` keeps Q/K/V of the *clean* context (text + ViT + clean-VAE) across denoising steps. Only the noisy-target block is recomputed each step. ~2-3× wall-clock speedup typical.

### 9. Inference defaults (`inference_lance.sh`)

```
NUM_GPUS=1
VALIDATION_NUM_TIMESTEPS=30
VALIDATION_TIMESTEP_SHIFT=3.5            # 4.0 for video
CFG_TEXT_SCALE=4.0
VALIDATION_DATA_SEED=42
NUM_FRAMES=50                            # max 121
VIDEO_HEIGHT=768   VIDEO_WIDTH=768
RESOLUTION=video_480p                    # or image_768res
USE_KVCACHE=true
```

Image upper resolution = **768×768**. Video upper resolution = **480×848** (480p), max **121 frames** (≈ 5 s @ 24 fps).

### 10. Training (for context only — not needed for inference)

Four phases, AdamW (β1=0.9, β2=0.95, ε=1e-15), grad-clip 1.0:

| Phase | LR | Sched | Steps | SeqLen | Tokens | Res. | TS shift | CE:MSE |
|---|---|---|---|---|---|---|---|---|
| PT | 1e-4 | const | 350k | 44–50K | 1.5T | 192–848 | 1.0 | 0.25:1 |
| CT | 1e-4 | const | 80k  | 74–80K | 300B | 480–848 | 4.0 | 0.5:1 |
| SFT | 2.5e-5 | cos | 15k  | 74–80K | 72B | 480–848 | 4.0 | 0.25:1 |
| RL | 2e-6 | const | 800  | 74–80K | 0.5B | 480–848 | 4.0 | — |

Trained on 128× A100. RL reward model = PaddleOCR (text consistency).

Benchmark scores:
- GenEval (T2I) **0.90 overall** — ties best unified model.
- DPG-Bench (T2I) **84.67**.
- GEdit-Bench (image editing) **7.30 avg** — best unified.
- VBench (T2V) **85.11 total** — best 3B unified.
- MVBench (video understanding) **62.0 avg** — best unified (+11.3 over runner-up).

## Key Numbers / Constants

| Constant | Value | Where it's used |
|---|---|---|
| `hidden_size` | 2048 | LLM backbone dim |
| `num_hidden_layers` | 36 | Decoder depth |
| `num_attention_heads` | 16 | head_dim = 128 |
| `num_key_value_heads` | 2 | GQA factor 8 |
| `intermediate_size` | 11008 | FFN inner dim |
| `rms_norm_eps` | 1e-6 | All RMSNorms |
| `rope_theta` | 1,000,000 | M-RoPE base |
| `mrope_section` | [16, 24, 24] | t/h/w axis split of 64 = head_dim/2 |
| `vocab_size` | 151,936 | Qwen2 BPE |
| `bos_token_id` | 151643 | — |
| `eos_token_id` | 151645 | (`<|im_end|>`) |
| Vision start / end | 151652 / 151653 | — |
| vision_pad / image_pad / video_pad | 151654 / 151655 / 151656 | Sequence sentinels |
| ViT depth × hidden | 32 × 1280 | Qwen2.5-VL ViT |
| ViT heads | 16 (head_dim 80) | — |
| ViT FFN | 3420 | — |
| ViT patch | 14 spatial, 2 temporal | — |
| ViT spatial merge | 2×2 | post-encoder token reduction |
| ViT full-attn layers | [7, 15, 23, 31] | others use window 112 |
| ViT output dim | 2048 | already at LLM hidden |
| VAE `z_channels` | 48 | latent feature dim |
| VAE downsample (spatial / temporal) | 16× / 4× | — |
| VAE `dim`, `dec_dim` | 160, 256 | base channels |
| VAE `dim_mult` | [1, 2, 4, 4] | — |
| VAE `num_res_blocks` | 2 | — |
| VAE `patch_size` | 2 | RGB → 12-channel pre-conv |
| VAE `CACHE_T` | 2 | CausalConv3d frame cache |
| `latent_patch_size` | (1, 2, 2) | (T, H, W) patchify of VAE latent |
| Patch feature dim | 192 | 48 × 1 × 2 × 2 |
| `max_latent_size` | 32 | latent grid H/W cap |
| `connector_act` | `gelu_pytorch_tanh` | ViT→LLM MLP activation |
| `TimestepEmbedder` dim | 256 | sinusoidal before MLP |
| `timestep_shift` | 3.5 (image) / 4.0 (video) | logit-normal |
| `num_timesteps` (inference) | 30 | default |
| `cfg_text_scale` | 4.0 | default text branch of 3-way CFG |
| Image max resolution | 768 × 768 | — |
| Video max resolution | 480 × 848 | 480p preset |
| Max frames (video) | 121 | ≈ 5 s @ 24 fps |
| FlashAttention block size | 128 | `create_sparse_mask` |
| Active parameters | 3 B | per-token (MoT routing) |
| Storage (Lance_3B safetensors) | 24.7 GB | image-only variant |
| Storage (Lance_3B_Video safetensors) | 28.4 GB | unified variant |
| Storage (ViT) | 1.34 GB | — |
| Storage (VAE .pth) | 2.82 GB | FP32 |

## Data Layouts / Formats

### Repository file tree (HuggingFace `bytedance-research/Lance`)

Total ≈ 57.4 GB.

```
bytedance-research/Lance/
├── .gitattributes                 6.71 kB
├── README.md                     38.7 kB
├── README_zh.md                  37.6 kB
├── config.json                     901 B    # top-level metadata, not a model config
├── Wan2.2_VAE.pth                2.82 GB    # PyTorch .pth (NOT safetensors)
├── Lance_3B/                                # image-only specialist
│   ├── model.safetensors        24.7 GB    # Lance backbone + connectors
│   ├── llm_config.json           1.37 kB
│   ├── generation_config.json     216 B
│   ├── tokenizer.json            7.03 MB
│   ├── vocab.json                2.78 MB
│   └── merges.txt                1.67 MB
├── Lance_3B_Video/                          # unified image + video
│   ├── model.safetensors        28.4 GB
│   ├── llm_config.json           1.37 kB
│   ├── generation_config.json     216 B
│   ├── tokenizer.json            7.03 MB
│   ├── tokenizer_config.json     5.7  kB
│   ├── vocab.json                2.78 MB
│   └── merges.txt                1.67 MB
├── Qwen2.5-VL-ViT/                          # frozen semantic ViT
│   ├── config.json               552 B
│   └── vit.safetensors          1.34 GB
└── assets/                                  # 65 README images / gifs / mp4s
```

**SHA-256 (Lance_3B/model.safetensors):** `06e413d5827a06921fac327ce46db2569a05107ca9723076176809dca1294563`.

### Expected (not yet dumped) tensor key prefixes inside `model.safetensors`

Based on the Python module hierarchy in `Lance(PreTrainedModel)`:

```
language_model.model.embed_tokens.weight
language_model.model.layers.{0..35}.self_attn.q_proj.{weight}
language_model.model.layers.{0..35}.self_attn.k_proj.{weight}
language_model.model.layers.{0..35}.self_attn.v_proj.{weight}
language_model.model.layers.{0..35}.self_attn.o_proj.{weight}
language_model.model.layers.{0..35}.self_attn.q_proj_moe_gen.{weight}        # MoT gen path
language_model.model.layers.{0..35}.self_attn.k_proj_moe_gen.{weight}
language_model.model.layers.{0..35}.self_attn.v_proj_moe_gen.{weight}
language_model.model.layers.{0..35}.self_attn.o_proj_moe_gen.{weight}
language_model.model.layers.{0..35}.mlp.{gate_proj,up_proj,down_proj}.weight
language_model.model.layers.{0..35}.mlp_moe_gen.{gate_proj,up_proj,down_proj}.weight
language_model.model.layers.{0..35}.input_layernorm.weight
language_model.model.layers.{0..35}.post_attention_layernorm.weight
language_model.model.layers.{0..35}.input_layernorm_moe_gen.weight
language_model.model.layers.{0..35}.post_attention_layernorm_moe_gen.weight
language_model.model.norm.weight
language_model.lm_head.weight                            # tied to embed_tokens unless untie_lm_head()
vit.{patch_embed, blocks.{0..31}.*, norm, merger}.weight # Qwen2.5-VL ViT
connector.{0,2}.{weight,bias}                            # MLPconnector
vae_in.{weight,bias}                                     # Linear(192→2048)
vae_out.{weight,bias}                                    # Linear(2048→192)
time_embedder.mlp.{0,2}.{weight,bias}                    # sinusoidal-MLP timestep
pos_embed_3d.*                                           # frozen 3D sincos
task_embed.weight        (10 × 1280, optional)
modality_embed.weight    (10 × 1280, optional)
```

**These names must be confirmed locally** by `safetensors_metadata --print-keys` on `Lance_3B/model.safetensors` — see Open Questions § 1.

### Token / chat layout

Standard Qwen2.5-VL chat template:

```
<|im_start|>system
<system prompt>
<|im_end|>
<|im_start|>user
<text>
<|vision_start|><|image_pad|><|vision_end|>           # for image input
<|vision_start|><|video_pad|><|vision_end|>           # for video input
<|im_end|>
<|im_start|>assistant
```

The vision-pad tokens are placeholders that the model replaces with actual ViT features (understanding) or VAE-noisy latents (generation) at the sequence-packing stage. Generation-only T2I/T2V prompts still use `<|vision_start|>…<|vision_end|>` as the slot where noisy-target VAE latents go.

## Algorithm Steps

### Image generation (T2I, `Lance_3B`, `LanceImagePipeline`)

```
1.  tokens = Qwen2Tokenizer.encode_chat(prompt, system_prompt, append_vision_slot=True)
2.  ids, pos_ids = build_sequence(
        text_tokens   = tokens,
        noisy_slot    = (H/16, W/16) at (t=1) VAE-latent grid,
        modality_roles = [text]*len(tokens) + [noisy_vae]*N_noisy,
    )
3.  apply MaPE Δ_m to pos_ids[t-axis] per role
4.  noise   = sample_normal(B, 48, 1, H/16, W/16)
5.  latents = noise.clone()
6.  steps   = logit_normal_shift(linspace(1, 0, 31), shift=3.5)
7.  for k in 0..29:
        t  = steps[k]; t_next = steps[k+1]; dt = t - t_next
        x_t = TimestepEmbedder(t) added to noisy slot

        # MoT-aware forward
        h = embed(ids)                                  # [seq, hidden]
        h[vae_slots] += vae_in(patchify(latents))       # 192-dim → 2048
        h[vae_slots] += PositionEmbedding3D(t,h,w)
        h[vit_slots] += MLPconnector(vit(image_cond))   # only if vision-conditioned

        for layer in 36:
            apply MoTDecoderLayer (dual QKV, dual norms, dual MLP, M-RoPE+MaPE)

        v = vae_out(h[noisy_slot])
        v = unpatchify(v)                               # back to (B, 48, 1, H/16, W/16)

        # 3-way CFG
        v_cond, v_text_uncond, v_text_vision_uncond = (run forward 3×)
        v_final = v_tv_uncond + cfg_text * (v_cond - v_text_uncond)
                              + cfg_vision * (v_text_uncond - v_tv_uncond)
        if cfg_renorm: renorm v_final against v_cond

        latents = latents - dt * v_final
8.  z = (latents - 0) / 1                               # already normalized
9.  pixels = Wan2_2_VAE.decode(latents * std[None,:,None,None,None] + mean[...])
10. return pixels                                       # (B, 3, 1, H, W) → save first frame
```

For an image-only KV-cache path, steps 6–7 reuse text + ViT + clean-VAE K/V across iterations (only `noisy_slot` Q/K/V is recomputed).

### Video generation (T2V, `Lance_3B_Video`, `LanceVideoPipeline`)

Same structure with `T = ceil(num_frames / 4) + 1` latent frames (4× temporal downsample), `timestep_shift = 4.0`, and PositionEmbedding3D varying over t. Decode produces `(B, 3, num_frames, H, W)`.

### Image editing

Identical to T2I but the input ViT image AND its VAE encode are *both* injected as clean-VAE / ViT-semantic slots before the noisy-target slot. The model conditions on the source image via attention.

### Image understanding (VQA / caption)

The "und" stream only. ViT features + text prompt are packed and the LLM does autoregressive next-token generation (no flow-matching, no VAE-out). Sampling uses `do_sample=True`, `temperature=1e-6` (≈ greedy), `repetition_penalty=1.05` per `generation_config.json`.

## Reference Implementations

**Primary:** [`github.com/bytedance/Lance`](https://github.com/bytedance/Lance)

Source-of-truth files:
- [`inference_lance.py`](https://github.com/bytedance/Lance/blob/main/inference_lance.py) — entry point. `init_from_model_path_if_needed`, `validate_on_fixed_batch`, `apply_inference_defaults`, `save_prompt_results`, `main()`.
- [`inference_lance.sh`](https://github.com/bytedance/Lance/blob/main/inference_lance.sh) — accelerate wrapper; default args.
- [`modeling/lance/lance.py`](https://github.com/bytedance/Lance/blob/main/modeling/lance/lance.py) — `LanceConfig`, `Lance(PreTrainedModel)`, `validation_gen`, `validation_gen_KVcache`, `validation_video_to_text`, `validation_video_to_text_KVcache`.
- [`modeling/lance/qwen2_navit.py`](https://github.com/bytedance/Lance/blob/main/modeling/lance/qwen2_navit.py) — `PackedAttention`, `PackedAttentionMoT`, `Qwen2DecoderLayer`, `Qwen2MoEDecoderLayer`, **`Qwen2MoTDecoderLayer`** (Lance uses this), `NaiveCache`, `Qwen2Model`, `Qwen2ForCausalLM`, `get_rope_index`, `create_sparse_mask`, `freeze_und_params`, `init_moe`, `untie_lm_head`.
- [`modeling/lance/modeling_utils.py`](https://github.com/bytedance/Lance/blob/main/modeling/lance/modeling_utils.py) — `TimestepEmbedder`, `MLPconnector`, `PositionEmbedding`, `PositionEmbedding3D`, `get_*_sincos_pos_embed`.
- [`modeling/qwen2_5_vl/configuration_qwen2_5_vl.py`](https://github.com/bytedance/Lance/blob/main/modeling/qwen2_5_vl/configuration_qwen2_5_vl.py) and [`modeling_qwen2_5_vl.py`](https://github.com/bytedance/Lance/blob/main/modeling/qwen2_5_vl/modeling_qwen2_5_vl.py).
- [`modeling/vit/qwen2_5_vl_vit.py`](https://github.com/bytedance/Lance/blob/main/modeling/vit/qwen2_5_vl_vit.py) — frozen `Qwen2_5_VisionTransformerPretrainedModel`.
- [`modeling/vae/wan/model.py`](https://github.com/bytedance/Lance/blob/main/modeling/vae/wan/model.py) — `WanVideoVAE` (v2.2) wrapper; `vae_encode` / `vae_decode`.
- [`modeling/vae/wan/vae2_2.py`](https://github.com/bytedance/Lance/blob/main/modeling/vae/wan/vae2_2.py) — `Wan2_2_VAE`, `WanVAE_`, `Encoder3d`, `Decoder3d`, `CausalConv3d` (with `CACHE_T=2`), `mean[48]`, `std[48]`, `patch_size=2`.
- [`config/examples/*.json`](https://github.com/bytedance/Lance/tree/main/config/examples) — per-task JSON configs (T2I/T2V/edit/understanding). **Not yet read.**

**Paper:** [arXiv:2605.18678](https://arxiv.org/abs/2605.18678) v2 (2026-05-20). 34 pages, 14 figures, 10 tables. Authors: Fengyi Fu, Mengqi Huang, Shaojin Wu (co-first); Jianzhu Guo (corresp., project lead); + 10 others.

**Project page:** [lance-project.github.io](https://lance-project.github.io/) — demo videos.

**HuggingFace:** [`bytedance-research/Lance`](https://huggingface.co/bytedance-research/Lance) — weights.

**Wan2.2 VAE background:** [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) (lineage), [DeepWiki Wan VAE variants](https://deepwiki.com/ModelTC/lightx2v/8.1-wan-vae-architecture-variants).

## Differences Between Implementations

There is **only one reference implementation** (the ByteDance codebase). No diffusers PR exists upstream as of 2026-05-24. No community ports (ComfyUI, sd-webui, etc.) have surfaced yet — this model is fresh.

## Open Questions

1. **Exact tensor key names in `model.safetensors`** — `Lance_3B/model.safetensors` is Xet-backed and too large to inline via the HF web viewer. The names above (§ Data Layouts) are my best inference from `Lance(PreTrainedModel)` source; HartsyInference's `LanceCheckpointConverter` should be written after dumping the actual keys with a small Python helper:
   ```python
   from safetensors import safe_open
   with safe_open("Lance_3B/model.safetensors", framework="pt") as f:
       for k in sorted(f.keys()): print(k)
   ```

2. **MaPE Δ_m offset values** — formula is given but the integer offsets per modality role live inside `get_rope_index()` (`modeling/lance/qwen2_navit.py`). Must be read locally and reproduced exactly; the ablation shows this is load-bearing for editing tasks.

3. **`llm_qk_norm` default in released checkpoints** — inference shell exposes the flag but its default isn't shown. Confirm by checking whether `q_norm` / `k_norm` weights exist in the safetensors key dump.

4. **`cfg_vision_scale` default** for image-edit / video-edit tasks. Only `cfg_text_scale=4.0` is in `inference_lance.sh`. Read `config/examples/{image_edit,video_edit}.json`.

5. **CFG renorm mode + `cfg_renorm_min` defaults** — `global` vs `channel` vs off; first-pass impl can default to off and add toggle later.

6. **Task / modality embedding usage** — `nn.Embedding(10, 1280)` for both task and modality are declared. Verify whether released weights actually populate them or zero them out. If used, document the task-id → task-name mapping (likely: 0=t2i, 1=t2v, 2=image_edit, 3=video_edit, 4=x2t_image, 5=x2t_video — to be confirmed).

7. **Per-task `config/examples/*.json`** — exact resolution presets, default frame counts, prompt templates.

8. **Sparse attention block-mask layout** — `create_sparse_mask(sample_lens, split_lens, attn_modes)` with `BLOCK_SIZE=128`. HartsyInference does not have a varlen-FlashAttention equivalent yet; the first port can use a padded dense mask and accept the quadratic cost on padding tokens, then optimize later. The exact `attn_modes` enum values need to be confirmed in source.

9. **`AutoEncoderParams` full field list** — only `downsample_spatial`, `downsample_temporal`, `z_channels` were extracted from the constructor call. Other fields (if any) need to be read from `modeling/vae/wan/vae2_2.py`.

10. **VAE scalar `scaling_factor` (if any)** beyond the embedded per-channel mean/std. Wan2.1 used a single scalar; Wan2.2 appears to use only the 48-vector. Confirm during integration — there should not be an additional global scale to multiply.

11. **CausalConv3d frame-cache (`CACHE_T=2`) plumbing** — required for chunked/streaming video decode without OOM. The image-only path can ignore this and decode at T=1. The video path needs it for any clip > a single chunk.

12. **MoT routing convention in `model.safetensors`** — whether the gen-path weights live in a `*_moe_gen` parameter suffix (as `qwen2_navit.py` suggests) or are stored as fused tensors that need splitting at load. Possibly the simplest case: every `*` key has an optional `*_moe_gen` sibling. Confirm via key dump (Q § 1).

## Implementation Notes

### How this maps to HartsyInference packages

- **`HartsyInference.Diffusion`** — adds:
  - `Models/Denoisers/LanceConfig.cs`, `Models/Denoisers/LanceTransformer.cs`, `Models/Denoisers/DiTBlocks/LanceMoTBlock.cs`, `Models/Denoisers/DiTBlocks/LanceMRopeMaPE.cs`, `Models/Denoisers/LanceDebugDump.cs`.
  - `Models/TextEncoders/Qwen25VlVit.cs` (frozen ViT — only the forward + load; no training paths).
  - `Models/Vae/Wan22VaeConfig.cs` and a new `Models/Vae/Wan22VaeDecoder.cs` (3D causal VAE). Possibly factor a shared `IVaeDecoder3D` interface so this and any future Wan/LTX VAEs share Box.
  - `Pipelines/LanceImagePipeline.cs` (image-only path, single-frame T=1 decode).
- **`HartsyInference.Video`** — adds:
  - `Pipelines/LanceVideoPipeline.cs` (multi-frame T>1 decode, frame streaming).
  - Shares `Wan22VaeDecoder.cs` with Diffusion (best: live in `HartsyInference.Diffusion` and import from Video).
- **`HartsyInference.ModelAssets`** — adds:
  - `CheckpointConverters/LanceCheckpointConverter.cs` (loads `model.safetensors`, splits into `language_model.*` / `vit.*` / `connector.*` / `vae_in.*` / `vae_out.*` / `time_embedder.*` / `pos_embed_3d.*` / `task_embed` / `modality_embed` buckets and demuxes the MoT `_moe_gen` sibling weights into per-stream dicts).
  - Wan2.2 `.pth` reader path (or convert to safetensors offline; the existing safetensors loader does not parse `.pth`). Simplest: ship a one-off Python script that converts `Wan2.2_VAE.pth` → `wan22_vae.safetensors` for users.
- **`HartsyInference.ModelAssets.Tokenizers`** — Qwen2 BPE is already covered by the existing Qwen support (`Qwen3Tokenizer`). Lance uses Qwen2 vocab (151,936); the existing BPE tokenizer with this vocab works. Chat template needs the Qwen2.5-VL pad-insertion variant.

### Net-new backend / kernel work required

1. **Packed / variable-length attention.** Lance's `PackedAttentionMoT` runs one `flash_attn_varlen_func` call over a packed `[token_0, token_1, ...]` buffer with per-sample lengths. Equivalents needed in `IBackend`:
   - `IBackend.PackedAttention(q, k, v, cu_seqlens, max_seqlen, block_mask, ...)` — first-pass can fall back to padded dense + bool mask (correct but slow on big sequences). A real varlen kernel (CUDA flash-attn-style) is a follow-up.
   - This is **shared with future Wan/LTX video pipelines** so the cost amortizes across Phase 9.
2. **3D Causal Conv** kernel — `Wan22VaeDecoder` uses `CausalConv3d` with `CACHE_T=2` frame cache. Need:
   - `IBackend.Conv3D(input, weight, bias, stride, padding, dilation)` — straight 3D im2col + GEMM is fine for v1; specialized streaming kernels are a follow-up.
   - A streaming-decode helper that maintains a per-conv 2-frame cache (so `LanceVideoPipeline` can decode chunks of frames without VRAM OOM).
3. **MoT routing primitive.** The per-layer dispatch is "gather tokens by role, run path-specific Q/K/V/O, scatter back, run one joint attention". Two viable implementations:
   - Re-batched dispatch — split the packed sequence into two contiguous sub-batches (und + gen), run `Linear` per path, scatter results back into the joint sequence. Works on any backend. Costs an extra two `Gather`/`Scatter` per layer.
   - Branchless dual-matmul — run both Linears on the full sequence and mask-select per role. Simpler to plumb but wastes ~2× compute on every linear.
   - First-pass: re-batched dispatch.
4. **3D sin-cos position embedding** (`PositionEmbedding3D`) — easy CPU-side precompute, upload once; no new kernel work.
5. **`TimestepEmbedder`** is the standard sinusoidal-MLP — already implemented in `DiTUtils`.
6. **GQA-2** is an extreme GQA factor (Lance uses 16 Q : 2 KV). Existing GEMM paths handle this fine, but make sure attention reshape doesn't accidentally assume `n_kv >= 4` anywhere (Z-Image uses 32:32 = no GQA; Qwen3-4B uses 32:8 = factor 4; Lance is **factor 8** — the most extreme so far).
7. **3-way CFG with renorm** — pipeline-level loop change, not a kernel. Three forward passes per step (vs the usual 2 for SDXL/Flux). On 12 GB cards the noisy slot has to be small enough to fit 3× activations; image at 512×512 should still work.
8. **KV-cache for diffusion (`NaiveCache`)** — significant pipeline change. Diffusion pipelines in HartsyInference today recompute the entire sequence every step. To match Lance's `validation_gen_KVcache` speedup, the pipeline needs to:
   - On step 0, cache K/V for the text + ViT + clean-VAE prefix.
   - On steps 1..N, only recompute Q/K/V for the noisy slot, append to the cached K/V, run attention against the joined K/V.
   - This is the first inference-time KV-cache use case in HartsyInference and will eventually be reused by Wan and LTX. Worth designing as a reusable `DenoiseKvCache` helper in `HartsyInference.Diffusion/Utilities/`.

### VRAM and viability per target GPU

| GPU | VRAM | Image (Lance_3B) | Video (Lance_3B_Video) |
|---|---|---|---|
| RTX 3060 12 GB | 12 GB | Tight — Lance_3B fp16 ≈ 12.4 GB transformer alone; will need FP8 cast at load (~6.2 GB) plus ViT (~0.7 GB) plus VAE (~1.4 GB fp16) ≈ 8.3 GB. Should fit at 512 / 768. |  Won't fit — 28.4 GB safetensors → ~14 GB FP8; plus video activations explode at 121 frames. |
| RTX 4090 24 GB | 24 GB | Comfortable FP16 (12.4 GB transformer + 1.4 GB ViT + 2.8 GB VAE ≈ 17 GB; +activations). | Tight at FP16; comfortable at FP8 (~14 GB transformer + …). Short clips OK. |
| A100 40 / 80 GB | 40+ | Comfortable. | Comfortable. |

Recommended quality presets (per `QualityProfileApplier`):
- Image: `High` = FP8 transformer + FP16 ViT + FP16 VAE.
- Video: `High` on a 24 GB card limits to ~50 frames @ 480p; `Maximum` (FP16) requires 40+ GB.

Q4_K GGUF dumps are not yet available for Lance (model just released). When `unsloth/Lance-GGUF` or similar appears, a Q4_K transformer (~3 GB) makes the image path trivial on 12 GB and the video path feasible on 24 GB.

### Ordering / dependencies for the build

1. **Land MoT + packed attention first.** This is the high-risk net-new infra. Validate on image-only T2I @ 512×512 against `Lance_3B`. Use the standard layer-by-layer Python diff harness (see SD3.5 / Z-Image patterns in `TROUBLESHOOTING.md`).
2. **Then add ViT (frozen forward).** Once T2I works text-only, plug the ViT into image-editing.
3. **Then add Wan2.2 VAE.** First with `T=1` (image), reusing existing 2D Conv2D paths where the 3rd dim is 1; then promote `CausalConv3d` to a real 3D conv for video.
4. **Finally, video pipeline + frame streaming + 4× temporal decode chunking.** Phase 9.

### Test-skipping discipline

Following the project convention (every `*GenerationTests` skips cleanly when env vars or VRAM are missing):

- `LanceImagePipelineTests` should require: `LANCE_3B_PATH`, `LANCE_VIT_PATH`, `LANCE_VAE_PATH`, plus PTX dir; VRAM probe ≥ 8 GB free for FP8 / ≥ 16 GB for FP16.
- `LanceVideoPipelineTests` (Phase 9) should additionally require: `LANCE_3B_VIDEO_PATH`; VRAM probe ≥ 24 GB for short-clip FP8; skip cleanly when frame count would exceed memory.

### Reuse opportunities

- `LlamaStyleEncoder` and `QkNorm` / `RmsNorm` / `SwiGluFfn` / `AdaLNModulation` sub-components are reusable — Lance is mostly a Qwen2-shaped backbone. The MoT dual-stream is the only genuinely novel block.
- `Qwen2Tokenizer` from the existing `Qwen3Tokenizer` infrastructure (same Qwen2 BPE format, vocab 151,936). Confirm vocab/merges files line up.
- `FlowMatchEulerDiscreteScheduler` with `shift = 3.5 / 4.0` matches the Z-Image / Flux flow-matching path; reuse `FlowMatchEulerDiscreteScheduler.cs`.
- `SinusoidalTimestepEmbedding` from `DiTUtils` is unchanged.
- The 3D sin-cos position embed has no analogue yet in the codebase — add a small `PositionEmbedding3D` helper in `DiTUtils`.
- The Wan2.2 VAE will be the **first 3D causal VAE** in HartsyInference and should be designed as a foundation other Wan-family / LTX video VAEs can extend.

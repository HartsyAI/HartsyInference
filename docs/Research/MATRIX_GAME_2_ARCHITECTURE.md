# Matrix-Game 2.0 — Research Notes

> Status: Complete (HF model card + GitHub source + paper + per-task configs captured; safetensors key dump still required) | Last Updated: 2026-05-24 | Needed Before: HartsyInference.World (Matrix-Game 2.0 pipeline, Phase 10)
> Source of truth: [SkyworkAI/Matrix-Game GitHub](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-2), [HF `Skywork/Matrix-Game-2.0`](https://huggingface.co/Skywork/Matrix-Game-2.0), [arXiv 2508.13009](https://arxiv.org/abs/2508.13009)
> License: **MIT** (confirmed on HF model card and on the Matrix-Game-2 GitHub README)
> Related: future `MATRIX_GAME_3_ARCHITECTURE.md` (the 5B sibling), [`FLOW_MATCHING_AUDIO.md`](FLOW_MATCHING_AUDIO.md) (flow-matching scheduler background), [`VAE_ARCHITECTURE.md`](VAE_ARCHITECTURE.md), [`FLUX_ARCHITECTURE.md`](FLUX_ARCHITECTURE.md) (AdaLN modulation lineage)

> **Stub.** The narrative walkthrough, restated pseudocode and resolved open questions were
> removed on 2026-08-06 — this model is built and verified, so the C# is the source of truth for
> *how it works*. What remains is what the code cannot tell you: upstream provenance, reference
> constants to diff a suspect port against, where implementations disagree, and bring-up traps.
> Full history is in git. Parity evidence: `docs/Checklists/PARITY_VERIFICATION.md`.

## Summary

**Matrix-Game 2.0** is Skywork AI's open-source 1.8B-parameter **interactive world model** — given a single reference image and a stream of keyboard + mouse actions, it autoregressively synthesizes a controllable video at **~25 FPS @ 352 × 640** on a single H100. It is the **entry-level / low-VRAM sibling** of the 5B Matrix-Game 3.0 (40 FPS @ 720p) and a successor to the 17B Matrix-Game 1.0 (offline only).

Architecturally, Matrix-Game 2.0 is **SkyReels-V2-I2V-1.3B-540P** (a Wan2.1 family 1.3B I2V DiT) with the text branch removed and per-block **ActionModules** added to the **first 15 of its 30 DiT layers** (the foundation model puts ActionModules in all 30 — the distilled checkpoints prune to layers 0–14). The "+0.5B" of new parameters is exactly those ActionModules (mouse MLP + self-attention + projection + keyboard MLP + cross-attention + projection per inserted block, totalling ~500M new weights).

Real-time streaming is achieved by **(a)** converting the bidirectional DiT into a **block-wise causal** transformer (each frame attends only to its own frame and a sliding window of the last 6 frames), **(b)** **DMD + Self-Forcing few-step distillation** of the original ~1000-step teacher into a **3- or 4-step student** (`denoising_step_list = [1000, 666, 333]` for universal/gta, `[1000, 750, 500, 250]` for templerun), and **(c)** **rolling KV cache** that evicts the oldest frame tokens once the local-attention window of 6 frames is exceeded.

The VAE is the **Wan2.1 3D causal VAE** (16 latent channels, **8 × 8 spatial / 4 × temporal** compression, 508 MB). Input image conditioning is dual: the input image is **VAE-encoded** (concatenated with the noisy latent along the channel dim → `in_dim = 36`) **and** passed through a frozen **CLIP-ViT-H/14 XLM-RoBERTa** encoder (4.77 GB) for "visual context" that is consumed by the I2V cross-attention. Action conditioning is **deterministic** (no learned router): every other denoising step, current-frame mouse deltas pass through an MLP and self-attention (acting like RoPE'd cross-attention over the spatial token grid) and keyboard one-hots pass through MLP + cross-attention.

For HartsyInference, Matrix-Game 2.0 is best treated as a **new `HartsyInference.World` package built on top of a Wan2.1-family DiT backbone** that will be shared with Matrix-Game 3.0 once we add it. The same DiT, RoPE, AdaLN, action-module, KV-cache, and Wan VAE primitives are reusable across both — only the backbone weights (1.3B vs 5B), action-block coverage (15 vs 30 blocks), and resolution (352×640 vs 1280×720) change.

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

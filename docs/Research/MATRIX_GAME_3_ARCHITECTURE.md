# Matrix-Game 3.0 — Research Notes

> Status: Complete (model card + arXiv v2 paper + GitHub source code + Wan2.2 base config captured; only safetensors tensor-key dump remains as a local follow-up) | Last Updated: 2026-05-24 | Needed Before: HartsyInference.World (Matrix-Game 3.0 pipeline, Phase 10)
> Source of truth: [HF `Skywork/Matrix-Game-3.0`](https://huggingface.co/Skywork/Matrix-Game-3.0), [GitHub `SkyworkAI/Matrix-Game`](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-3), [arXiv 2604.08995 v2](https://arxiv.org/abs/2604.08995v2), [project page](https://matrix-game-v3.github.io/), [base `Wan-AI/Wan2.2-TI2V-5B`](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)
> License: Apache-2.0 (Matrix-Game 3.0 code + weights), Apache-2.0 (Wan2.2-TI2V-5B base), Apache-2.0 (UMT5-XXL encoder). No model-card-imposed use restrictions beyond standard Apache-2.0 terms.
> Related: [`LANCE_ARCHITECTURE.md`](LANCE_ARCHITECTURE.md) (Wan2.2 3D causal VAE — exact same `Wan2.2_VAE.pth`, same 48-channel latent, same mean/std), [`TEXT_ENCODERS.md`](TEXT_ENCODERS.md) (UMT5-XXL is also used by AuraFlow / Pile-T5-XL), [`FLOW_MATCHING_AUDIO.md`](FLOW_MATCHING_AUDIO.md) (rectified-flow background; Matrix-Game uses FlowUniPC).

> **Stub.** The narrative walkthrough, restated pseudocode and resolved open questions were
> removed on 2026-08-06 — this model is built and verified, so the C# is the source of truth for
> *how it works*. What remains is what the code cannot tell you: upstream provenance, reference
> constants to diff a suspect port against, where implementations disagree, and bring-up traps.
> Full history is in git. Parity evidence: `docs/Checklists/PARITY_VERIFICATION.md`.

## Summary

Matrix-Game 3.0 (Skywork AI Matrix-Game Team, arXiv:2604.08995, 2026-03 initial release) is a **memory-augmented interactive world model** that performs real-time streaming video generation at **720p (704×1280) @ 40 FPS** on multi-GPU A/H-series hardware. It is a finetune of **Wan2.2-TI2V-5B** with two structural additions: an **ActionModule** (keyboard via cross-attention, mouse via self-attention, attached to a subset of DiT blocks) and a **camera-aware long-horizon memory mechanism** that retrieves 5 past latent frames by camera-frustum overlap and injects them through the same joint-attention space as the noisy prediction. The underlying DiT is the **Wan2.2 5B dense** transformer (40 layers, dim 5120, 40 heads, head_dim 128, FFN 13824, RMSNorm with QK-norm and cross-attn-norm), patchified `(1,2,2)` over 48-channel VAE latents, with a 4× temporal / 16×16 spatial Wan2.2 3D causal VAE. The text branch is **UMT5-XXL** with 512-token context.

Two checkpoints ship under one HuggingFace repo: a **base model** (12.9 GB safetensors, 50-step FlowUniPC inference, sample_shift=5.0, CFG=5.0) and a **base_distilled_model** (25.9 GB safetensors — larger because it bundles student + critic / EMA — runs at **3 inference steps** via multi-segment Distribution Matching Distillation). Inference is autoregressive over **segments of latent length 15** (57 RGB frames for the first segment, 40 for every subsequent segment); the next segment conditions on the last 4 past latent frames plus 5 retrieved memory frames plus the new noisy prediction. A separate **MG-LightVAE** decoder (a pruned distillation of the Wan2.2 VAE decoder; 50 % or 75 % pruning shipping as `MG-LightVAE.pth` 2.74 GB and `MG-LightVAE_v2.pth` 841 MB) replaces the Wan2.2 decoder at inference time for a 2.6× / 5.2× decode speedup. There is also a paper-only **2×14B MoE** variant ("Coming Soon") that splits high-noise denoising between a first-person expert and a third-person expert; the 5B is what's actually downloadable today.

For HartsyInference this is a Phase-10 `HartsyInference.World` pipeline that reuses substantially all of the work needed for a Wan2.2 video pipeline (DiT backbone, 3D causal VAE, UMT5-XXL encoder, FlowUniPC scheduler) and adds three new pieces: (1) the **ActionModule** block (a small ~16-head dual-attention block with its own RoPE θ=256), (2) a **camera-pose + Plücker-embedding** preprocessor, and (3) a streaming **per-segment loop** that maintains the past-frame buffer, the 5-slot memory cache, and routes decoding to an async worker (or to a single in-process VAE call on a smaller install).

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

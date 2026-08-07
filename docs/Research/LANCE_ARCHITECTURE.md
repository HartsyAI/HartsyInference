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
> - Engine: `LanceCheckpointConverter`/`LanceTransformer`/`LancePipelineCommon`/`LanceImagePipeline` reconciled; parity via `tests/python-reference/dump_lance_reference.py` + `diff_lance_layers.py` (the C# parity test was removed 2026-08-06).
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

> **Stub.** The narrative walkthrough, restated pseudocode and resolved open questions were
> removed on 2026-08-06 — this model is built and verified, so the C# is the source of truth for
> *how it works*. What remains is what the code cannot tell you: upstream provenance, reference
> constants to diff a suspect port against, where implementations disagree, and bring-up traps.
> Full history is in git. Parity evidence: `docs/Checklists/PARITY_VERIFICATION.md`.

## Summary

Lance ("Unified Multimodal Modeling by Multi-Task Synergy", ByteDance Research, May 2026) is a **single 3B-active-parameter** native unified multimodal model. One checkpoint covers six tasks: text-to-image (T2I), text-to-video (T2V), image editing, video editing, image understanding (VQA / captioning), and video understanding. There are **two release variants** sharing the same architecture: `Lance_3B` (image-only specialist, 24.7 GB safetensors) and `Lance_3B_Video` (general image+video, 28.4 GB safetensors).

The backbone is a **modified Qwen2.5-VL 3B decoder** (36 layers / hidden 2048 / 16 heads / 2 KV heads / FFN 11008 / RMSNorm / SwiGLU / GQA factor 8). The novel piece is a per-layer **dual-stream "MoT" (Mixture-of-Tokens) routing**: every token is *also* annotated with a modality role (text-und, ViT-semantic, clean-VAE, noisy-VAE) and is sent through one of two parallel sets of (Q, K, V, O, gate_proj, up_proj, down_proj, input_layernorm, post_attention_layernorm) — the standard set for **understanding** tokens and a parallel `*_moe_gen` set for **generation** tokens. Both streams attend to each other through a single shared joint attention. On top of this Lance applies **MaPE** (Modality-Aware Positional Encoding), which shifts only the *temporal* axis of the model's 3D M-RoPE by a per-role offset Δ_m so the four token roles don't collide in the shared position space.

Generation is **rectified-flow velocity prediction** with logit-normal timestep shifting (3.5 for image, 4.0 for video), explicit Euler integration over a default 30 steps, and **three-way CFG** (text + vision conditional, default text scale 4.0, optional renorm). Understanding is autoregressive next-token over the LLM's text vocab. The frozen Qwen2.5-VL ViT (32 layers / 1280 dim / 14-px patches / 4-frame temporal patches / windowed attention with full-attention at indices 7/15/23/31) provides semantic vision tokens; the frozen **Wan2.2 3D causal VAE** (48 latent channels, 16× spatial / 4× temporal downscale, RMSNorm, CausalConv3d with 2-frame cache) provides generation latents. Latents are patchified `(t,h,w)=(1,2,2)` → 192-dim tokens → `Linear(192→2048)` and joined back via `Linear(2048→192)` after the transformer.

For HartsyInference, Lance is best treated as **two related pipelines sharing one transformer backbone**: a Phase 4 (image breadth) `LanceImagePipeline` using `Lance_3B`, and a Phase 9 (video) `LanceVideoPipeline` using `Lance_3B_Video`. The image path can ship without 3D temporal infra; the video path requires the Wan2.2 3D causal VAE and CausalConv3d streaming-decode plumbing. Either path requires net-new backend work for **packed/var-length attention** (FlashAttention's `flash_attn_varlen_func`) and **MoT routing** (dispatching different weights for different token slices in the same attention call).

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

> **Superseded 2026-08-06.** The per-model pipeline/generation tests this section specified were
> removed in the test-suite cleanup, and the rule is now the opposite: **do not add a test that
> proves a model works end to end** — a model that stops working is visible the moment anyone uses
> it. Test what breaks quietly instead (kernel numerics, cross-device equivalence, quantization and
> codec round-trips, padding/tiling geometry, format and key mapping), and put shared-component
> parity in `tests/<Project>/Parity/` with a `*ParityTests` name. See `docs/CODE_STYLE.md` §Testing.

### Reuse opportunities

- `LlamaStyleEncoder` and `QkNorm` / `RmsNorm` / `SwiGluFfn` / `AdaLNModulation` sub-components are reusable — Lance is mostly a Qwen2-shaped backbone. The MoT dual-stream is the only genuinely novel block.
- `Qwen2Tokenizer` from the existing `Qwen3Tokenizer` infrastructure (same Qwen2 BPE format, vocab 151,936). Confirm vocab/merges files line up.
- `FlowMatchEulerDiscreteScheduler` with `shift = 3.5 / 4.0` matches the Z-Image / Flux flow-matching path; reuse `FlowMatchEulerDiscreteScheduler.cs`.
- `SinusoidalTimestepEmbedding` from `DiTUtils` is unchanged.
- The 3D sin-cos position embed has no analogue yet in the codebase — add a small `PositionEmbedding3D` helper in `DiTUtils`.
- The Wan2.2 VAE will be the **first 3D causal VAE** in HartsyInference and should be designed as a foundation other Wan-family / LTX video VAEs can extend.

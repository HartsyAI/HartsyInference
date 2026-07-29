# Kandinsky 5.0 T2V (Video) — Research Notes

> Status: Research complete — **T2V/I2V Lite IMPLEMENTED 2026-06-12** (structurally verified end-to-end on CPU via synthetic weights; numeric validation vs the real checkpoint pending). Pro 19B is config-only support.
> Source of truth: HF `kandinskylab/Kandinsky-5.0-T2V-Lite-*-Diffusers` (`transformer/config.json`, `vae/config.json`, `scheduler/scheduler_config.json`) + diffusers source verified verbatim against the local reference venv:
> `diffusers/models/transformers/transformer_kandinsky.py`, `diffusers/pipelines/kandinsky5/pipeline_kandinsky.py` (T2V), `pipeline_kandinsky_i2v.py`, `diffusers/models/autoencoders/autoencoder_kl_hunyuan_video.py`.
> License: Apache 2.0 (ai-forever / kandinskylab)
> Related: `Kandinsky5Transformer` / `Kandinsky5Config` / `Kandinsky5Rope` (T2I image variant already built), [`LTX_VIDEO_ARCHITECTURE.md`](LTX_VIDEO_ARCHITECTURE.md), [`WAN_ARCHITECTURE.md`](WAN_ARCHITECTURE.md) (sibling video pipelines), and the diffusion-pipeline conventions in `docs/Agents/AGENTS.md`.

## Summary

Kandinsky 5.0 T2V Lite is a **2B-parameter video DiT** (`Kandinsky5Transformer3DModel`) sharing the exact block architecture with the already-built T2I Lite variant — text encoder blocks + visual decoder blocks with AdaLN modulation, biased QKV/out linears, RMSNorm QK-norm, bias-free `Linear → GELU → Linear` FFN (NOT SwiGLU), and 1D/3D RoPE. The video deltas vs T2I are: smaller dims (model_dim 1792 vs 2560), **temporal latent axis** (T_lat = (frames−1)/4 + 1), **visual conditioning input** (`visual_cond=true` → 33-channel model input = noisy(16) + cond(16) + mask(1)), **resolution-dependent RoPE scale factors** dividing the rope args, and a different VAE — **AutoencoderKLHunyuanVideo** (3D causal, 16-ch latent, 8× spatial / 4× temporal) instead of the Flux image VAE.

Text conditioning is identical to T2I: Qwen2.5-VL-7B last-hidden sequence embeddings (3584-dim, prompt-template prefix of 129 tokens dropped upstream) + CLIP-L pooled (768-dim → projected to time_dim and added to the timestep embedding). **The HartsyInference pipeline takes PRECOMPUTED embeddings** — same contract as the existing `Kandinsky5Pipeline` T2I.

Scheduler: `FlowMatchEulerDiscreteScheduler`, shift **5.0** (diffusers checkpoints; the native single-file release uses 10.0 — config field), v-prediction Euler. Defaults: 50 steps, CFG 5.0 (1.0 for nocfg/distilled variants), 121 frames @ 24 fps, 512×768.

## Transformer — T2V Lite 2B (`transformer/config.json`)

| Field | Lite 2B (video) | Pro 19B (config-only) | T2I Lite 6B (built) |
|---|---|---|---|
| `model_dim` | **1792** | 4096 | 2560 |
| `ff_dim` | **7168** | 16384 | 10240 |
| `time_dim` | **512** | 1024 | 512 |
| `num_text_blocks` | **2** | 4 | 2 |
| `num_visual_blocks` | **32** | 60 | 50 |
| `axes_dims` (t,h,w) | **[16, 24, 24]** → head_dim 64 | [32, 48, 48] → head_dim 128 | [32, 48, 48] |
| heads (`model_dim/head_dim`) | **28** | 32 | 20 |
| `patch_size` | (1, 2, 2) | (1, 2, 2) | (1, 2, 2) |
| `in_visual_dim` / `out_visual_dim` | 16 / 16 | 16 / 16 | 16 / 16 |
| `in_text_dim` / `in_text_dim2` | 3584 / 768 | 3584 / 768 | 3584 / 768 |
| `visual_cond` | **true** | true (i2v) / false (t2v) | false |

- **visual_cond input packing:** `visual_embed_dim = 2*in_visual_dim + 1 = 33` when `visual_cond=true`. Model input per latent pixel = `concat([noisy(16), cond_tensor(16), cond_mask(1)])` (channel-last in diffusers). `visual_embeddings.in_layer` is `Linear(prod(patch_size)*33 = 132 → model_dim)`.
  - T2V: cond tensor and mask are **zeros** (still 33 channels — the weights expect them).
  - I2V: VAE-encode the image → first latent frame: `noisy[:, 0:1] = img_latent` (set once at init; the scheduler only steps frames `1:`), `cond_tensor[:, 0:1] = img_latent`, `mask[:, 0:1] = 1`.
- **RoPE positions** (from `pipeline_kandinsky.py` step 6): `t ∈ [0..T_lat)`, `h ∈ [0..H_lat/2)`, `w ∈ [0..W_lat/2)` (patch units); per-axis freqs `exp(-ln(10000)·i/(d/2))` ≡ `1/10000^(2i/d)`, args **divided by `scale_factor`**.
- **RoPE scale factor** (`Kandinsky5T2VPipeline._get_scale_factor`): `(1, 2, 2)` when both height and width ∈ [480, 854], else `(1, 3.16, 3.16)`. 512×768 → (1, 2, 2). The T2I pipeline passes no scale factor (≡ (1,1,1)); HartsyInference makes it a parameter that defaults to 1.0 so T2I numerics are untouched.
- **Timestep / modulation / out layer:** identical to T2I (sinusoidal dim = model_dim, `in_layer → SiLU → out_layer`; per-block modulation `Linear(time_dim → {6,9}·model_dim)`; out layer `Linear(time_dim → 2·model_dim)` modulation + non-affine LN + `Linear(model_dim → prod(patch)·16)`).
- **CFG (from the reference pipeline):** dual full forward passes, `v = v_uncond + g·(v_cond − v_uncond)`; scheduler steps only the first 16 channels of the packed latent.

### State-dict keys (verified against local diffusers source — top level, no prefix)

```
time_embeddings.{in_layer,out_layer}.{weight,bias}
text_embeddings.in_layer.{weight,bias}             text_embeddings.norm.{weight,bias}
pooled_text_embeddings.in_layer.{weight,bias}      pooled_text_embeddings.norm.{weight,bias}
visual_embeddings.in_layer.{weight,bias}                            # [model_dim, 132] for visual_cond
text_transformer_blocks.{0..1}.text_modulation.out_layer.{weight,bias}     # [6·model_dim, time_dim]
text_transformer_blocks.{i}.self_attention.{to_query,to_key,to_value,out_layer}.{weight,bias}
text_transformer_blocks.{i}.self_attention.{query_norm,key_norm}.weight    # RMSNorm [head_dim]
text_transformer_blocks.{i}.feed_forward.{in_layer,out_layer}.weight       # bias-free
visual_transformer_blocks.{0..31}.visual_modulation.out_layer.{weight,bias} # [9·model_dim, time_dim]
visual_transformer_blocks.{i}.{self_attention,cross_attention}.(same as above)
visual_transformer_blocks.{i}.feed_forward.{in_layer,out_layer}.weight
out_layer.modulation.out_layer.{weight,bias}                                # [2·model_dim, time_dim]
out_layer.out_layer.{weight,bias}
```

**Naming discrepancy note:** secondary web sources describe the attention keys as `to_q/to_k/to_v/to_out.0`; the actual diffusers `Kandinsky5Attention` module (verified in source) uses **`to_query/to_key/to_value/out_layer`** — which is what the already-built T2I converter and `Kandinsky5Block` use 1:1. The `self_attention_norm`/`cross_attention_norm`/`feed_forward_norm` LayerNorms are non-affine (no weights in the state dict).

## Attention: trained with NABLA, implemented DENSE

The checkpoints were trained with **NABLA block-sparse attention** (`attention_type="nabla"` in some configs: P=0.9 top-cdf binarized block map at 64-token granularity, unioned with an STA sliding-tile mask `wT=11, wH=3, wW=3`, dispatched through flex-attention `BlockMask`; see `nablaT_v2` + `fast_sta_nabla` in the diffusers source). Dense SDPA over the same tokens is the mathematical superset (sparse masks only *drop* attention edges that the model learned to mostly ignore — outputs are close but not bit-identical).

HartsyInference implements **dense SDPA**: at the 5 s default (121 frames, 512×768) the visual sequence is 31×32×48 = **47,616 tokens** — large but tractable. For 10 s / 241-frame configs (61 latent frames ≈ 93k tokens) the pipeline logs a warning; NABLA is documented here as a **future optimization** (block-sparse map estimation at 64-token granularity is backend-implementable without flex-attention).

## VAE — AutoencoderKLHunyuanVideo (NEW implementation)

`vae/config.json`: in 3 ch, `latent_channels` 16, `block_out_channels` [128, 256, 512, 512], `layers_per_block` 2, GroupNorm(32, eps 1e-6), SiLU, `scaling_factor` **0.476986**, spatial 8× / temporal 4×, mid-block with 1 attention. Latent math: `T_lat = (F−1)/4 + 1`, `H_lat = H/8`, `W_lat = W/8`.

- **CausalConv3d (HunyuanVideo flavor):** `F.pad(mode="replicate")` with temporal `k−1` **left-only** (replicates the *first frame*) and spatial `k//2` symmetric **replicate** (not zeros!). Maps onto the shared `CausalConv3d` (`replicateFirstPad: true` + new `spatialReplicatePad: true` mode).
- **Encoder:** `conv_in(3→128)` → 4 down blocks (2 resnets each) → mid (resnet, causal attention, resnet) → GroupNorm+SiLU → `conv_out(512→32)` (double-z) → `quant_conv` (Conv3d 1×1×1, 32→32). Down-sampling rule (from source, temporal_compression=4): **block 0 spatial-only (1,2,2); blocks 1–2 spatial+temporal (2,2,2); block 3 none.** Downsample conv: k3, the stated stride, conv padding 0 (the causal replicate-pad supplies the borders).
- **Decoder:** `post_quant_conv` (16→16, 1×1×1) → `conv_in(16→512)` → mid → 4 up blocks (**3 resnets each** = layers_per_block+1) → GroupNorm+SiLU → `conv_out(128→3)`. Upsample rule mirrors the encoder: **up blocks 0 spatial-only, 1–2 spatial+temporal, 3 none.** The upsampler splits the first frame off: frame 0 is spatially nearest-×2 only, frames 1..T−1 get full nearest interpolation (incl. temporal ×2 = each frame repeated twice) → `T_out = 1 + 2·(T−1)`, then CausalConv3d k3 s1. Channel ramp: 512→512→512→256→128.
- **Mid-block attention:** single-head (head_dim = channels = 512), `group_norm` + `to_q/to_k/to_v/to_out.0`, residual, over all `T·H·W` tokens with a **frame-causal mask** (tokens in frame t attend only to frames ≤ t; `prepare_causal_attention_mask`).
- **Keys:** `encoder.conv_in.conv.*`? — **No**: diffusers `HunyuanVideoCausalConv3d` wraps `nn.Conv3d` as `.conv`, so the state-dict keys are `encoder.conv_in.conv.{weight,bias}`, `encoder.down_blocks.{i}.resnets.{j}.{norm1,norm2}.{weight,bias}` + `.{conv1,conv2}.conv.{weight,bias}` (+ `.conv_shortcut.conv.*` when channels change), `encoder.down_blocks.{i}.downsamplers.0.conv.conv.*`, `encoder.mid_block.resnets.{0,1}.*` + `encoder.mid_block.attentions.0.{group_norm,to_q,to_k,to_v,to_out.0}.*`, `encoder.conv_norm_out.*`, `encoder.conv_out.conv.*`, `quant_conv.*`; decoder mirrored (`decoder.up_blocks.{i}.resnets.{0..2}.*`, `.upsamplers.0.conv.conv.*`, `post_quant_conv.*`). The HartsyInference loader accepts **both** `.conv1.conv.weight` and `.conv1.weight` spellings (fallback-gated pending a key dump of the actual K5 vae shard).
- **Latent scaling:** encode → `latent · 0.476986`; decode → `latent / 0.476986` first. No per-channel mean/std (unlike Wan).
- **Memory note:** dense mid-block attention over T·H·W latent tokens means full-clip single-pass decode at 512×768×121f is ~190k tokens in the mid block — diffusers handles this with tiled decode (256-px / 16-frame tiles). HartsyInference ships the untiled core first; tiling is a follow-up before large-clip GPU runs.

## I2V specifics (from `pipeline_kandinsky_i2v.py`)

1. Image → VAE encode (sample of the diagonal gaussian; HartsyInference uses the mean/mode deterministically) → `· scaling_factor` → becomes latent frame 0 (noisy stream AND cond stream), mask frame 0 = 1.
2. Scheduler steps only frames `1:` — frame 0 stays the image latent for the whole loop.
3. Post-loop **`normalize_first_frame`** "mesh artifact" fix: adaptive mean/std normalization of the first 4 latent frames against frames 4..8 (mean clamp +0.1/−0.05, std clamp +0.25/−0.10), skipped when T_lat ≤ 1 (HartsyInference also skips when T_lat ≤ 5 — not enough reference frames).

## HartsyInference mapping

| Piece | Implementation |
|---|---|
| Config | `Kandinsky5Config.VideoLite2B` / `VideoPro19B` presets + `RopeScaleFactor` (default (1,1,1) → preserves T2I) |
| Transformer | `Kandinsky5Transformer.ForwardVideo` — BCTHW in/out, T>1 patch embed, 33-ch packing validated against `VisualCond` |
| RoPE | `Kandinsky5Rope.Precompute3D(..., scaleT, scaleH, scaleW)` — defaults 1.0, T2I call sites unchanged |
| VAE | `Models/Vae/HunyuanVideoVae{Config,Decoder,Encoder}.cs` + `HunyuanVideoResnetBlock3d` (+ causal-attention mid block reusing `VaeAttention` weights layout) |
| Pipeline | `HartsyInference.Video/Pipelines/Kandinsky5VideoPipeline.cs` — T2V + I2V, precomputed embeds, CFG dual pass, FlowMatchEuler(shift 5.0), Preload/FreeWeights paired |
| Converter | `Kandinsky5CheckpointConverter.LoadVideoTransformer` / `LoadHunyuanVideoVae` |

## Uncertainties / validation-gated

1. **Native vs diffusers key naming** — the native (non-Diffusers) single-file release may use the original repo names; only the diffusers layout is implemented. Single-file repacks with `transformer.`/`model.` prefixes are handled by the existing prefix strip.
2. **HunyuanVideo VAE weights** — architecture is identical to hunyuanvideo-community/HunyuanVideo's VAE; whether kandinskylab retrained/fine-tuned the weights is unknown. Always load the `vae/` shard from the K5 repo itself.
3. **VAE conv key spelling** — `.conv1.conv.weight` (module-wrapped) is what the diffusers module structure produces; the loader fallback also accepts `.conv1.weight` pending a key dump.
4. **Dense vs NABLA equivalence** — dense SDPA is not bit-identical to the trained block-sparse mask; expect small deviations (the model was also evaluated dense upstream for ≤5 s clips). 241-frame configs warn.
5. **Pro 5 s attention flag** — whether Pro checkpoints set `attention_type="flash"` (dense) or `"nabla"` at 5 s is unverified; config-only support either way.
6. **Scheduler shift for native single-file** — diffusers checkpoints use shift 5.0; the native release reportedly uses 10.0. Shift is a pipeline constructor parameter.
7. **I2V gaussian sampling** — the reference samples the encoder posterior with the generator; HartsyInference uses the mean (mode). Deterministic and within-noise, but not bit-identical.
8. **VAE tiling** — untiled decode/encode only; large-clip GPU decode needs the diffusers-style spatial/temporal tiling follow-up.
9. **T2I unpatchify fix (2026-06-12)** — while unifying the image/video forward, the existing T2I `Unpatchify` was found to read the out-layer projection vector **C-minor** (`(py, px, c)`), but diffusers' `Kandinsky5OutLayer` does `view(..., -1, p_t, p_h, p_w)` which is **C-major** (`(c, pt, py, px)`); the shared implementation now uses C-major. (The patch-embed input really is C-minor — the two orders legitimately differ.) If T2I output was previously visually validated as correct, re-verify against the reference; this changes T2I numerics by design.

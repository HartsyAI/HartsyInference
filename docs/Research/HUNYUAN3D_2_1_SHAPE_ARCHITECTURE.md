# Hunyuan3D-2.1 shape model: source spike

Read 2026-11-10 from source and configs only. **No weights were downloaded and no model code was written**, so nothing here is
verified against a checkpoint. The status doc row is [MODEL_STATUS_3D.md](../Checklists/MODEL_STATUS_3D.md#hunyuan3d-21-shape).
The 2.0 family is in [HUNYUAN3D_ARCHITECTURE.md](HUNYUAN3D_ARCHITECTURE.md).

## Pins

| Item | Pin |
|---|---|
| Code | `Tencent-Hunyuan/Hunyuan3D-2.1` `main` at `82920d64` (2025-10-17), package `hy3dshape` |
| Weights | `tencent/Hunyuan3D-2.1` revision `0b946776`, not gated |
| DiT | `hunyuan3d-dit-v2-1/model.fp16.ckpt`, 7,366,389,768 bytes, plus `config.yaml` (2,078 B) |
| VAE | `hunyuan3d-vae-v2-1/model.fp16.ckpt`, 655,648,152 bytes (includes the encoder), plus `config.yaml` (409 B) |

**Both checkpoints are pickles (`.ckpt`); the repository has no safetensors for the shape model.** They must go through the
safe-subset pickle parser and be repacked to safetensors once; nothing may execute them. Key names are not known until a file is
inspected, which needs an approved download.

## Licence: Tencent Hunyuan 3D 2.1 Community License Agreement

Release date June 13, 2025. The text is the 2.0 agreement with the name and date changed and one added item: a request for a
licence above 1 million monthly active users must also describe plans to modify the Works or create Model Derivatives (section
4). Everything recorded for 2.0 holds: the agreement does not apply in the European Union, United Kingdom and South Korea, and use,
reproduction or display of the Works **or of Output** outside that Territory is unlicensed (5c); more than 1 million monthly active
users on the release date needs a separate licence (4); the Works and Output must not be used to improve another AI model (5b);
Tencent claims no rights in Outputs (6d); the Acceptable Use Policy applies (military use, unlabelled machine-generated public
content, high-stakes automated decisions and the rest). The full terms are summarised in
[HUNYUAN3D_ARCHITECTURE.md](HUNYUAN3D_ARCHITECTURE.md#licence-tencent-hunyuan-3d-20-community-license-agreement). Use here is personal.

## What the model is

2.1 is **not** the 2.0 Flux-style MMDiT with a bigger config. `model.target` is `HunYuanDiTPlain`
(`hy3dshape/models/denoisers/hunyuandit.py`), a U-ViT-style transformer:

| Item | Value (from `config.yaml` and source) |
|---|---|
| Latent | 4096 tokens x 64 channels (`input_size`, `in_channels`) |
| Width, depth, heads | 2048, 21 blocks, 16 heads (head dim 128) |
| Conditioning | DINOv2-large at 518 px, no registers, no SwiGLU: 1370 tokens x 1024 (`context_dim`, `text_len`); unconditional input is zeros |
| Time | `Timesteps(2048)`: sin then cos of `t` in [0, 1] (no `time_factor`), max period 10000, then `Linear(2048, 8192)`, GELU, `Linear(8192, 2048)`. The vector is **prepended as one extra token**; there is no AdaLN. `final_layer` drops it again. |
| Block | `x = skip_norm(skip_linear(cat[skip, x]))` on blocks 11 to 20 (U-skips from blocks 0 to 9); `x += self_attn(LN(x))`; `x += cross_attn(LN(x), dino)`; `x += mlp_or_moe(LN(x))`. LayerNorm eps 1e-6. |
| Attention | Q, K, V without bias, RMSNorm on Q and K per head (eps 1e-6), no RoPE, no position embedding (`use_pos_emb: false`); cross-attention K and V project 1024 to 2048. Decoupled cross-attention and attention pooling are off. |
| MoE | the last 6 blocks (15 to 20): 8 experts, top 2, plus one shared expert |
| Output | `LayerNorm`, drop the time token, `Linear(2048, 64)` |

About 3.0 billion parameters by count (self-attention 16.8 M, cross-attention 12.6 M and skip 8.4 M per block, a 33.5 M MLP in 15
blocks, nine 33.5 M GELU MLPs in each of the 6 MoE blocks). In F16 the DiT is about 6 GB, so a 12 GB card is plausible but
tight with 4096 + 1370 tokens of activations.

### The MoE block (`moe_layers.py`)

DeepSeek-style: a bias-free gate `Linear(2048, 8)`, softmax over the 8 scores, `top_k = 2` selected unsorted, **weights not
renormalised** (`norm_topk_prob = False`), each expert a diffusers `FeedForward(gelu)` (`Linear(2048, 8192)` with bias, exact GELU,
`Linear(8192, 2048)` with bias), the routed outputs summed with their gate weights, and the shared expert (same shape, run on every
token, not gated) added. Parameter names follow the modules: `moe.gate.weight`, `moe.experts.<i>.net.0.proj.*`,
`moe.experts.<i>.net.2.*`, `moe.shared_experts.net.0.proj.*` and `.net.2.*`.

### Sampler

Same as 2.0: ascending `np.linspace(0, 1, steps)` sigmas with the flow-match Euler step, timestep fed as sigma, CFG with a zero
unconditional input (`guidance_scale` 5.0 in the call, 50 steps, octree 384 default, `mc_level` 0.0, `box_v` 1.01). The DiT has no
guidance embedding (`guidance_cond_proj_dim` is null in this config).

### VAE (`hunyuan3d-vae-v2-1`)

The config differs from the 2.0 decoder only in `num_latents: 4096`, `scale_factor: 1.0039506`, and the encoder inputs
(`point_feats: 4`, `pc_size: 81920`, `pc_sharpedge_size: 0`, unused for decoding). The geo decoder is the 2.0 shape (width 1024,
16 heads, MLP ratio 4, `ln_post`, 8 Fourier octaves, no pi). `Hunyuan3DShapeVae` should run it from a `Hunyuan3DConfig` with those
numbers; unverified.

## Reuse map

| Piece | Verdict |
|---|---|
| DINOv2-large conditioner, 518 px | `Dinov2Preset.Large` exists. Shared as is. |
| ShapeVAE decoder, grid, marching cubes, `Hunyuan3DSurface` | Shared; `Hunyuan3DConfigFile` would need to read the new DiT keys. |
| Sampler, schedule, CFG | `Hunyuan3DFlowSchedule` and `GuidedEulerStep` as is; the timestep is sigma without `time_factor`. |
| DiT | **New model.** Nothing in `Hunyuan3DDit` applies: no double or single streams, no AdaLN, no joint attention, plus a time token, U-skips, cross-attention and head dim 128. |
| MoE FFN | **Does not fit.** The LLM `MoeFeedForward` and `ExpertProgram` compute `down(act(gate(x)) * up(x))` (three matrices, no bias); these experts are two-matrix GELU MLPs with biases. The router part (softmax, top-k, no renormalisation, ungated shared expert) is expressible in `RouterDescriptor`, so the routing contract can be reused while the expert body is written for this model. At 4096 tokens a gathered per-expert `Linear` is enough; the LLM device-resident decode path is irrelevant. |
| Checkpoint load | Safe pickle parse and one repack to safetensors. |

## Plan, if approved

1. Download the VAE `.ckpt` first (0.66 GB), inspect keys through the safe parser, repack, and check `Hunyuan3DShapeVae` runs it.
2. Download the DiT `.ckpt` (7.4 GB), inspect keys, repack to F16 safetensors.
3. `Hunyuan3D21Dit` (new): blocks, time token, skips, cross-attention, MLP and MoE, in F32 first; per-block parity against `hy3dshape`
   on CPU (the Python needs `timm` and `diffusers` stand-ins for `FeedForward`; neither is in the reference venv).
4. Catalog id `hunyuan3d-21`, config reader extension, GPU run, render, then status.

## Open questions and risks

- Checkpoint key layout, top-level groups and whether the DiT file holds EMA weights: unknown until inspected.
- Memory on the 12 GB card: 6 GB of F16 weights plus the MoE gather buffers; F32 on the CPU reference needs about 12 GB for the DiT alone.
- `topk(sorted=False)` ordering and near-ties can flip an expert between backends; the combine is order-independent, but parity wants a
  tolerance on the MoE outputs.
- The 2.1 repository also ships a Paint pipeline (PBR texture) that this spike does not cover.

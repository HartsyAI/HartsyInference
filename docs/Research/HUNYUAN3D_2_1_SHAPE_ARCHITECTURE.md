# Hunyuan3D-2.1 shape model: architecture, checkpoint and port

Source spike read 2026-11-10, then the checkpoint keys were read and the DiT was ported the same day (sections marked "checked"). The
status doc row is [MODEL_STATUS_3D.md](../Checklists/MODEL_STATUS_3D.md#hunyuan3d-21-shape). The 2.0 family is in
[HUNYUAN3D_ARCHITECTURE.md](HUNYUAN3D_ARCHITECTURE.md).

## Pins

| Item | Pin |
|---|---|
| Code | `Tencent-Hunyuan/Hunyuan3D-2.1` `main` at `82920d64` (2025-10-17), package `hy3dshape` |
| Weights | `tencent/Hunyuan3D-2.1` revision `0b946776`, not gated |
| DiT | `hunyuan3d-dit-v2-1/model.fp16.ckpt`, 7,366,389,768 bytes, plus `config.yaml` (2,078 B) |
| VAE | `hunyuan3d-vae-v2-1/model.fp16.ckpt`, 655,648,152 bytes (includes the encoder), plus `config.yaml` (409 B) |

**Both checkpoints are pickles (`.ckpt`); the repository has no safetensors for the shape model.** They go through the safe-subset
pickle parser (`PytorchPickleLoader`, nothing executes) and `PickleCheckpointRepacker` once. Files downloaded 2026-11-10 with sha256
checked: DiT `6b519fc7242f78e9b5f47ea4d55668fe3d944a2d27332f4ca68d29a6ff603f5e`, VAE `5cbe97f25e6e7abd4bccc80ab07524ec0c86d24118486a9ba49bb5dfb070288a`.

**Checked: the DiT file is a bundle, like 2.0.** 1601 tensors, all F16, in three groups: `model.*` (752, the DiT), `conditioner.*` (439,
DINOv2-large under `conditioner.main_image_encoder.model.`) and `vae.*` (410, the full VAE with its encoder). That accounts for the 7.37 GB
against about 6 GB for the DiT alone, so the separate VAE download is not needed to run the model. The `vae.*` decoder keys and shapes
are identical to the 2.0 bundled VAE (266 keys, no differences), so `Hunyuan3DShapeVae` loads them from a config with 4096 latents and
scale 1.0039506. The DiT key names are exactly the ones in the source (`blocks.N.attn1.to_q.weight`, `moe.experts.E.net.0.proj.*`, ...)
and the shapes match the config (for example `t_embedder.mlp.0.weight` 8192 x 2048, `attn2.to_k.weight` 2048 x 1024, 192 expert tensors).

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
| Attention | Q, K, V without bias, RMSNorm on Q and K per head (eps 1e-6), no RoPE, no position embedding (`use_pos_emb: false`); cross-attention K and V project 1024 to 2048. Decoupled cross-attention and attention pooling are off. **Head split (checked against `hy3dshape/models/denoisers/hunyuandit.py` at `82920d64`):** the module concatenates the Q, K and V projections (`cat(q, k, v)` for self-attention, `cat(k, v)` for cross-attention) and views the result as `(heads, 3 * head_dim)` (`2 *` for cross), so a head's Q, K and V are three consecutive channel groups of the fused row, not that head's own slices of each projection. The released weights were trained that way; the engine regroups the projection rows once at load (`Hunyuan3DPlainBlock.Regroup`). Without it block 0 has correlation 0.17 against the reference; with it, 1.0000. |
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

## Port (checked 2026-11-10)

`Hunyuan3DPlainDit` and `Hunyuan3DPlainBlock` implement the DiT; `Hunyuan3DConfigFile` reads the 2.1 config (including its YAML
anchors); `Hunyuan3DShapePipeline` picks the denoiser and DINOv2-large from the config. The MoE routes on the host (softmax, top 2,
unnormalised weights), gathers rows per expert with `GatherRows`, runs the GELU experts with `Linear`, and combines with
`ScatterAddRows`; the shared expert runs on every token. Every GELU here (time embedder, dense MLPs, shared and routed experts) is the exact erf form, `backend.GeluErf`, as in torch's `nn.GELU()` and diffusers' `gelu` activation; the 2.0 Flux blocks keep the tanh form. Against the Python reference on the CPU in fp32 (see the status doc for the
numbers) the DiT and the bundled VAE are compared by `Hunyuan3D21ShapeParityTests`, fed by `dump_shape21_reference.py`.

## Open questions and risks

- The file holds one set of weights (no EMA or optimizer copy was found); which training stage they are from is not stated.
- Memory on the 12 GB card: about 6 GB of F16 weights plus the MoE gather buffers (the F32 CPU path needs about 12 GB for the DiT alone). This is an estimate: the DiT has only been run on the CPU backend (one forward), not on CUDA or Vulkan at scale.
- `topk(sorted=False)` ordering and near-ties can flip an expert between backends; the combine is order-independent, but parity wants a
  tolerance on the MoE outputs.
- The 2.1 repository also ships a Paint pipeline (PBR texture) that this spike does not cover.

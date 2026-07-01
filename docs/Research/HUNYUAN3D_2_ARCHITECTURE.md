# Hunyuan3D-2 (shape) — Architecture Notes

First 3D model in HartsyInference. Single image → mesh. Two model components: an image-conditioned
**flow-match DiT** that denoises a VecSet shape latent, and a **ShapeVAE decoder** that turns the latent
into an occupancy/SDF field, which marching cubes turns into a mesh. Texture ("Paint") is **out of scope**.

> ⚠️ **Status: structural build was a WRONG-ARCHITECTURE guess — rewrite required (2026-06-30).** The real
> `tencent/Hunyuan3D-2` DiT is **Flux double/single-stream (16 double + 32 single, NO RoPE), conditioned on
> DINOv2-GIANT (1536-dim, 40 layers)**, and the ShapeVAE is a `transformer.resblocks` self-attn stack + a
> `geo_decoder` cross-attn head — NOT the PixArt-style single-stream DiT + simple cross-attn VAE the sections
> below describe. The current C# `Hunyuan3DDit`/`Hunyuan3DShapeVae`/`Hunyuan3DConfig` do not match and must be
> rewritten. **The full confirmed spec + key tables + build order live in
> [`../Checklists/E2E_3D_WORKLOG.md`](../Checklists/E2E_3D_WORKLOG.md) § Hunyuan3D-2** (extracted from the real
> weights + `hy3dgen/shapegen/models/denoisers/hunyuan3ddit.py`). The sections below are the original guess,
> kept only for history — follow the worklog, not this.

## Pipeline

```
image ─DINOv2─▶ cond tokens [B, S_img, D_img]
                       │ (linear proj → width)
noise VecSet [B, N, C] ─flow-match Euler (+CFG)─▶ shape latent [B, N, C]
                       │
ShapeVae.Decode: query grid points [P,3] ──cross-attn to latent──▶ occupancy[P] ─▶ ScalarField3D
                       │
              MarchingCubes ─▶ Mesh ─▶ GlbWriter
```

## Conditioning — DINOv2

- `facebook/dinov2` ViT (large is the Hunyuan3D-2 default). Output is the full token sequence
  (CLS + registers + patches); the DiT cross-attends to it after a linear projection to the DiT width. **[VG]**
  whether CLS/registers are included or stripped, and the exact projection.
- Implemented in `HartsyInference.Vision/Dinov2/` (`Dinov2VisionEncoder`, `Dinov2ImagePreprocessor`).

## Shape DiT (`Hunyuan3DDit`)

- Operates on **VecSet** latent: `N` tokens × `C` channels (no spatial grid; a permutation-invariant set).
- Per-block structure (PixArt/DiT-with-cross-attn family): timestep → AdaLN 6-way modulation
  `(shift/scale/gate)_{attn,mlp}`; `x += gate_attn · SelfAttn(modulate(norm1(x)))`;
  `x += CrossAttn(norm2(x), cond)`; `x += gate_mlp · MLP(modulate(norm3(x)))`. **[VG]** exact block wiring
  (single vs double stream, whether cross-attn is gated, QK-norm presence).
- Timestep embedding: sinusoidal → 2-layer MLP. **[VG]** timestep scaling convention (raw t vs t·1000).
- Rectified flow, **velocity** prediction. Scheduler: shifted flow-match Euler (reuse
  `FlowMatchEulerDiscreteScheduler` / `LancePipelineCommon`). **[VG]** flow shift value.

## ShapeVAE decoder (`Hunyuan3DShapeVae`)

- VecSet autoencoder; **decoder only** for v1 (generation doesn't need the encoder). Given a batch of 3D
  query coordinates, the decoder cross-attends those queries (after a Fourier/positional embed) to the
  latent token set, runs a few transformer/MLP layers, and projects to a scalar occupancy/SDF per query.
- We query a dense `R³` grid (default R=256, coarse→fine optional) → `ScalarField3D` → marching cubes at
  the model's iso level. **[VG]** query positional embedding (Fourier bands), latent normalization
  (scale/shift), iso level, grid bounds, and decoder depth.

## Checkpoint layout

`Hunyuan3DCheckpointConverter` (in `ModelHandler/CheckpointConverters`) splits the HF checkpoint into
`{ Dit, ShapeVae, Dinov2 }` weight dicts, detecting diffusers vs native `hy3dgen` naming and folding any
FP8 scales (follow `WanVideoCheckpointConverter` / `FluxCheckpointConverter`). **[VG]** exact key tables.

## Validation plan

`tests/python-reference/dump_hunyuan3d_full_forward.py` + `diff_hunyuan3d_layers.py` (mirror existing dump/
diff scripts); walk `Hunyuan3DDebugDump` hooks to the first layer with `avg_err > 1e-3`, fix, repeat. Then
an env-gated end-to-end test on the real checkpoint (RTX 3060) asserting a watertight mesh + `.glb`.

# TripoSR — Architecture Notes

Second 3D model in HartsyInference. Single image → mesh, **feed-forward / deterministic** (no diffusion
loop) — the cheapest model in the project to validate, and it exercises the triplane + NeRF-MLP foundation
paths that Hunyuan3D doesn't. TripoSR (Stability AI / Tripo, MIT) is an LRM.

> ⚠️ **Status: structural build, numerics validation-pending.** Green end-to-end on synthetic weights
> (shapes + finiteness, image→triplane→density→mesh). `[VG]` items below are reconciled against the
> reference (`stabilityai/TripoSR`) via the layer-diff harness before output is trustworthy.

## Pipeline

```
image ─DINO ViT─▶ image tokens [1, S, 768]
                       │ (linear proj → width)
learned triplane tokens [1, 3·R², W] ─(self-attn + cross-attn to image + MLP)×depth─▶ triplane [3, C, R, R]
                       │
NeRF MLP: query point ─project to 3 planes─bilinear sample─concat[3C]─MLP─▶ [density, rgb]
                       │
density grid ─Marching Cubes (iso = density threshold)─▶ Mesh (+ per-vertex color) ─▶ .glb
```

## Components

- **Image tokenizer** — a DINO ViT (TripoSR uses **DINOv1** `vit-base`, no LayerScale). Reuses
  `HartsyInference.Vision/Dinov2/Dinov2VisionEncoder` — LayerScale was made **optional** (absent → identity)
  precisely so this DINOv1 case works. `[VG]` exact preset (patch 16, image size) and whether CLS is used.
- **Triplane transformer** (`TripoSrTransformer`) — learned triplane position tokens (`3·R²` of them) that
  self-attend and cross-attend to the image tokens through `depth` blocks (pre-norm, no-affine LN, GELU MLP),
  then project to `C` channels and reshape to a `[3, C, R, R]` `Triplane`. Feed-forward; no timestep. Reuses
  `Hunyuan3DAttention` + `DiTUtils.LayerNormNoAffine`. `[VG]` block wiring, norm affine-ness, token layout,
  triplane upsampling (the reference may upsample the triplane after the transformer).
- **NeRF decoder** (`TriplaneNerfDecoder`) — projects a 3D point onto the three planes (XY/XZ/YZ), bilinearly
  samples each (`GridSampler.BilinearPlane`), concatenates `3C` features, and runs an MLP → `[density, rgb]`
  (rgb via sigmoid). `[VG]` plane→axis mapping, density activation (exp/ReLU/raw), MLP depth, density
  threshold, bbox.

## Extraction

Density field over an `R³` grid → marching cubes. Surface is **density > threshold**; since `MarchingCubes`
treats "inside" as `value < iso`, the pipeline extracts on the **negated** field at `-threshold` (keeps
outward normals). Per-vertex colors come from `DecodeColors` at the mesh vertices.

## Checkpoint layout

`TripoSrCheckpointConverter` splits into `{ Dino, Transformer, Decoder }` by coarse prefix. `[VG]` exact
prefix set + per-key rename tables (finalized during the diff pass).

## Validation plan

Same harness shape as Hunyuan3D: a `dump_triposr_full_forward.py` (deterministic forward, dumps image
tokens, triplane, density at fixed probe points, per-block outputs) + a `diff_triposr_layers.py`. Because
TripoSR is deterministic (no sampling), the diff is a single clean comparison — no scheduler/seed alignment
needed, which is why it's the easiest model to drive to ✅.

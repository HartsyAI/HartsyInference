# Phase 11 — 3D Asset Generation (`HartsyInference.ThreeD`)

Image/text → 3D mesh (and, later, Gaussian splats). The one previously-empty modality. New
`HartsyInference.ThreeD` package (deps: Diffusion + Vision). Status legend matches the rest of the project:
✅ verified vs reference · 🔧 built, numerics validation-pending · ❌ not started.

Build doc / research: [HUNYUAN3D_2_ARCHITECTURE.md](../Research/HUNYUAN3D_2_ARCHITECTURE.md),
[TRIPOSR_ARCHITECTURE.md](../Research/TRIPOSR_ARCHITECTURE.md).

## 1. Reusable foundation (representation-agnostic) — **DONE + CPU-tested (2026-06-14)**

- [x] Geometry types: `Geometry/Mesh.cs`, `ScalarField3D.cs`, `PointCloud.cs`, `GaussianSplatCloud.cs`, `Triplane.cs`
- [x] Geometry ops: `Geometry/Ops/MarchingCubes.cs` (canonical Bourke tables, watertight + outward normals), `MeshOps.cs` (normals), `GridSampler.cs` (trilinear + triplane bilinear), `SurfaceSampler.cs` (FPS)
- [x] Exporters: `Io/GlbWriter.cs` (binary glTF 2.0, primary artifact), `ObjWriter.cs`, `PlyWriter.cs` (mesh + 3DGS layout)
- [x] Pipeline scaffolding: `Pipelines/ThreeDPipelineBase.cs`, `Requests/ImageTo3DRequest.cs`, `ThreeDResult.cs`
- [x] DINOv2 conditioning encoder in **Vision** (`Vision/Dinov2/Dinov2VisionEncoder.cs` + preset + preprocessor); LayerScale **optional** so it also serves DINOv1 (TripoSR)
- [x] CPU unit tests: marching-cubes watertightness/normals on a sphere SDF, GLB/OBJ/PLY round-trip, trilinear sampling

## 2. Hunyuan3D-2 (image → mesh) — 🔧 structural build, numerics validation-pending (2026-06-14)

Flow-match DiT over a VecSet latent + ShapeVAE occupancy decode → marching cubes. Texture (Paint) out of scope.

- [x] `Models/Hunyuan3D/Hunyuan3DConfig.cs`, `Hunyuan3DDit.cs` (VecSet DiT, AdaLN + self/cross-attn to DINOv2, velocity), `Hunyuan3DShapeVae.cs` (decoder-only: query points → cross-attn latent → chunked occupancy grid), `Hunyuan3DAttention.cs`, `Hunyuan3DDebugDump.cs`
- [x] `Pipelines/Hunyuan3DShapePipeline.cs` (+ `LoadFromPath`); reuses `LancePipelineCommon`, `DiTUtils`, `SeedGenerator`, `MarchingCubes`
- [x] `ModelHandler/CheckpointConverters/Hunyuan3DCheckpointConverter.cs`
- [x] CPU structural tests (DiT finite velocity, ShapeVAE finite field, pipeline image→mesh end-to-end)
- [ ] **[VG]** numeric pass: download `tencent/Hunyuan3D-2`, fill `tests/python-reference/dump_hunyuan3d_full_forward.py` TODO[VG], run `diff_hunyuan3d_layers.py`, drive to ✅ (DiT block wiring, timestep scaling, VAE Fourier/iso/bounds, converter keys, config dims)

## 3. TripoSR (image → mesh, feed-forward) — 🔧 structural build, numerics validation-pending (2026-06-14)

Deterministic LRM → triplane → NeRF MLP → marching cubes (no diffusion → easiest to validate).

- [x] `Models/TripoSr/TripoSrConfig.cs`, `TripoSrTransformer.cs` (learned triplane tokens self+cross-attn to DINO image tokens → `Triplane`), `TriplaneNerfDecoder.cs` (3-plane sample → MLP → density+rgb)
- [x] `Pipelines/TripoSrPipeline.cs` (+ `LoadFromPath`); density field → marching cubes → mesh + per-vertex colors
- [x] `ModelHandler/CheckpointConverters/TripoSrCheckpointConverter.cs`
- [x] CPU structural tests (transformer triplane shape, NeRF density field, pipeline end-to-end)
- [ ] **[VG]** numeric pass vs `stabilityai/TripoSR` (single clean diff — no scheduler/seed alignment); DINO preset, plane→axis mapping, density activation/threshold, converter keys

## 4. Sample + wiring — DONE

- [x] `samples/HartsyInference.ThreeD.Cli` (`hartsyinference-3d`, `--type hunyuan3d|triposr`) → image → `.glb`
- [x] Package registered in `HartsyInference.Meta`, both solutions; full solution builds clean (both TFMs, warnings-as-errors)

## 5. Deferred (next 3D models)

- [ ] **TRELLIS** (image → Gaussian splat + mesh) — needs sparse 3D conv/attention (no backend op yet) + flexicubes + splat rendering. The `GaussianSplatCloud` type + PLY splat export are already in place.
- [ ] Texture/PBR (Hunyuan3D Paint — multiview diffusion + UV bake)
- [ ] Splat **rendering** (rasterizer) for previews

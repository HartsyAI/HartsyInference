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

## 2. Hunyuan3D-2 (image → mesh) — 🔧 REBUILD REQUIRED; arch reverse-engineered (2026-06-30)

The existing scaffold (`Hunyuan3DDit` as a simple AdaLN DiT with `in_proj`/`blocks.{i}.ada_mod`/`self_attn`/
`cross_attn` keys) is the **wrong architecture** — none of those keys exist in the real checkpoint. The real
`tencent/Hunyuan3D-2` shape model (single `hunyuan3d-dit-v2-0/model.safetensors`, also bundles the conditioner)
is a **Flux-lineage MMDiT + VecSet ShapeVAE**, flow-matched (FlowMatchEulerDiscreteScheduler). Reference source
vendored at `/tmp/hy3d/hy3dgen/shapegen`; full arch in [[hunyuan3d2-is-flux-lineage]] memory.

**Conditioner — ✅ verified (2026-06-30).** DINOv2-giant (1536/40L/24H) with **SwiGLU FFN**, bundled as
`conditioner.main_image_encoder.model.*`; native 518px (1370 tokens), no registers, no pos-interp. Added a
SwiGLU path + `Dinov2Preset.Giant` to the shared `Dinov2VisionEncoder`; parity vs HF `Dinov2Model` (oracle
`tests/python-reference/hunyuan3d_reference/dump_dino_giant_reference.py`, test `Hunyuan3DDinoParityTests`).

**DiT (`model.*`) — REBUILD.** Original-Flux key layout (NOT diffusers), **no RoPE** (`pe=None`):
- `latent_in` Linear(64→1024); `cond_in` Linear(1536→1024); `time_in` = MLPEmbedder(256→1024→1024) on
  `timestep_embedding(t, 256, time_factor=1000)` (×1000, freqs `exp(-ln(10000)·arange(128)/128)`, **cos then sin**).
- `double_blocks.{0..15}` (Flux DoubleStreamBlock): `img_mod.lin`/`txt_mod.lin` [6144,1024] (6-way: shift,scale,gate ×2),
  no-affine LN eps 1e-6, `{img,txt}_attn.qkv` [3072,1024] (bias) → split q,k,v → `norm.{query,key}_norm.scale` [64]
  RMSNorm eps 1e-6 → **joint attn over cat(txt,img)** → `proj`; `{img,txt}_mlp.0/2` GELU-tanh; gated residuals.
- `single_blocks.{0..31}` (Flux SingleStreamBlock): `linear1` [7168,1024] → split [qkv 3072 | mlp 4096], QKNorm,
  `linear2` [1024,5120] on cat(attn, gelu-tanh(mlp)); `modulation.lin` [3072,1024] (3-way) gated residual. Runs on
  cat(cond, latent); slice latent back before the final layer.
- `final_layer`: `adaLN_modulation.1` [2048,1024] (SiLU→2-way shift,scale), no-affine LN, `linear` [64,1024].
- The engine's `FluxSingleStreamBlock`/`FluxDoubleStreamBlock` use **diffusers keys + always apply RoPE** → write
  Hunyuan3D-specific blocks reusing `QkNorm` + SDPA + `AdaLNModulation`, no rope. hidden 1024, 16 heads, mlp_ratio 4.

**VecSet ShapeVAE (`vae.*`) — REBUILD.** `post_kl` Linear(64→1024) → `vae.transformer.resblocks.{0..15}`
(CLIP-style: ln_1 → self-attn [`c_qkv` [3072,1024] packed, `attention.{q,k}_norm.scale` [64] RMSNorm, `c_proj`]
→ ln_2 → mlp [`c_fc` 1024→4096, GELU, `c_proj`]) → `geo_decoder`: Fourier point embed (num_freqs 8, include_pi
false → 3+3·2·8=**51**) → `query_proj` [1024,51] → `cross_attn_decoder` (`c_q`, packed `c_kv` [2048,1024],
q/k_norm, ln_1/2/3, mlp) attending to the 3072 latents → `ln_post` → `output_proj` [1,1024] → occupancy →
marching cubes. qk_norm true; scale_factor 0.9990943042622529 (latent ×= 1/scale_factor before decode).

**Pipeline.** FlowMatchEulerDiscreteScheduler (`prev = sample + (σ_next−σ)·v`, sigmas shifted; init latent = noise
[1,3072,64]); 2-way CFG (guidance_scale ~5); the conditioner uncond is **zeros** `[B,1370,1536]`. Validate the DiT
+ VAE on **CUDA** (CPU forward is impractically slow at this size).

- [x] Conditioner (DINOv2-giant SwiGLU) verified.
- [ ] **[VG]** Rebuild DiT (Flux double/single, no rope) → parity vs hy3dgen `Hunyuan3DDiT` (CUDA).
- [ ] **[VG]** Rebuild ShapeVAE (post_kl + resblocks + geo_decoder + Fourier) → parity vs hy3dgen `ShapeVAE` (CUDA).
- [ ] **[VG]** Wire pipeline (FlowMatchEuler + CFG + converter) → e2e mesh on CUDA.

## 3. TripoSR (image → mesh, feed-forward) — ✅ verified e2e on real weights (2026-06-30)

Deterministic LRM → triplane → NeRF MLP → marching cubes (no diffusion → easiest to validate).

- [x] `Models/TripoSr/TripoSrConfig.cs`, `TripoSrTransformer.cs` (learned triplane tokens self+cross-attn to DINO image tokens → `Triplane`), `TriplaneNerfDecoder.cs` (3-plane sample → MLP → density+rgb)
- [x] `Pipelines/TripoSrPipeline.cs` (+ `LoadFromPath`); density field → marching cubes → mesh + per-vertex colors
- [x] `ModelHandler/CheckpointConverters/TripoSrCheckpointConverter.cs`
- [x] CPU structural tests (transformer triplane shape, NeRF density field, pipeline end-to-end)
- [x] **[VG] DONE (2026-06-30)** numeric pass vs `stabilityai/TripoSR` — rewrote the model to the real arch (the scaffold guessed wrong on every component): built `DinoViTEncoder` (HF ViT + bicubic pos-interp + exact-erf GELU), rewrote `TripoSrTransformer` (diffusers `Transformer1D` + ConvTranspose2d upsampler) and `TriplaneNerfDecoder` (grid_sample align_corners=False, exp density, iso 25). DINO tokens corr ~1.0, backbone scene_codes corr 1.0 (std 409.33), decoder density/color to 1.6e-2/5e-6; coherent 84k-tri chair `.glb` on the 3060. Fixed 2 bugs (DINO pos-embed `+0.1`, CUDA activation-cache reshape identity — see PARITY_VERIFICATION §Bugs). Tests: `TripoSrParityTests` (CPU + `PARITY_BACKEND=cuda`), `TripoSrGenerationTests` (GPU e2e), `CudaOpBisectTests` (CPU-vs-CUDA op regression). **Follow-up:** port the rembg + resize_foreground + gray-0.5 composite preprocessing to C# (currently done in Python) for raw-RGBA input.

## 4. Sample + wiring — DONE

- [x] `samples/HartsyInference.ThreeD.Cli` (`hartsyinference-3d`, `--type hunyuan3d|triposr`) → image → `.glb`
- [x] Package registered in `HartsyInference.Meta`, both solutions; full solution builds clean (both TFMs, warnings-as-errors)

## 5. Deferred (next 3D models)

- [ ] **TRELLIS** (image → Gaussian splat + mesh) — needs sparse 3D conv/attention (no backend op yet) + flexicubes + splat rendering. The `GaussianSplatCloud` type + PLY splat export are already in place.
- [ ] Texture/PBR (Hunyuan3D Paint — multiview diffusion + UV bake)
- [ ] Splat **rendering** (rasterizer) for previews

## 6. Foreground preprocessing — pure-C# background removal (**REQUIRED for raw-image input; no Python in the app**)

Both TripoSR and Hunyuan3D condition on a **foreground-isolated** image (object on a neutral gray-0.5 background),
which the upstream repos produce with Python `rembg` (U²-Net ONNX) + `resize_foreground` + composite. Our app must
never shell out to Python, so this needs a native tool. During TripoSR e2e validation the compositing was done in
Python (`/tmp/prep_triposr.py`) — that is a **stopgap, not shippable**.

- [ ] **Salient-object segmentation model in `HartsyInference.Vision`** → per-pixel alpha mask (U²-Net / ISNet /
  BiRefNet; U²-Net is the rembg default and smallest). This is a normal model-build (weights + converter + forward
  + parity), reusable beyond 3D.
- [ ] **`ForegroundComposite` helper in `HartsyInference.ThreeD`** (pure array ops, no model): alpha-bbox crop →
  pad to square → resize to `foreground_ratio` (0.85) → composite `rgb·α + (1−α)·0.5`. Mirrors TripoSR
  `resize_foreground` + `run.py`.
- [ ] Wire into `TripoSrPipeline.Generate` / `Hunyuan3DShapePipeline` + the CLI (flag: raw vs pre-composited input).
  Code TODOs are marked `TODO(3D/no-python)` in `TripoSrPipeline.cs` and `samples/HartsyInference.ThreeD.Cli/Program.cs`.

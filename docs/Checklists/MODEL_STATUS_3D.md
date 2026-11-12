# 3D Models — status

Concise status for 3D-generation (image → mesh) models. Open work is in the
[Remaining work](#remaining-work) section below; bring-up debugging notes live in
[TROUBLESHOOTING.md](TROUBLESHOOTING.md). Parity evidence lives in
[PARITY_VERIFICATION.md](PARITY_VERIFICATION.md). Legend: [MODEL_STATUS.md](MODEL_STATUS.md).

## Verified end-to-end (✅)

| Model | Notes |
|---|---|
| **TripoSR** | Real-weight image → mesh, verified on **both CPU F32 and the RTX 3060** vs the upstream `tsr` reference (`stabilityai/TripoSR`, 2026-06-30). ([details](#triposr)) |

Tests: `CudaOpBisectTests`. (The TripoSR parity and generation tests were removed in the 2026-08-06
suite cleanup — the parity run above is recorded here, and a broken mesh is visible in the CLI.)
Detail in the retired 3D build plan (git history).

## Validation-pending / known defective (🔧)

| Model | Notes |
|---|---|
| **HairDeck** (image → hair cards, per-card colours, spring chains; static or rigged glTF) | CUDA (RTX 3060) generations of demo `ex01` (seed 0), static and rigged, with unit parity of the spring simulation against the reference's Python. Only `ex01` has been run; `ex02`–`ex12` are not downloaded. VRM packaging is not implemented. ([details](#hairdeck)) |
| **Hunyuan3D-2** | Image → mesh (`tencent/Hunyuan3D-2`). The melting-wax defect is root-caused and fixed; one image and 20 steps are checked, so it is not yet re-promoted. ([details](#hunyuan3d-2)) |
| **Hunyuan3D-2.1 shape** | 🔧 Ported 2026-11-11: new C# DiT with MoE, checked against the Python (final velocity included) and rendered on one image. No catalog id yet. ([details](#hunyuan3d-21-shape)) |
| **Hunyuan3D Paint** (2.0 albedo, 2.1 PBR texturing of a mesh; `hartsy 3d <mesh> --image <png> -m hunyuan3d-paint\|hunyuan3d-paint21`) | 🔧 Paint 2.0 (albedo) has run end to end on real weights on the RTX 3060 (a recognisable bust texture on a test mesh); 2.1 and the delight stage have no weights yet, and nothing is compared with the Python reference. ([details](#hunyuan3d-paint)) |
| **Hunyuan3D-2 variants** (Fast, Turbo + FlashVDM, mini, 2mv) | Each build matches the Python reference stage by stage and renders a clean mesh on one image; one subject, one seed, 2mv with three views, no cuDNN. Not ✅. ([details](#hunyuan3d-2-variants)) |
| **CoLa3D decomposer (q20)** | Implemented; parity pending [#694](https://github.com/HartsyAI/HartsyInference/pull/694). Not verified against the Python reference. |
| **UniRig** (mesh → skeleton + skin weights → rigged GLB) | Skeleton, skinning, rigged-GLB export and `hartsy 3d -m unirig` implemented in `ThreeD/Models/UniRig`; both real checkpoints load with exact key maps and produce plausible rigs (CPU F32; skeleton also on CUDA). No Python reference, so nothing is parity-checked. ([details](#unirig)) |
| **Pixal3D** (image → shape-only mesh; `hartsy 3d -m pixal3d`) | 🔧 Implemented 2026-11-11, shape-only, **runs end to end on real weights (RTX 3060, 12 GB); parity with the Python pipeline pending**. Verified: the three projection flows' headers (700 F32 tensors each, the structure flow plus its C64 `rope_phases`, which equals the rebuilt RoPE table to 4.5e-7); the camera projection and the bilinear sampler against upstream's own `ProjGrid` / `sample_features` code under torch; the NAF encoder, RoPE and upsampling against upstream's NAF modules on the real weights (with `natten` replaced by a brute-force transcription of its definition); the flow's projection branch against a scalar transcription on tiny configs; pipeline orchestration over stubs. Real-weight runs of the 1024 cascade on a mug and on an example bust render as the input objects (about 8.5 and 6.4 minutes, peak 9.3 GB). Not verified: any stage's agreement with the Python pipeline, 1536, texture, multi-view, MoGe (the field of view is `--fov`). `naf_release.pth` must be placed in `3D/pixal3d` by hand. ([details](#pixal3d), [research](../Research/PIXAL3D_ARCHITECTURE.md)) |
| **TRELLIS.2** (image → mesh, optional PBR texture; `hartsy 3d -m trellis2`) | Implemented, **parity pending**: no Python reference dumps exist, so no stage is checked against upstream. The texture path (texture flow, guided texture decoder, bake to a PBR GLB) is implemented, unit-tested on tiny synthetic networks and has run on real weights (512 in 140 s warm, 1024 cascade in 564 s on the 3060; the 512 and 1024 busts render as a recognisable cream marble bust, and the UV atlas fragments), with nothing compared to Python. Verified: all three shape flow and the three texture checkpoints' safetensors headers (640 BF16 tensors per flow), the structure, shape and texture decoders' headers match the key maps; the block, RoPE, sampler and cascade logic against hand-computed values and a scalar transcription of the upstream forward (tiny configs, CPU); a real-weight CUDA run produced a finite GLB (see [details](#trellis2)). Not verified: any numerical agreement with upstream, mesh quality against the reference, the BiRefNet stand-in for RMBG-2.0, 1536. ([research](../Research/TRELLIS2_ARCHITECTURE.md)) |

Tests: `Hunyuan3DShapeParityTests` (in `tests/HartsyInference.ThreeD.Tests/Parity/`: DiT velocity at the schedule timestep
and ShapeVAE occupancy against `tests/python-reference/hunyuan3d_reference/dump_shape_reference.py`), plus the
`Hunyuan3DFlowScheduleTests` and `Hunyuan3DSurfaceTests` unit tests. The earlier DINO, VAE, DiT-parity and generation tests
are gone (2026-08-06 suite cleanup); the generation test only asserted `TriangleCount > 100`, which the defective mesh
passed. Detail in the retired 3D build plan (git history).

| Model | Notes |
|---|---|
| **TRELLIS** (image → Gaussian splat) | **Wired into the CLI 2026-07-21** (`TrellisImageTo3DPipeline`, `hartsy 3d -m trellis`) — DINOv2-with-registers-large conditioner → sparse-structure flow → active voxels → SLAT flow → GS decoder → `TrellisGaussianRepresentation` → binary-PLY splat. ([details](#trellis)) |

## Deferred / not started (❌)

| Model | Notes |
|---|---|
| **TRELLIS mesh/RF decoders** (flexicubes mesh, radiance field) | Not ported — TRELLIS is GS-splat-only today (see above). |
| **SAM 3D Objects** (image + mask → Gaussian splat or mesh, with pose) | Building blocks implemented, **no end-to-end path and nothing compared with the Python reference** (source read at `facebookresearch/sam-3d-objects@f91db411`). Done on tiny synthetic networks: flow sampler, pose decoder, prune/downsample, pointmap normaliser, mask crop, checkpoint reader (`.ckpt` and `.safetensors`), embedder fuser, structure flow, point-patch embedder. The weights (ungated mirrors, SAM License binds on use, `SAM3D_LICENSE` kept beside them) were read: key layouts, shapes and dtypes match what those modules and the existing TRELLIS flow and decoders load (see the dossier), the numbers are unverified. The third-party safetensors conversion (`apozz/sam-3d-objects-safetensors`) is byte-identical to Meta's `.ckpt` for the four generator and decoder files and for both DINO towers; the mesh decoder and MoGe files have no Meta file here to compare. Still missing: DINO wiring, pipeline, mesh decoder, catalog id. Layout optimisation is dropped, the feed-forward pose is kept. ([research](../Research/SAM3D_OBJECTS_ARCHITECTURE.md)) |

## Foundation

The `HartsyInference.ThreeD` package provides the reusable mesh / splat / triplane foundation: marching
cubes plus glTF / OBJ / PLY export, shared by all three built models.

## Remaining work

Distilled from the retired PHASE_11_THREED / TRELLIS_BUILD_PLAN / THREED_GENPERF_PLAN plans.
See [ROADMAP.md](ROADMAP.md) for cross-cutting infra (multi-GPU, kernel perf, quant, serving).

### Correctness
- [x] Hunyuan3D-2 mesh "melting-wax" surface defect: bisected and fixed (see [details](#hunyuan3d-2)).
- [ ] Hunyuan3D-2 before ✅: compare the C# conditioner output with the reference on this input, run the default 50 steps at grid 256, add a second subject (a chair), and port the upstream image preprocessing (below).
- [ ] Hunyuan3D-2 variants before ✅: more subjects, the default step counts, a four-view 2mv run, the unchecked builds (mini-Fast, 2mv-Fast, 2mv-Turbo), and a `MeshRequest.Views` proposal so 2mv is reachable from the CLI and API.
- [ ] Hunyuan3D-2 raw RGBA input: upstream recentres the foreground with a 15% border, composites it on **white** and resizes to 512 (`ImageProcessorV2`); the engine takes the RGB bytes as they are, so the caller must prepare that image. The "foreground-on-gray" hint in `MeshService` is the TripoSR contract and is wrong for Hunyuan3D.

### HairDeck
- [ ] Run demo `ex02`–`ex12` (downloads need approval; ~5.9 MB total).
- [ ] VRM packaging: decide VRM 1.0 vs 0.x and supply a VRM body asset.
- [ ] cuDNN 9 for CUDA 13 (fast convolution path); record per-step time and peak VRAM.

### TRELLIS
- [ ] Sparse-3D conv / attn backend op.
- [ ] Flexicubes + splat rendering.
- [ ] Mesh (flexicubes) + radiance-field decoders (only the GS decoder is done).
- [ ] Raw-image preprocessing for a full in-engine CLI path.
- [ ] Phase-F perf (stage-1, F16, CUDA graphs).

### TRELLIS.2
- [ ] Python reference dumps and per-stage parity (needs the GPU server or a CUDA reference env), then promote the status row.
- [ ] Texture flow, texture decoder and the GLB bake (a separate PR); hole filling.
- [ ] Cache the cross-attention keys and values per block, keep the flows resident or in BF16, and tile attention for the 1024/1536 high-resolution step.

### Other models / features
- [ ] Hunyuan3D Paint: per-stage parity against the Python reference (2.0 runs on real weights and renders), the 2.1 weights and path, the delight stage, a view-count field on `MeshRequest`. Hunyuan3D Part and Omni follow as their own spikes. ([details](#hunyuan3d-paint))
- [ ] Splat rasterizer.
- [ ] `ForegroundComposite` helper + CLI wiring (currently `TODO(3D/no-python)`).

## Details

Verification evidence, bugs found, and caveats for the rows above. Moved out of the status
tables on 2026-08-06 so the tables stay scannable — no content was dropped.

### TripoSR

Real-weight image → mesh, verified on **both CPU F32 and the RTX 3060** vs the upstream `tsr` reference (`stabilityai/TripoSR`, 2026-06-30). Every stage matches: DINO ViT-B/16 image tokens maxAbs **8.5e-6** (corr ~1.0), Transformer1D backbone scene_codes **corr 1.00000000** (std 409.33 vs 409.33; CPU maxAbs 3.4e-3), NeRF decoder density/color to 1.6e-2 / 5e-6, 64³ density grid corr **1.0** — and the density field (1.2% > iso 25) meshes into a coherent 84k-tri chair `.glb`. The scaffold was a wrong-architecture guess and was rewritten; two bugs fixed (see [PARITY_VERIFICATION.md §Bugs](PARITY_VERIFICATION.md)): DINO pos-embed interp `+0.1` fudge, and a CUDA activation-cache reshape-identity bug in the backbone. **Caveat:** background removal (rembg + resize_foreground + gray-0.5 composite) is still done in Python — the C# pipeline expects an already-composited RGB input; port it for raw-RGBA CLI use. **Re-confirmed 2026-07-21** by rendering the `hartsy 3d` CLI's own GLB output from 4 angles and visually inspecting it (not just triangle count) — clean mesh, all chair parts present and correctly composed.

### Hunyuan3D-2

Image → mesh on GPU (`tencent/Hunyuan3D-2`, weights revision `9cd649ba`, `hunyuan3d-dit-v2-0/model.fp16.safetensors`). The Flux-lineage MMDiT (16 double + 32 single blocks, QK-RMSNorm, no RoPE), the DINOv2-giant conditioner (SwiGLU) and the VecSet ShapeVAE (`post_kl`, 16 self-attention blocks, cross-attention `geo_decoder` with 51-dim Fourier queries) match upstream `hy3dgen`. Architecture, pins and licence terms: [HUNYUAN3D_ARCHITECTURE.md](../Research/HUNYUAN3D_ARCHITECTURE.md).

**The melting-wax defect (2026-11-10).** Two pipeline-level bugs, neither visible in the per-component parity that had been recorded:

1. **Timestep.** The sampler fed the DiT `sigma·1000` while `Hunyuan3DDit.Forward` takes `sigma` and applies `time_factor` (1000) itself, so the DiT saw a time 1000× too large and a different embedding at every step. Against the Python dump at the same latent and conditioning, the DiT velocity correlated **0.73** (step 10 of 20) and **0.53** (step 5) with the sampler's value, and **1.00000000** (relative L2 4.6e-5 and 1.3e-6) with the correct one. The schedule also started at sigma 0.001 where upstream starts at 0 (`np.linspace(0, 1, steps)`).
2. **Winding.** The decoder's logits are positive inside, and `MarchingCubes` treats low values as inside, so every triangle faced inward (signed volume negative) and the lighting looked wrong.

Ruled out against the reference: ShapeVAE decode (occupancy at 4000 grid points, corr 0.99999999, relative L2 7.2e-5), Fourier query encoding (include_pi is false in the config; layout matches), grid coordinates, iso level (0.0), CFG formula, unconditional tokens (zeros), and the VAE scale factor. Upstream applies no floater removal inside the pipeline (`FloaterRemover` and `FaceReducer` are separate post-processors).

| Same image, seed 0, 20 steps, grid 128 | Triangles | Components | Watertight | Signed volume |
|---|---|---|---|---|
| Before (origin/main `da9751ca3`) | 227,816 | 101 | no (10 boundary edges) | −0.614 |
| After (this change) | 77,328 | 1 | yes | +0.978 |
| Python `hy3dgen`, fp32 CPU, other noise | 81,024 | 1 | yes | +0.980 |

Rendered from four angles with the three.js viewer, the old mesh is a melted blob and the new one is the upstream demo mascot with a smooth body and a legible "HY3D" sign, matching the Python mesh. Input: `assets/demo.png` from the Hunyuan3D-2 repository, recentred and composited on white as upstream does, 512×512.

**Still open before ✅.** One subject only. The C# conditioner output was not re-compared with the reference in this pass (it was corr 1.0 earlier). The default 50 steps at grid 256 was not re-run. The GPU runs had no cuDNN (the only copy on the machine is built for CUDA 12 and is rejected under the CUDA 13 driver), so they used the custom flash kernel. Raw RGBA preprocessing is not ported (see Remaining work). Vertex positions follow the physical grid; upstream divides by `res+1` instead of `res`, which shrinks its mesh by 0.4% at 256. Historical perf data (not a correctness claim): gen-perf Rounds 1–8 took 71.3 → 9.2 s (30 steps, grid 128) with fused kernels, cuDNN fused SDPA, DiT CUDA graph and F16 activations; the DiT and ShapeVAE blocks are GPU-resident. Earlier bugs: timestep `max_period=time_factor=1000`; CUDA activation-cache reshape identity; async-race/perf (see [PARITY_VERIFICATION.md §Bugs](PARITY_VERIFICATION.md)).

### Hunyuan3D-2 variants

Added 2026-11-10 on top of the fixed base build. Each build is its own catalog id with its own download set under `Models/3D/<id>/` (`hunyuan3d`, `hunyuan3d-fast`, `hunyuan3d-turbo`, `hunyuan3d-mini`; `hunyuan3d-mv` downloads but has no CLI path). The checkpoint's `config.yaml` sets the architecture, so the id only picks files. Why ids and not a resolver change: [HUNYUAN3D_ARCHITECTURE.md](../Research/HUNYUAN3D_ARCHITECTURE.md#variant-selection). Pins, sizes and what differs per build are in the same file.

Evidence against the upstream Python (`hy3dgen` `f8db6309`, fp32 CPU, `assets/demo.png` on white, seed 0; the 2mv row uses `assets/example_mv_images/1` front, left and back). Metrics are correlation and relative L2:

| Build | DiT velocity at a mid step | ShapeVAE occupancy | Mesh through the CLI (CUDA, RTX 3060) |
|---|---|---|---|
| Fast | 1.00000000, 3.0e-5 (step 5 of 10) | 0.99999999, 6.6e-5 | 10 steps, grid 128: 85,340 triangles, 2 components (one 132-triangle floater), watertight |
| Turbo + FlashVDM | 1.00000000, 3.9e-5 (step 2 of 5, consistency sigma 0.3994) | 1.00000000, 3.8e-5 | 5 steps, grid 256 (FlashVDM grid 253): 284,827 triangles, 1 component, 3 boundary edges |
| mini | 1.00000000, 3.6e-6 (step 10 of 20) | 1.00000000, 5.4e-5 | 20 steps, grid 128: 74,692 triangles, 1 component, watertight |
| 2mv | 1.00000000, 6.1e-5 (step 4 of 8) | 1.00000000, 4.6e-5 | 8 steps, grid 128: 68,196 triangles, 3 components (largest 68,156), watertight; Python 68,886 and 4 |

FlashVDM against `FlashVDMVolumeDecoding` on the same latents (127³ grid): 194,648 points evaluated by both, 8 only by the reference and 4 only here (top-k ties), correlation 0.99999979, relative L2 6.4e-4. The conditioner: single-image tokens on the GPU against the Python tokens, correlation 0.9999926 (F16 activations); 2mv tokens against upstream's `DinoImageEncoderMV`, 0.9999957 overall and above 0.99999 per view. The unit tests pin the consistency sigma table and the view embedding to the upstream formulas.

Renders from four angles: Fast, mini and Turbo give the demo mascot with its sign (Turbo's lettering is softer, as a 5-step distilled model would be); 2mv gives a coherent character from three views.

**Not verified.** One subject and one seed per build, except base. Default steps and grid not run for the variants (Fast 10, Turbo 5 and the 50-step standard builds are upstream gradio and library defaults). 2mv was run with three of four views, no real right view, and only at 8 steps. mini-Fast, mini-Turbo, 2mv-Fast and 2mv-Turbo were not downloaded: mini-Turbo is refused (DINO at 1022 px needs position-embedding interpolation), the others should load from their configs but are untested. FlashVDM was checked only with the turbo VAE and Turbo; its `merge` top-k mode is not implemented, nor is the `dmc` extractor. Turbo's mesh has 3 boundary edges where FlashVDM left a band edge unevaluated; upstream's handling of those cells was not compared. No GPU parity runs beyond the conditioner (the numeric checks are CPU fp32), and no cuDNN. Raw RGBA preprocessing is still not ported (3a), and 2mv has no engine path because `MeshRequest` carries one image: a `Views` list would be a public contract change to agree first.

### Hunyuan3D-2.1 shape

Source spike, checkpoint inspection and port, all 2026-11-10 to 11-11: [HUNYUAN3D_2_1_SHAPE_ARCHITECTURE.md](../Research/HUNYUAN3D_2_1_SHAPE_ARCHITECTURE.md). The DiT (`HunYuanDiTPlain`) is 21 blocks at width 2048 with U-skips, a time token instead of AdaLN, cross-attention to DINOv2-large and a mixture of experts (8 experts, top 2, one shared expert) in the last 6 blocks, about 3.0 B parameters over 4096 latents. `Hunyuan3DPlainDit` implements it; the DINOv2-large conditioner, the VAE decoder, the sampler and the mesher are shared with 2.0. The LLM MoE feed-forward does not fit (gated SwiGLU without biases against GELU MLPs with biases), so the experts run through `GatherRows`, `Linear` and `ScatterAddRows`. The upstream checkpoint is a pickle bundle (`model.*`, `conditioner.*`, `vae.*`, all F16, 7.37 GB); it was read through the safe parser and repacked to safetensors. Licence: Tencent Hunyuan 3D 2.1 Community License, the 2.0 terms (EU, UK and South Korea excluded, 1M MAU, no training other models).

**A trap that cost the first run.** Upstream concatenates Q, K and V and views the result as `(heads, 3 * headDim)`, so a head's "q", "k" and "v" are consecutive channel groups of the fused row, not that head's own slices of three projections (the cross-attention K and V do the same with two). The weights were trained that way, so the rows are regrouped once at load. Before the fix the DiT velocity correlated 0.65 with the reference and block 0 only 0.17.

**A second trap.** The first parity run used the tanh GELU in the MLPs, experts and time embedder; upstream uses the exact erf form (`nn.GELU()`, diffusers `gelu`). That showed as a 1.2e-2 relative L2 at the first MoE block. With `GeluErf` every stage is at fp32 rounding level.

Evidence against the upstream Python (`hy3dshape` `82920d64`, fp32 CPU, `demo.png` on white, seed 0, sigma 0.35, one forward with the image tokens): the timestep token (relative L2 2.9e-7) and latent embedding (1.6e-7) correlate 1.00000000; block outputs correlate 1.00000000 at relative L2 1.2e-5 (block 0, dense), 2.0e-5 (block 10, dense), 6.1e-5 (block 15, the first MoE block) and 2.0e-5 (block 20); the final velocity correlates 1.00000000 (relative L2 2.4e-5). The bundled VAE occupancy at 2000 grid points correlates 0.99999392 (relative L2 2.0e-4; measured before the GELU change, which does not touch the VAE). Through a scratch probe on the RTX 3060 (F16 matrices, F32 activations, no cuDNN) the pipeline makes a 10-step, grid-128 mesh in 34 s: 76,656 triangles, watertight, 3 components (largest 76,634), rendered from four angles as the demo mascot with a legible sign.

**Not verified.** Only the first sigma (one forward, 32 minutes on the CPU) was compared; the unconditional pass and the second sigma were not re-run. One subject and one seed; the 10-step GPU run has no Python counterpart; F16 weights against fp32 references were not compared numerically. MoE routing is on the host and unoptimised. There is no catalog id: the repository's checkpoint is a pickle, which the downloader cannot use, so a repacked file or a repack-on-fetch step is needed first. Raw RGBA preprocessing is not ported (3a).

### Hunyuan3D Paint

Source spike, pins, file sizes and hashes, licence and reuse map: [HUNYUAN3D_PAINT_ARCHITECTURE.md](../Research/HUNYUAN3D_PAINT_ARCHITECTURE.md). Paint 2.0 (`tencent/Hunyuan3D-2`) and 2.1 (`tencent/Hunyuan3D-2.1`) are one family: Stable Diffusion 2.1-base (zero terminal SNR, v-prediction) with multiview, reference-image and (2.1) material, DINO and 3D-rotary attention, six to twelve views, then a back-projection bake.

**Licence (read from `Hunyuan3D-2.1` `LICENSE` at `82920d64`).** Tencent Hunyuan 3D 2.1 Community License: not valid in the European Union, United Kingdom and South Korea (use or display of the Works **or Output** there is unlicensed); more than 1 million monthly active users needs a separate Tencent licence; no use of the Works or Output to improve another AI model; Tencent claims no rights in Outputs; a service built on it must name its real provider and disclaim Tencent affiliation (section 3e, absent from `hy3dpaint/LICENSE`); the Acceptable Use Policy applies. The paint UNet is built on SD 2.1-base (CreativeML Open RAIL++-M use restrictions), DINOv2 is Apache-2.0, Real-ESRGAN BSD-3-Clause. The 2.0 paint files carry the 2.0 agreement with the same territory, scale, output and training terms.

**Implemented (C#, `HartsyInference.ThreeD/Paint` and `Models/Hunyuan3D/Paint`).** Camera, mesh frame and the depth-buffered screen-space rasterizer follow `custom_rasterizer`'s pixel and depth conventions; `PaintViewRenderer` renders normal, position and face maps; `PaintViewSelector` is the greedy 6-to-12 view choice; `PaintBackProjector`, `PaintTextureMerger` and the Telea `Inpainter` are the bake. The network is `HunyuanPaintUnet` (generating and reference streams over the stock ResNet and timestep blocks, with its own 2.5D transformer block: material attention, reference attention, multiview attention with 3D rotary encoding, DINO attention), `HunyuanPaintSchedule` with UniPC (2.1) and Euler-ancestral (2.0) samplers, `HunyuanPaintPipeline`, and `HunyuanPaintLoader` (safetensors, or pickles through the safe parser only). Engine: `MeshModelIds.PaintMesh`, `MeshService`, catalog ids `hunyuan3d-paint` (3.8 GB, safetensors) and `hunyuan3d-paint21` (UNet and VAE pickles, DINOv2-giant, Real-ESRGAN); CLI `--image`, with `--grid` as the texture size.

**What the tests prove (CPU, tiny random weights).** The 2.5D UNet equals the stock SD UNet when the added output projections are zero (batch 2, four levels, max difference under 2e-3); the block equals a double-precision host transcription of the upstream forward (reference attention's concatenated per-material values included; a wrong value order or multiview scale fails it); the rotary tables, voxel pooling, schedule, trailing timesteps and both samplers (an oracle model must end on its data) are hand-computed; the back-projection of a rendered position map recovers each texel's own position; the whole pipeline runs for both generations. The transcription and the oracle share this repository's reading of upstream.

**Differences from upstream, all unverified in effect.** The VAE encoder returns the posterior mean (upstream samples it). Canny edges are Sobel L1 magnitude with hysteresis and no non-maximum suppression. Hole filling is Telea (upstream: mesh-vertex fill, then Navier-Stokes). xatlas and pymeshlab are replaced by `UvUnwrapper` (the input mesh is not decimated to 40,000 faces first). The UV raster has no half-texel offset. Noise comes from `System.Random`, so a seed does not reproduce Python. DINO input resizes with PIL-style bicubic where `AutoImageProcessor` may differ in rounding; DINOv2 at 224 px uses the encoder's bicubic position interpolation. Cells whose voxel pooling upstream computes in half precision are computed in double. Metallic is 0 and roughness 1 for 2.0, whose model makes albedo only. Views are fixed at six for 2.1 through the engine because `MeshRequest` has no view-count field (adding one is a public-contract change to agree first).

**Real weights, Paint 2.0 end to end (2026-11-11).** `tencent/Hunyuan3D-2` `9cd649ba`: UNet (3,662,636,472 B, sha256 `1c5ce434…`, 1535 F16 tensors: 849 `unet.*`, 686 `unet_dual.*`) and VAE (167,335,310 B, `abcec86e…`) under `Models/3D/hunyuan3d-paint`. Findings: `unet_dual` has no `class_embedding` (only the generating stream reads camera classes); the VAE is F16 and is widened on load. The released VAE round-trips a 256 px image with a mean absolute error of 6.2/255 (CPU: encode 14.8 s, decode 29.7 s).

Run (a `paintrun` harness, not in the repository): a 19,800-triangle ellipsoid unwrapped by the toolkit into 16 charts, the reference a render of a bust, six fixed views at 512 px, 12 Euler-ancestral steps (the 2.0 default is 30), 2048 px render and texture, seed 1, RTX 3060 (driver 610.88, cuDNN 9.21 fused attention, weights promoted to F16 on the device), the GPU shared with another process. Per stage: geometry maps 2.9 s, VAE encode of 13 images 5.5 s, reference stream 1.6 s, 5.4 to 6.2 s per step (two UNet passes over six views), VAE decode of 6 views 8.6 s, bake 9.3 s; generate 94.9 s plus 6 s to load, so about 175 s at 30 steps. Memory: the device total peaked at 11.87 GB with another process resident (7.5 to 9.6 GB), so this run's own share was roughly 2.2 to 4.4 GB by difference and is not isolated. CPU numbers at 128 px (i7-9700K): one UNet pass 8.1 s, a 2-step run 274 s (VAE encode 66.9 s, steps 76.6 and 63.4 s, decode 57.1 s). Smaller GPU runs: 256 px with 4 steps 26.7 s, 128 px with 2 steps 14.7 s.

Look (rendered in a glTF viewer from the front, the back and a high three-quarter view; the images are in the shared notes under `paint-results`): the front shows the reference's face, hair, collar and buttons, upright and in the right place on the ellipsoid. The back is a plausible soft hair texture with a dark seam line across the top. Defects: the test geometry is a blob, not a bust, so the model textures a head on an egg; 12 steps are not converged (the back is soft); the dark cap seam; no delight stage, so any shading in the reference is baked in.

Two defects found by this run and fixed: Telea inpainting of the baked texture ran away over large gaps (zebra and saturated stripes), so `Inpainter` gained an optional `clampToUnit`, Paint uses it, and texels more than 12 (a square reach) away from any texel the mesh covers take the mean colour; and meshes without UVs were unwrapped for a 2048 atlas whatever the texture size. On the clean ellipsoid every atlas texel is seen by the six views. On the TRELLIS.2 test busts the toolkit's unwrapper (not Paint) is the limit: 16,237 charts for 40,929 triangles (111,770 for the 1M-triangle mesh) use only 5 to 18 % of the atlas, so most of the texture is gutter; sculpt-like inputs need a mesh with a clean atlas.

**Not verified.** Nothing is compared with the Python reference (UNet latents, VAE, sampler or bake): the run above shows the pipeline runs and looks plausible on one seed, not that it matches Tencent's. One subject, one seed, 12 of 30 steps, a test ellipsoid, and the six fixed views (view selection has not run on real weights). The seed does not reproduce Python's noise. The 2.1 path (UniPC, DINOv2-giant, material attention, 3D rotary, Real-ESRGAN) has no weights: its `.bin` key layout (`unet.*`, `unet_dual.*`, `attn1.processor.to_*_mr`, `attn_dino`, `image_proj_model_dino`) is read from the source, not from the pickle. The delight stage (InstructPix2Pix SD) is not implemented. Per-process VRAM was not isolated on the shared GPU; 2.1 adds DINOv2-giant (about 9 GB in F32 on the host).

### HairDeck

Image → hair cards from `hyz317/HairDeck` (pinned revision in the catalog). Static GLB (`hair_cards`, `spring_chains`) and rigged GLB (skin to spring chains, clips `turn`, `nod`, `jump`, `sway`).

- **Evidence:** CUDA (RTX 3060, driver 610.88) on `ex01.png`, seed 0: static and rigged GLBs written. Independent Python check of the rigged GLB: rest-pose error 1.5e-8 m; peak displacement 0.19–0.30 m per clip. Unit tests: spring simulation matches the reference's `sim.py` output to 1e-9 for turn and jump.
- **Not run:** `ex02`–`ex12`; cuDNN fast path (the CUDA 12 cuDNN build is rejected under the CUDA 13 driver); per-step timings and VRAM; real-weight integration tests.
- **Assumptions in the rig:** head-motion pivot is the mean card root; face width is 0.15 m (the reference takes both from a VRM body).
- **Cancellation:** the image sampler checks the token every step. Preprocessing (SAM 3, BiRefNet, DINOv3) does not observe it, so a cancel takes effect after the current preprocessing stage.
- **Not implemented:** VRM packaging. Upstream exports VRM 0.x; `VRMC_springBone` is VRM 1.0. A VRM export needs a VRM body asset, which is not in the repo.
- **Licences:** HairDeck weights Apache-2.0 (`hyz317/HairDeck`). SAM 3 under the SAM License (Hugging Face tag `other`, gated). DINOv3 under the DINOv3 License (Meta, `ai.meta.com`). BiRefNet-matting MIT (`ZhengPeng7/BiRefNet-matting`).

### TRELLIS

**Wired into the CLI 2026-07-21** (`TrellisImageTo3DPipeline`, `hartsy 3d -m trellis`) — DINOv2-with-registers-large conditioner → sparse-structure flow → active voxels → SLAT flow → GS decoder → `TrellisGaussianRepresentation` → binary-PLY splat. Every network stage is parity-verified vs the real model in isolation (0.9999+, see below), and the assembled CLI pipeline was run end-to-end producing a 260K-splat PLY that, viewed with a rough splat-center scatter plot (not a real Gaussian rasterizer — see caveat), shows a recognizable chair with correct dark-wood/cream-cushion coloring, rougher and less complete than TripoSR's output on the same input (e.g. not all four legs are clearly visible). **The cause of that roughness is unconfirmed** — plausible candidates include a genuine reconstruction defect, the crude non-rasterizing viewer under-rendering thin structures (few splats per leg at `s=2` scatter size), and/or the input (`chair_prep.png`) being prepared for TripoSR/Hunyuan3D's gray-composite contract rather than TRELLIS's own (alpha-premultiply crop + 518 resize) — do not assume any one of these without isolating it. **Not parity-verified against the reference pipeline's own render** (unlike the per-stage network parity below), and not verified with a real Gaussian-splat rasterizer — treat as validation-pending, not verified, until both are done. Mesh (flexicubes) and radiance-field decoders are not ported, so this pipeline only ever emits `ThreeDResult.Splats`, never `Mesh`. SLAT normalization mean/std are the fixed `microsoft/TRELLIS-image-large` `pipeline.json` constants (baked into the pipeline, not a runtime reference dump). **Caveat:** raw-image preprocessing (premultiply-alpha crop + 518 LANCZOS resize) is still Python-only, same as TripoSR/Hunyuan3D. **Network parity**: DINOv2-vitl14-reg conditioner corr 0.99994861; `SparseStructureDecoder` corr 1.0; `SparseStructureFlow` DiT velocity corr 0.99999866; `SlatFlowModel` velocity corr 0.99999984; `SlatGaussianDecoder` corr 0.99999973. Tests: `TrellisSparseOpsParityTests` (in `tests/HartsyInference.ThreeD.Tests/Parity/`). The stage-1/2, GS-decoder and generation tests were removed in the 2026-08-06 suite cleanup — the per-stage parity figures above are the record. The generation test only asserted `cloud.Count == nv*32`, a count check rather than a correctness check. **A pipeline-level check with a real splat-rasterizer render is still needed and still missing.**

### UniRig

Source spike and licences: [`UNIRIG_ARCHITECTURE.md`](../Research/UNIRIG_ARCHITECTURE.md). Upstream `VAST-AI-Research/UniRig@6793c66`, weights `VAST-AI/UniRig@36842e2`.

- **Stage 1 (skeleton), this PR:** point encoder (checked against a plain double-precision transcription on a tiny config), OPT decoder on `GenericTransformer`, tokenizer round-trip and grammar tests, mesh welding/normalization/sampling. Real-weight evidence (2026-11-09, RTX 3060 and CPU F32, sha256 `d8cf9b42…` of the 1,439,617,174-byte checkpoint): the pickle holds 585 F32 tensors that map one-to-one onto the model, and generation yields connected trees inside [-1, 1]. Rendered from four angles, Xbot gives a 52-joint humanoid (spine, head, arms, fingers, legs, feet), Horse 32 joints (spine, neck, four legs), Parrot 11 joints (wings, tail).
- **Not verified:** any numeric parity (upstream samples stochastically and no reference was dumped); upstream's 15-beam sample (single sequence here); bone count and placement quality beyond the three looks above; meshes with non-indexed GLB primitives (the existing `GlbReader` refuses them).
- **Stage 2 (skinning):** UniRig's point transformer, bone encoder, mesh-bone cross-attention, scoring head, voxel geodesic prior and `reskin`. Key map against the 4,375,464,854-byte skin checkpoint (sha256 `9d40cf42…`) is exact (1,155 tensors). Invariants hold on synthetic and real weights (rows non-negative, sum to 1); the rendered Horse skin shows contiguous dominant-joint regions per leg segment, neck, head and torso. CPU F32 only (Horse: 400 s).
- **Stage 3 (export and CLI):** `SkinnedGlbWriter` (mesh, skin, bones named `bone_i`, inverse bind matrices, JOINTS_0/WEIGHTS_0, tails in `extras`) and `hartsy 3d -m unirig --model-path <folder>` through `MeshRequest.SceneMesh` and the GLB result; no contract field added.
- **Not implemented / not verified:** FBX; GPU timing of the skin stage (the engine's CUDA backend aborted at start-up on this machine after it auto-fetched a cuDNN build); mesh decimation; non-indexed GLB primitives (existing `GlbReader` limit); anything beyond structure and a rendered look.
- **Licences:** code and weights MIT; training data Articulation-XL2.0 CC-BY-4.0; upstream's `michelangelo/` sources carry a GPL-3.0 header, so this port implements that encoder from the architecture, not from those files.

### Pixal3D

**Implemented 2026-11-11, shape-only, 🔧** (`Pixal3DImageTo3DPipeline`, `hartsy 3d <image.png> -m pixal3d [--grid 1024|1536] [--fov <radians>]`; `--grid` is the final dual-grid resolution, 1024 by default, always the cascade; 512 is rejected because Pixal3D has no 512 pipeline type). Source reading, pins, licences and the reuse map: [PIXAL3D_ARCHITECTURE.md](../Research/PIXAL3D_ARCHITECTURE.md). It is TRELLIS.2's cascade with the image reaching each flow as global DINOv3 tokens plus per-voxel features read at the voxel's projected position (and, for the shape flows, upsampled with NAF); the decoders and DINOv3 are TRELLIS.2's files. Files: `3D/pixal3d` (flows, configs, `pipeline.json`, decoders from `TencentARC/Pixal3D` at b0cb2e1b) and `naf_release.pth` from the `valeoai/NAF` release, which the catalog cannot fetch (a run without it fails at load naming the file; the catalog entry repeats the two decoders TRELLIS.2 also ships, about 1.1 GB, so it stands alone, and the resolver falls back to `3D/trellis2` when they are absent).

**Verified:**
- Flow headers (`Pixal3DRealWeightsTests`, no data read): 700 F32 tensors each, names and shapes equal the projection key map; the structure flow's saved `rope_phases` (C64) equals the rebuilt `Trellis2RopeTable` to 4.5e-7.
- `Pixal3DProjection` (camera, grid cells, `(pix + 0.5) / S · 2 - 1`) and the bilinear border sampler agree to 3e-5 with the values upstream's own `ProjGrid` and `sample_features` produce (`tests/python-reference/dump_pixal3d_projection.py`, torch 2.6, fixture in `tests/`).
- `Pixal3DNaf` (conv encoder, axial RoPE, cell-window attention evaluated where sampled) against upstream's NAF modules with the real `naf_release.pth` (`dump_pixal3d_naf.py`: restricted unpickler; `natten.na2d` replaced by a brute-force transcription of NATTEN's dilated definition) to 1e-5.
- The projection branch of the flow block (`image_attn_mode = "proj"`) against a scalar transcription of the upstream forward on tiny configs, bit-equal to the plain flow when `proj_linear` is zero; the rounding cascade quantizer against upstream's float formula for every coordinate; pipeline orchestration, cancellation and tensor release over stub stages.

- Real-weight end-to-end runs on the RTX 3060 (12 GB, CUDA), 1024 cascade, 12 steps per flow, seed 1, the default field of view, 2026-11-11 (times are wall clock, VRAM is the peak of the whole card's `memory.used` with 1.1 GB belonging to other processes; other sessions' work shared the CPU, so the times are not a benchmark):

  | Stage | Mug, 256 px synthetic | Example bust, 1248 px |
  |---|---|---|
  | DINOv3 at 512, structure flow + decoder | 44.8 s, 7.5 GB | 45.9 s, 7.6 GB |
  | 512 shape flow | 43.4 s (3,962 voxels), 7.7 GB | 30.5 s (2,980 voxels), 7.3 GB |
  | upsample ×4 | 7.2 s, 2.9 GB | 7.6 s, 2.9 GB |
  | DINOv3 at 1024 + NAF | 3.9 s, 5.6 GB | 5.9 s, 5.6 GB |
  | 1024 shape flow | 374.7 s (16,215 voxels), 9.3 GB | 256.9 s (13,030 voxels), 8.7 GB |
  | shape VAE decode | 29.7 s, 8.9 GB | 32.4 s, 8.3 GB |
  | total | 510 s, 4.24 M vertices, 8.42 M triangles | 386 s, 3.82 M vertices, 7.67 M triangles |

  Rendered (point-splat views, flat shading): the mug is an upright hollow cylinder with an open top and a loop handle (the export frame puts the handle on -x, as upstream's half turn does); the bust has a face, two braids, the hair bun, ears and both pauldrons with their spikes. Mesh health: no degenerate triangles; the mug has 88,482 boundary and 21,953 non-manifold edges of 12.7 M, the bust 152,042 and 151,199 of 11.5 M (thin sheets such as hair and pauldron tips), so neither is watertight. The mug's wall shows faint vertical streaks. A sweep that offloads the conditioning and frees all device memory leaves it unchanged (max |difference| 0).
- Two defects the real-weight runs found, both fixed: a projection block now syncs the stream at its end (without it the first 1024 step did not finish in 240 to 400 s, with it a forward over the real structure takes 17.5 to 17.9 s, matching the plain TRELLIS.2 flow's 19.3 s on synthetic coordinates), and the mesh is exported in upstream's frame, a half turn about Y, not TRELLIS.2's Z-up conversion (the first mug lay on its side).

**Not verified:** the structure stage and every later stage against the Python pipeline (no reference dump of any stage was made, so no parity claim; the run times above are this repository's only), 1536, quality beyond the two renders (one synthetic mug, one example image), the cause of the stall beyond the bisection above, and NATTEN's own border behaviour (only a transcription of it was run). Not implemented: texture, multi-view, MoGe-2 (the field of view is a flag; upstream's default 0.8576 rad is used otherwise), the `paper` branch. NAF's weights licence is not stated by its repository. Seeds: structure, 512 flow and 1024 flow use `seed`, `seed + 1`, `seed + 2` from this repository's generator, not `torch.manual_seed`, so a seed does not reproduce Python's latent.

### TRELLIS.2

**Implemented 2026-11-09, parity pending** (`Trellis2ImageTo3DPipeline`; the shape-only runs below are on real weights, the texture path was added later and is verified on synthetic networks only; `hartsy 3d <image.png> -m trellis2 --grid 512|1024|1536`; `--grid` is the final dual-grid resolution, 1024 by default, mapping to upstream's `512`, `1024_cascade`, `1536_cascade`). Architecture, pins and licences: [TRELLIS2_ARCHITECTURE.md](../Research/TRELLIS2_ARCHITECTURE.md). Checkpoints: `microsoft/TRELLIS.2-4B` at `af44b45f`, `ss_dec` from `TRELLIS-image-large`; the flow files' sha256 match the Hugging Face LFS oids.

**Verified:**
- Safetensors headers (no data read) of the three flows (640 BF16 tensors each), the structure decoder (the v1 `SparseStructureDecoder` reads exactly those 73 keys) and the shape decoder match the key maps (`Trellis2RealWeightsTests`).
- The flow block, RoPE layout, adaLN combination, sampler schedule and CFG rescale against hand-computed values and a scalar double-precision transcription of the upstream forward on tiny configs (CPU). That transcription shares this repository's reading of upstream; it is not a parity reference.
- Real-weight CUDA runs on the RTX 3060 (12 GB) at seed 1 on a synthetic 256 px alpha image of a mug (the input had an alpha channel, so no matting ran): the 512 type produced a GLB of 1,282,266 vertices and 2,528,776 triangles in about 95 s (summed step times, excluding the first model load; a structure step 3.4 s, a shape step 3.7 s); rendered in a three.js viewer it is a recognizable upright mug with a handle whose wall is built of flat axis-aligned terraces (see the next item) and whose rim is rough. The 1024 cascade produced 5,596,157 vertices and 11,070,214 triangles (261 MB) in about 23 minutes (summed step times); its first high-resolution step took 12 minutes and later steps 25 to 52 s. GPU memory peaked at 11.7 GB (about 2.8 GB of that belonged to other processes) and the process at 13.7 GB resident. Numerical inspection of the 512 mesh: finite, no degenerate triangles, 98 connected components (the largest holds 99.7% of the vertices), 41,231 boundary edges and 7,249 non-manifold edges, so it is not watertight. `fill_holes` is not applied.
- Local numerical check on the CPU with the upstream Python code (torch 2.7 CPU, a naive attention shim, F32), same noise, voxels and conditioning: DINOv3 conditioning at 512 vs HF `DINOv3ViTModel` correlation 0.99997; the 512 shape flow forward (t = 700, conditional and zero-conditioning) correlation 1.000000, max abs 2e-3; the full 12-step guided sampler with CFG rescale, denormalized, correlation 0.99993 (mean abs 0.024 on a latent of std 4.7). The structure flow and decoder were not checked this way. The CUDA and CPU decodes of one latent give the same mesh (2,528,776 vs 2,528,826 triangles).
- Shape VAE kernel-axis convention: among the six kernel-axis permutations and four of the eight per-axis flips applied to the decoder weights, the identity (kernel dimensions in the order x, y, z, as CoLa3D's shim assumes) is by far the best on a sampled latent (99.6% of flagged quads complete against at most 98%, 98 components against at least 547); an encode-then-decode of a synthetic 512 cup reproduces its voxels (651,287 of 651,456) with a smooth surface. The cup is symmetric under x and y mirroring, so that round trip cannot tell those flips apart; the sampled-latent sweep did.
- Cause of the vertical streaks and speckle: a dual-grid quad is wound the same way whichever side is inside, so about 42% of the triangles of the mug mesh (and 50% of a real-image mesh) face inward and the vertex normals averaged across them are wrong. The pipeline now unifies the winding (`FaceOrientation.Unify`, upstream's `unify_face_orientations`) before computing normals: inconsistent shared edges fall from 1.2% to 0.04% on the mug and from 19.8% to 0.01% on a bust, and the bust renders smooth. The mug's wall stays terraced: its decoded dual vertices are almost all axis-flat (99% of quads span under 0.04 voxel along their axis against a median of 0.55 for a bust from a real example image), which follows from the flat synthetic input, not from a stage fault found here.

**Not verified:** the structure stage and the 1024 stages against upstream (the cascade coordinates, the VAE upsample, the 1024 flow forward; the DINOv3 conditioning at 1024 was compared with HF `DINOv3ViTModel` on the CPU on 2026-11-11: correlation 0.999996, max abs 0.17, as at 512), and every stage on a GPU reference run; the shape VAE inherits CoLa3D's caveat (a naive convolution shim and an assumed kernel-axis convention); 1536; the BiRefNet-matting stand-in for RMBG-2.0 (which is not used); mesh handedness; any quality judgment, which needs the user's own images (a flat synthetic drawing was the only input). Noise comes from the repository's generator, so a seed does not reproduce Python's latent. The texture path: it ran on real weights on 2026-11-11 (headers match the key maps; 512 and 1024 chains complete in 140 s and 700 s on the 3060; see [the architecture note](../Research/TRELLIS2_ARCHITECTURE.md#real-weight-run-rtx-3060-12-gb-cuda-seed-1-2026-11-11)), but nothing is compared with Python, the 512 and 1024 cascade busts render as a recognisable cream marble bust, and the UV atlas fragments (specks and seams). Stays validation-pending. Memory: each 1.3B flow is read from disk and widened to F32 (5.2 GB) per request and released afterwards, so a request pays that load time; the 1024 cascade is near the 12 GB limit and its high-resolution flow dominates the time. The stages now release the pool, cached activations and cached weights the previous stage left behind (`Trellis2DeviceMemory`): without it the fine shape and texture flows ran 6 to 35 times slower on the 12 GB card, which matches the earlier "first high-resolution step took 12 minutes".

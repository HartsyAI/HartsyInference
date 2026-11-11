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
| **Hunyuan3D-2 variants** (Fast, Turbo + FlashVDM, mini, 2mv) | Each build matches the Python reference stage by stage and renders a clean mesh on one image; one subject, one seed, 2mv with three views, no cuDNN. Not ✅. ([details](#hunyuan3d-2-variants)) |
| **CoLa3D decomposer (q20)** | Implemented; parity pending [#694](https://github.com/HartsyAI/HartsyInference/pull/694). Not verified against the Python reference. |
| **UniRig** (mesh → skeleton + skin weights → rigged GLB) | Stage 1 (skeleton) implemented in `ThreeD/Models/UniRig`; real checkpoint loads (585 tensors, key map exact) and generates plausible skeletons for a humanoid, a horse and a parrot on CUDA and CPU. Skinning, rigged-GLB export and `hartsy 3d -m unirig` are not done. No Python reference, so nothing is parity-checked. ([details](#unirig)) |
| **TRELLIS.2** (image → shape-only mesh; `hartsy 3d -m trellis2`) | Implemented, shape-only, **parity pending**: no Python reference dumps exist, so no stage is checked against upstream. Texture/PBR is not implemented. Verified: all three flow checkpoints' safetensors headers (640 BF16 tensors each), the structure decoder's and shape decoder's headers match the key maps; the block, RoPE, sampler and cascade logic against hand-computed values and a scalar transcription of the upstream forward (tiny configs, CPU); a real-weight CUDA run produced a finite GLB (see [details](#trellis2)). Not verified: any numerical agreement with upstream, mesh quality against the reference, the BiRefNet stand-in for RMBG-2.0, 1536. ([research](../Research/TRELLIS2_ARCHITECTURE.md)) |

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
| **Hunyuan3D Paint** (texture/PBR) | Out of scope for the shape pipelines (multiview diffusion + UV bake). |

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
- [ ] Hunyuan3D Paint (texture / PBR).
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
- **Not implemented:** skinning (needs the 4.38 GB skin checkpoint, a PTv3 variant and a voxel geodesic prior), rigged-GLB export, CLI wiring, FBX.
- **Licences:** code and weights MIT; training data Articulation-XL2.0 CC-BY-4.0; upstream's `michelangelo/` sources carry a GPL-3.0 header, so this port implements that encoder from the architecture, not from those files.

### TRELLIS.2

**Implemented 2026-11-09, shape-only, parity pending** (`Trellis2ImageTo3DPipeline`, `hartsy 3d <image.png> -m trellis2 --grid 512|1024|1536`; `--grid` is the final dual-grid resolution, 1024 by default, mapping to upstream's `512`, `1024_cascade`, `1536_cascade`). Architecture, pins and licences: [TRELLIS2_ARCHITECTURE.md](../Research/TRELLIS2_ARCHITECTURE.md). Checkpoints: `microsoft/TRELLIS.2-4B` at `af44b45f`, `ss_dec` from `TRELLIS-image-large`; the flow files' sha256 match the Hugging Face LFS oids.

**Verified:**
- Safetensors headers (no data read) of the three flows (640 BF16 tensors each), the structure decoder (the v1 `SparseStructureDecoder` reads exactly those 73 keys) and the shape decoder match the key maps (`Trellis2RealWeightsTests`).
- The flow block, RoPE layout, adaLN combination, sampler schedule and CFG rescale against hand-computed values and a scalar double-precision transcription of the upstream forward on tiny configs (CPU). That transcription shares this repository's reading of upstream; it is not a parity reference.
- Real-weight CUDA runs on the RTX 3060 (12 GB) at seed 1 on a synthetic 256 px alpha image of a mug (the input had an alpha channel, so no matting ran): the 512 type produced a GLB of 1,282,266 vertices and 2,528,776 triangles in about 95 s (summed step times, excluding the first model load; a structure step 3.4 s, a shape step 3.7 s); rendered in a three.js viewer it is a recognizable upright mug with a handle whose wall is built of flat axis-aligned terraces (see the next item) and whose rim is rough. The 1024 cascade produced 5,596,157 vertices and 11,070,214 triangles (261 MB) in about 23 minutes (summed step times); its first high-resolution step took 12 minutes and later steps 25 to 52 s. GPU memory peaked at 11.7 GB (about 2.8 GB of that belonged to other processes) and the process at 13.7 GB resident. Numerical inspection of the 512 mesh: finite, no degenerate triangles, 98 connected components (the largest holds 99.7% of the vertices), 41,231 boundary edges and 7,249 non-manifold edges, so it is not watertight. `fill_holes` is not applied.
- Local numerical check on the CPU with the upstream Python code (torch 2.7 CPU, a naive attention shim, F32), same noise, voxels and conditioning: DINOv3 conditioning at 512 vs HF `DINOv3ViTModel` correlation 0.99997; the 512 shape flow forward (t = 700, conditional and zero-conditioning) correlation 1.000000, max abs 2e-3; the full 12-step guided sampler with CFG rescale, denormalized, correlation 0.99993 (mean abs 0.024 on a latent of std 4.7). The structure flow and decoder were not checked this way. The CUDA and CPU decodes of one latent give the same mesh (2,528,776 vs 2,528,826 triangles).
- Shape VAE kernel-axis convention: among the six kernel-axis permutations and four of the eight per-axis flips applied to the decoder weights, the identity (kernel dimensions in the order x, y, z, as CoLa3D's shim assumes) is by far the best on a sampled latent (99.6% of flagged quads complete against at most 98%, 98 components against at least 547); an encode-then-decode of a synthetic 512 cup reproduces its voxels (651,287 of 651,456) with a smooth surface. The cup is symmetric under x and y mirroring, so that round trip cannot tell those flips apart; the sampled-latent sweep did.
- Cause of the vertical streaks and speckle: a dual-grid quad is wound the same way whichever side is inside, so about 42% of the triangles of the mug mesh (and 50% of a real-image mesh) face inward and the vertex normals averaged across them are wrong. The pipeline now unifies the winding (`FaceOrientation.Unify`, upstream's `unify_face_orientations`) before computing normals: inconsistent shared edges fall from 1.2% to 0.04% on the mug and from 19.8% to 0.01% on a bust, and the bust renders smooth. The mug's wall stays terraced: its decoded dual vertices are almost all axis-flat (99% of quads span under 0.04 voxel along their axis against a median of 0.55 for a bust from a real example image), which follows from the flat synthetic input, not from a stage fault found here.

**Not verified:** the structure stage and the 1024 stages against upstream (the DINOv3 conditioning at 1024, the cascade coordinates, the VAE upsample), and every stage on a GPU reference run; the shape VAE inherits CoLa3D's caveat (a naive convolution shim and an assumed kernel-axis convention); 1536; the BiRefNet-matting stand-in for RMBG-2.0 (which is not used); mesh handedness; any quality judgment, which needs the user's own images (a flat synthetic drawing was the only input). Noise comes from the repository's generator, so a seed does not reproduce Python's latent. Texture and PBR are not implemented. Memory: each 1.3B flow is read from disk and widened to F32 (5.2 GB) per request and released afterwards, so a request pays that load time; the 1024 cascade is near the 12 GB limit and its high-resolution flow dominates the time.

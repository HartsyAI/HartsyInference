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
| **CoLa3D decomposer (q20)** | Implemented; parity pending [#694](https://github.com/HartsyAI/HartsyInference/pull/694). Not verified against the Python reference. |

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

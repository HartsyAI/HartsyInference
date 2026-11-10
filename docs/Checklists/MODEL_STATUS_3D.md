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
| **Hunyuan3D-2** | Real-weight image → mesh on GPU (`tencent/Hunyuan3D-2`). ([details](#hunyuan3d-2)) |
| **CoLa3D decomposer (q20)** | Implemented; parity pending [#694](https://github.com/HartsyAI/HartsyInference/pull/694). Not verified against the Python reference. |
| **TRELLIS.2** (image → shape-only mesh; `hartsy 3d -m trellis2`) | Implemented, shape-only, **parity pending**: no Python reference dumps exist, so no stage is checked against upstream. Texture/PBR is not implemented. Verified: all three flow checkpoints' safetensors headers (640 BF16 tensors each), the structure decoder's and shape decoder's headers match the key maps; the block, RoPE, sampler and cascade logic against hand-computed values and a scalar transcription of the upstream forward (tiny configs, CPU); a real-weight CUDA run produced a finite GLB (see [details](#trellis2)). Not verified: any numerical agreement with upstream, mesh quality against the reference, the BiRefNet stand-in for RMBG-2.0, 1536. ([research](../Research/TRELLIS2_ARCHITECTURE.md)) |

Tests: `Hunyuan3DDinoParityTests`, `Hunyuan3DVaeParityTests` (both in
`tests/HartsyInference.ThreeD.Tests/Parity/`). The DiT-parity and generation tests were removed in
the 2026-08-06 suite cleanup; the generation test only asserted `TriangleCount > 100`, which the
defective 760K-triangle mesh trivially passed, so it was never a correctness check. **The mesh-surface
defect above is still open and still needs a dedicated bisect.** Detail in the retired 3D build plan
(git history).

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
- [ ] Hunyuan3D-2 mesh "melting-wax" surface defect — bisect (ShapeVAE decode numerics / Fourier query encoding / missing floater post-process) before re-promoting to ✅.

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

Real-weight image → mesh on GPU (`tencent/Hunyuan3D-2`). Rewritten from a wrong-architecture scaffold to the real **Flux-lineage MMDiT** (16 double + 32 single, QK-RMSNorm, no RoPE) + **DINOv2-giant** conditioner (SwiGLU) + **VecSet ShapeVAE** (`post_kl` → 16 self-attn resblocks → cross-attn `geo_decoder` w/ 51-dim Fourier queries). Every *component* is numerically verified in isolation vs hy3dgen: conditioner **corr 1.0**, DiT velocity **corr 0.99999738**, VAE occupancy **corr 0.99999518**. **However, the full pipeline's actual mesh output is defective** — re-verified 2026-07-21 by rendering the `hartsy 3d` CLI's GLB output from 4 angles and visually inspecting it (not just triangle count): the mesh is a recognizable chair silhouette (seat, backrest, four legs) but the surface is covered in "melting wax" drip artifacts, at both reduced (`--steps 20 --grid 128`) and full-default (`--steps 50 --grid 256`) settings — 380K vertices / 760K triangles vs. TripoSR's clean 42K/84K on the same input, consistent with spurious surface crossings from a noisy occupancy field. This was previously marked "✅ verified end-to-end" on the strength of `TriangleCount > 100` passing and a "coherent" claim below that was never re-checked against the actual rendered geometry — see `Hunyuan3DShapePipeline.cs`'s own doc comment, which already said "Numerics validation-pending — produces a real, watertight mesh structurally; per-checkpoint fidelity awaits the reference-diff pass." Root cause **not isolated** (candidates: ShapeVAE decode numerics, Fourier query-point encoding, or a missing floater/post-process step) — needs a dedicated bisect before re-promoting to ✅. Historical perf-pass data (kept for reference, not a correctness claim): gen-perf campaign Rounds 1–8 took 71.3 → 9.2 s (30 steps, grid 128) via a bit-exact fused `Concat` kernel, device DINOv2-giant LayerScale/SwiGLU, fused DiT adaLN + QKV-split-norm kernels, a device FourierEmbed for the VAE, cuDNN fused SDPA, DiT CUDA-graph, and F16 activations; the DiT + ShapeVAE blocks are GPU-resident. Three bugs previously fixed (see [PARITY_VERIFICATION.md §Bugs](PARITY_VERIFICATION.md)): timestep `max_period=time_factor=1000`; CUDA activation-cache reshape identity; async-race/perf (the GPU-residency rewrite) — none of these explain the current mesh-surface defect. **Caveat:** background removal is done in Python (shared C# foreground-tool TODO with TripoSR).

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

### TRELLIS.2

**Implemented 2026-11-09, shape-only, parity pending** (`Trellis2ImageTo3DPipeline`, `hartsy 3d <image.png> -m trellis2 --grid 512|1024|1536`; `--grid` is the final dual-grid resolution, 1024 by default, mapping to upstream's `512`, `1024_cascade`, `1536_cascade`). Architecture, pins and licences: [TRELLIS2_ARCHITECTURE.md](../Research/TRELLIS2_ARCHITECTURE.md). Checkpoints: `microsoft/TRELLIS.2-4B` at `af44b45f`, `ss_dec` from `TRELLIS-image-large`; the flow files' sha256 match the Hugging Face LFS oids.

**Verified:**
- Safetensors headers (no data read) of the three flows (640 BF16 tensors each), the structure decoder (the v1 `SparseStructureDecoder` reads exactly those 73 keys) and the shape decoder match the key maps (`Trellis2RealWeightsTests`).
- The flow block, RoPE layout, adaLN combination, sampler schedule and CFG rescale against hand-computed values and a scalar double-precision transcription of the upstream forward on tiny configs (CPU). That transcription shares this repository's reading of upstream; it is not a parity reference.
- Real-weight CUDA runs on the RTX 3060 (12 GB) at seed 1 on a synthetic 256 px alpha image of a mug (the input had an alpha channel, so no matting ran): the 512 type produced a GLB of 1,282,266 vertices and 2,528,776 triangles in about 95 s (summed step times, excluding the first model load; a structure step 3.4 s, a shape step 3.7 s); rendered in a three.js viewer it is a recognizable upright mug with a handle (a vertical-streak artifact is visible on the wall and the rim is rough). The 1024 cascade produced 5,596,157 vertices and 11,070,214 triangles (261 MB) in about 23 minutes (summed step times); its first high-resolution step took 12 minutes and later steps 25 to 52 s. GPU memory peaked at 11.7 GB (about 2.8 GB of that belonged to other processes) and the process at 13.7 GB resident. Numerical inspection of the 512 mesh: finite, no degenerate triangles, 98 connected components (the largest holds 99.7% of the vertices), 41,231 boundary edges and 7,249 non-manifold edges, so it is not watertight. `fill_holes` is not applied.

**Not verified:** every numerical stage against upstream (no Python dumps exist; the structure and shape stages, the DINOv3 conditioning at 512 and 1024, the cascade coordinates, the VAE upsample); the shape VAE inherits CoLa3D's caveat (a naive convolution shim and an assumed kernel-axis convention); 1536; the BiRefNet-matting stand-in for RMBG-2.0 (which is not used); mesh handedness; any quality judgment, which needs the user's own images (a flat synthetic drawing was the only input). Noise comes from the repository's generator, so a seed does not reproduce Python's latent. Texture and PBR are not implemented. Memory: each 1.3B flow is read from disk and widened to F32 (5.2 GB) per request and released afterwards, so a request pays that load time; the 1024 cascade is near the 12 GB limit and its high-resolution flow dominates the time.

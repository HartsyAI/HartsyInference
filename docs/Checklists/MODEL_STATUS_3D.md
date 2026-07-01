# 3D Models — status

Concise status for 3D-generation (image → mesh) models. Build detail lives in
[PHASE_11_THREED.md](PHASE_11_THREED.md). Parity evidence lives in
[PARITY_VERIFICATION.md](PARITY_VERIFICATION.md). Legend: [MODEL_STATUS.md](MODEL_STATUS.md).

## Verified end-to-end (✅)

| Model | Notes |
|---|---|
| **Hunyuan3D-2** | Real-weight image → mesh on the RTX 4090 (`tencent/Hunyuan3D-2`, 2026-07-01). Rewritten from a wrong-architecture scaffold to the real **Flux-lineage MMDiT** (16 double + 32 single, QK-RMSNorm, no RoPE) + **DINOv2-giant** conditioner (SwiGLU) + **VecSet ShapeVAE** (`post_kl` → 16 self-attn resblocks → cross-attn `geo_decoder` w/ 51-dim Fourier queries). Every component numerically verified vs hy3dgen: conditioner **corr 1.0**, DiT velocity **corr 0.99999738**, VAE occupancy **corr 0.99999518**. Full pipeline (FlowMatchEuler ascending-sigma flow-match + 2-way CFG → ShapeVAE grid decode → marching cubes) produces a **coherent 160k-tri chair `.glb`** — **30 steps + grid 128 in 87 s on the 4090** (~2.7 s/step). The DiT + ShapeVAE blocks are **GPU-resident** (mirror the Boogu/Qwen device-resident pattern — all glue via `IBackend` ops, no mid-forward host `DataPointer` reads), which fixed both the perf (was ~62 s/step) and an async CUDA mem-pool race that used to require `CUDA_LAUNCH_BLOCKING=1`. Three bugs fixed (see [PARITY_VERIFICATION.md §Bugs](PARITY_VERIFICATION.md)): timestep `max_period=time_factor=1000`; CUDA activation-cache reshape identity; async-race/perf (the GPU-residency rewrite). **Only caveat:** background removal is done in Python (shared C# foreground-tool TODO with TripoSR). |
| **TripoSR** | Real-weight image → mesh, verified on **both CPU F32 and the RTX 3060** vs the upstream `tsr` reference (`stabilityai/TripoSR`, 2026-06-30). Every stage matches: DINO ViT-B/16 image tokens maxAbs **8.5e-6** (corr ~1.0), Transformer1D backbone scene_codes **corr 1.00000000** (std 409.33 vs 409.33; CPU maxAbs 3.4e-3), NeRF decoder density/color to 1.6e-2 / 5e-6, 64³ density grid corr **1.0** — and the density field (1.2% > iso 25) meshes into a coherent 84k-tri chair `.glb`. The scaffold was a wrong-architecture guess and was rewritten; two bugs fixed (see [PARITY_VERIFICATION.md §Bugs](PARITY_VERIFICATION.md)): DINO pos-embed interp `+0.1` fudge, and a CUDA activation-cache reshape-identity bug in the backbone. **Caveat:** background removal (rembg + resize_foreground + gray-0.5 composite) is still done in Python — the C# pipeline expects an already-composited RGB input; port it for raw-RGBA CLI use. |

Tests for both: `Hunyuan3DDinoParityTests`, `Hunyuan3DDitParityTests`, `Hunyuan3DVaeParityTests`,
`Hunyuan3DGenerationTests` (Hunyuan3D-2); `TripoSrParityTests`, `TripoSrGenerationTests`, `CudaOpBisectTests`
(TripoSR). Detail in [PHASE_11_THREED.md](PHASE_11_THREED.md) and [E2E_3D_WORKLOG.md](E2E_3D_WORKLOG.md).

## Deferred / not started (❌)

| Model | Notes |
|---|---|
| **TRELLIS** (image → Gaussian splat + mesh) | Not implemented — needs sparse 3D conv/attention (no backend op yet) + flexicubes + splat rendering. The `GaussianSplatCloud` type + PLY splat export are already in place as foundation. |
| **Hunyuan3D Paint** (texture/PBR) | Out of scope for the shape pipelines (multiview diffusion + UV bake). |

## Foundation

The `HartsyInference.ThreeD` package provides the reusable mesh / splat / triplane foundation: marching
cubes plus glTF / OBJ / PLY export, shared by both built models.

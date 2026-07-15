# 3D Models — status

Concise status for 3D-generation (image → mesh) models. Build detail lives in
[PHASE_11_THREED.md](PHASE_11_THREED.md). Parity evidence lives in
[PARITY_VERIFICATION.md](PARITY_VERIFICATION.md). Legend: [MODEL_STATUS.md](MODEL_STATUS.md).

## Verified end-to-end (✅)

| Model | Notes |
|---|---|
| **Hunyuan3D-2** | Real-weight image → mesh on the RTX 4090 (`tencent/Hunyuan3D-2`, 2026-07-01). Rewritten from a wrong-architecture scaffold to the real **Flux-lineage MMDiT** (16 double + 32 single, QK-RMSNorm, no RoPE) + **DINOv2-giant** conditioner (SwiGLU) + **VecSet ShapeVAE** (`post_kl` → 16 self-attn resblocks → cross-attn `geo_decoder` w/ 51-dim Fourier queries). Every component numerically verified vs hy3dgen: conditioner **corr 1.0**, DiT velocity **corr 0.99999738**, VAE occupancy **corr 0.99999518**. Full pipeline (FlowMatchEuler ascending-sigma flow-match + 2-way CFG → ShapeVAE grid decode → marching cubes) produces a **coherent 160k-tri chair `.glb`** — **30 steps + grid 128 in ~9.2 s on the 4090** (gen-perf campaign
Rounds 1–8, [THREED_GENPERF_PLAN.md](THREED_GENPERF_PLAN.md): 71.3 → 9.2 s = **7.75×**, within 1.6× of the Python
hy3dgen fp16 reference at 5.76 s; phase split dinov2-cond 0.87 / dit-loop 6.80 / vae-decode 1.42 / mc 0.14 s). The
big wins: a bit-exact fused `Concat` kernel (a per-slice-memcpy loop was 8.4 ms/call → 0.06 ms, dit-loop 27.7 →
7.5 s), porting the DINOv2-giant LayerScale/SwiGLU host loops to device (cond 4.1 → 0.87 s), fused DiT adaLN +
QKV-split-norm kernels, and a device FourierEmbed for the VAE; earlier: cuDNN fused SDPA + DiT CUDA-graph +
F16 activations. Every perf change is bit-exact or coherence-gated (`CudaOpBisectTests`). The DiT + ShapeVAE blocks are **GPU-resident** (mirror the Boogu/Qwen device-resident pattern — all glue via `IBackend` ops, no mid-forward host `DataPointer` reads), which fixed both the perf (was ~62 s/step) and an async CUDA mem-pool race that used to require `CUDA_LAUNCH_BLOCKING=1`. Three bugs fixed (see [PARITY_VERIFICATION.md §Bugs](PARITY_VERIFICATION.md)): timestep `max_period=time_factor=1000`; CUDA activation-cache reshape identity; async-race/perf (the GPU-residency rewrite). **Only caveat:** background removal is done in Python (shared C# foreground-tool TODO with TripoSR). |
| **TripoSR** | Real-weight image → mesh, verified on **both CPU F32 and the RTX 3060** vs the upstream `tsr` reference (`stabilityai/TripoSR`, 2026-06-30). Every stage matches: DINO ViT-B/16 image tokens maxAbs **8.5e-6** (corr ~1.0), Transformer1D backbone scene_codes **corr 1.00000000** (std 409.33 vs 409.33; CPU maxAbs 3.4e-3), NeRF decoder density/color to 1.6e-2 / 5e-6, 64³ density grid corr **1.0** — and the density field (1.2% > iso 25) meshes into a coherent 84k-tri chair `.glb`. The scaffold was a wrong-architecture guess and was rewritten; two bugs fixed (see [PARITY_VERIFICATION.md §Bugs](PARITY_VERIFICATION.md)): DINO pos-embed interp `+0.1` fudge, and a CUDA activation-cache reshape-identity bug in the backbone. **Caveat:** background removal (rembg + resize_foreground + gray-0.5 composite) is still done in Python — the C# pipeline expects an already-composited RGB input; port it for raw-RGBA CLI use. |

Tests for both: `Hunyuan3DDinoParityTests`, `Hunyuan3DDitParityTests`, `Hunyuan3DVaeParityTests`,
`Hunyuan3DGenerationTests` (Hunyuan3D-2); `TripoSrParityTests`, `TripoSrGenerationTests`, `CudaOpBisectTests`
(TripoSR). Detail in [PHASE_11_THREED.md](PHASE_11_THREED.md) and [E2E_3D_WORKLOG.md](E2E_3D_WORKLOG.md).

## Deferred / not started (❌)

| Model | Notes |
|---|---|
| **TRELLIS** (image → Gaussian splat + mesh) | 🚧 **Build underway** ([TRELLIS_BUILD_PLAN.md](TRELLIS_BUILD_PLAN.md)) — architecture mapped from the reference (two-stage flow: dense sparse-structure 16³→64³ occupancy → sparse SLAT over active voxels → GS/mesh/RF decoders), phased plan + `TrellisConfig` (exact `image-large` dims) landed (Phase A). Still needs new backend ops: Conv3d, SparseTensor + sparse conv/attention, flexicubes. `GaussianSplatCloud` + PLY export already in place. |
| **Hunyuan3D Paint** (texture/PBR) | Out of scope for the shape pipelines (multiview diffusion + UV bake). |

## Foundation

The `HartsyInference.ThreeD` package provides the reusable mesh / splat / triplane foundation: marching
cubes plus glTF / OBJ / PLY export, shared by both built models.

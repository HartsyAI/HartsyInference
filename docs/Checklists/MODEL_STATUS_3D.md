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
(TripoSR). Detail in [PHASE_11_THREED.md](PHASE_11_THREED.md).

## Deferred / not started (❌)

| Model | Notes |
|---|---|
| **TRELLIS** (image → Gaussian splat + mesh) | 🔬 **Generates end-to-end** ([TRELLIS_BUILD_PLAN.md](TRELLIS_BUILD_PLAN.md)) — the full C# generative path runs image→3D Gaussian splat: SS flow → SS decoder → active voxels → SLAT flow → denorm → GS decoder → `to_representation` → valid 3DGS `.ply` (dragon: 13115 voxels → 419680 gaussians, values sane, no NaN). Every network stage parity-verified vs the real model (0.9999+). Two-stage flow (dense sparse-structure 16³→64³ occupancy → sparse SLAT over active voxels → GS/mesh/RF decoders). **Caveats:** mesh/RF decoders + real-render check pending; the raw-image preprocessing (premultiply-alpha crop + 518 LANCZOS resize) is still done in Python (as with TripoSR/Hunyuan3D background removal). **DINOv2-vitl14-reg conditioner DONE (2026-07-15): ported to `TrellisImageConditioner` (shared `Dinov2VisionEncoder` + new `LargeReg` 4-register preset + `x_prenorm` tap + non-affine LayerNorm), parity corr 0.99994861 vs the real `dinov2_vitl14_reg` (spec/converter proven exact — torch F32 forward matches to maxAbs 1.9e-4; residual is TF32 GEMM noise on DINOv2's high-norm tokens). The generation now computes cond in-engine (no pre-computed feed).** **Stage-2 perf pass DONE (2026-07-15): SLAT flow 290 → 35.5 s (8.2×)** via an spconv rulebook conv (replaced the dense `res³`-grid `Conv3d` — a 2.1 GB grid at 2048ch — with coord-hash gather → per-offset cuBLAS GEMM → scatter-add), velocity corr preserved at 0.99999983; full generation now **~65 s** (load 5.6 · stage1 20.9 · stage2 35.5 · gs 2.8). **Phases B + C DONE + parity-verified vs the real TRELLIS.** Stage-1 dense: `Conv3d` (bit-exact), `SparseStructureDecoder` (corr 1.0), `SparseStructureFlow` DiT (velocity corr 0.99999866), FlowEuler sampler → occupancy corr 0.99964. Stage-2 sparse: `SparseTensor`/`SparseOps` (submanifold conv = scatter→Conv3d→gather; avg-pool down/up) + `SlatFlowModel` (sparse U-Net, 24 blocks reuse `SsFlowBlock`) → **SLAT flow velocity corr 0.99999984** vs the real `SLatFlowModel` (gold ref via dense-conv/attn monkeypatch). **GS decoder network** (`SlatGaussianDecoder`, swin windowed attention as a masked SDPA) → 448 gaussian params/voxel: **corr 0.99999973** vs the real `SLatGaussianDecoder`. So the **entire TRELLIS network path is parity-verified** (stage-1 + stage-2 flow + GS decoder). Tests: `TrellisStage1/2ParityTests`, `TrellisSparseOpsParityTests`, `TrellisGsDecoderParityTests`. Remaining for the first image→3D e2e: `to_representation` (448 params → `Gaussian`/`GaussianSplatCloud` → PLY) + the DINOv2-reg conditioner (Phase A); then mesh (flexicubes) / RF decoders. |
| **Hunyuan3D Paint** (texture/PBR) | Out of scope for the shape pipelines (multiview diffusion + UV bake). |

## Foundation

The `HartsyInference.ThreeD` package provides the reusable mesh / splat / triplane foundation: marching
cubes plus glTF / OBJ / PLY export, shared by both built models.

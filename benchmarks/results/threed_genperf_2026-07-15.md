# 3D mesh generation — HartsyInference vs Python reference (2026-07-15, RTX 4090)

Image → mesh, warm end-to-end seconds on a single RTX 4090
(`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`, engine `CudaBackend` ordinal 0). Input =
`examples/chair.png` foreground-composited on gray-0.5 512² (`/tmp/chair_prep.png`). Both engines produce the
coherent chair `.glb`; tri-counts differ by iso/marching-cubes settings, not correctness. Python reference =
`/tmp/benchvenv` (torch 2.13.0+cu130), `hy3dgen` fp16 / `tsr` F32.

| Model | HartsyInference | Python ref | Notes |
|---|---|---|---|
| **TripoSR** (256³ density grid) | **2.1 s** | 0.58 s (neural) | was 26.2 s (12.5×); our GPU density decode beats the reference — residual gap = host marching-cubes + small-GEMM occupancy |
| **Hunyuan3D-2 Shape** (30 steps, grid 128, fp16) | **9.2 s** | 5.76 s | was 71.3 s (7.75×); 1.6× off the reference |
| **TRELLIS-image-large** (image → 3DGS splat, 25+25 steps) | **~65 s** | — (no local baseline) | full C# path, every stage parity-verified (0.9999+); Python e2e baseline pending (needs spconv + the DINOv2-reg conditioner port). Stage-2 SLAT flow **290 → 35.5 s (8.2×)** this round |

## Hunyuan3D-2 phase split (4090, 30 steps / grid 128)

| Phase | Start (2026-07-14) | Now (2026-07-15) |
|---|---|---|
| DINOv2-giant cond | 6.0 s | **0.87 s** |
| DiT denoise (30 × 2 CFG) | ~60 s | **6.80 s** (113 ms/forward; ref ~96 ms) |
| ShapeVAE decode (128³) | 2.2 s | 1.42 s |
| Marching cubes (host) | 0.14 s | 0.14 s |
| **Total** | **71.3 s** | **9.2 s** |

## TRELLIS phase split (4090, dragon cond, 13115 active voxels)

| Phase | Before stage-2 pass | Now (2026-07-15) |
|---|---|---|
| Weights load | 5.6 s | 5.6 s |
| Stage 1 (SS flow 25× + SS decoder → active voxels) | 20.9 s | 20.9 s |
| **Stage 2 (SLAT flow 25× over active voxels)** | **~290 s** | **35.5 s** |
| GS decode + `to_representation` → PLY | 2.8 s | 2.8 s |
| **Total** | **~320 s** | **~65 s** |

Stage-2 mover: the submanifold `Conv3d` ran **dense over a full `res³` grid** (2.1 GB grid at 2048 channels,
3789 ms/conv) — nearly all of stage-2. Replaced with an **spconv rulebook** (`SparseOps.SubmanifoldConv3dSparse`):
a coord hash builds per-kernel-offset (in,out) index pairs → per offset `RowGather` → cuBLAS GEMM (the offset's
`[Cout,Cin]` weight slice) → `RowScatterAdd` — only the active voxels, no multi-GB grid, ~20–90× less compute at
high channel counts. New `row_gather_f32`/`row_scatter_add_f32` kernels + `IBackend.RowGather`/`RowScatterAdd`.
Velocity parity preserved: SLAT flow corr **0.99999983** (was 0.99999984); rulebook-conv unit corr 0.99999996.

## What moved it (all coherence-/parity-gated; see `docs/Checklists/THREED_GENPERF_PLAN.md` Rounds 1–8)

1. **Fused `Concat` kernel** (dit-loop 27.7 → 7.5 s). `CudaBackend.Concat`'s `dim>0` path issued one
   `cuMemcpyDtoDAsync` per outer element — the single-block `cat(attn, mlp)` at outer=4442 = ~8900 memcpys/concat,
   ~280k graph nodes/forward. New `dit_concat2_f32/_f16` kernel (one launch). Mesh **bit-identical**;
   `CudaOpBisectTests.Concat_*` corr 1.0 (F32) / 0.99999998 (F16). Shared win across every model that concats.
2. **DINOv2-giant host loops → device** (cond 4.07 → 0.87 s). Per-block `LayerScale` and SwiGLU silu-gate ran host
   `DataPointer` loops that drained the compute stream each block → `AffineBroadcastLastDim` + `SliceLastDim/Silu/Mul`.
3. **Fused DiT glue** (dit-loop 7.37 → 6.80 s). adaLN NormModulate (LayerNorm+(1+scale)+affine) → one
   `dit_layernorm_modulate` kernel; per-stream QKV split + QK-RMSNorm → one `dit_qkv_split_norm` kernel. Bit-exact.
4. **VAE FourierEmbed → device** (vae-decode 2.08 → 1.42 s). `fourier_embed` kernel; features stay GPU-resident
   (no host trig loop, no per-chunk H2D of the 51-dim features). Bit-exact.
5. Earlier rounds: cuDNN fused SDPA (`allowF16`), DiT CUDA-graph step capture (bit-exact replay), F16 activations.

**Batched CFG ruled out** (Round 7): measured graph-off (141 ms/fwd) ≈ graph-on (137 ms/fwd) post-Concat → <3 %
per-forward overhead left to amortize, so batching 2 forwards into one batch-2 forward ≈ same wall time. Not built.

## Reproduce

```bash
# Hunyuan3D-2 (weights at /tmp/hunyuan3d, prepped image at /tmp/chair_prep.png)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
HY3D_MODEL_DIR=/tmp/hunyuan3d HY3D_IMAGE=/tmp/chair_prep.png HY3D_STEPS=30 HY3D_GRID=128 HARTSY_3D_PHASE=1 \
dotnet test tests/HartsyInference.ThreeD.Tests/HartsyInference.ThreeD.Tests.csproj -c Release -f net10.0 \
  --filter "FullyQualifiedName~Hunyuan3D_Gpu_ImageToMesh" -l "console;verbosity=detailed"

# Python baseline
STEPS=30 RES=128 python /tmp/bench_hunyuan3d.py
```

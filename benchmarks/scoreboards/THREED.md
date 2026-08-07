# 3D mesh generation scoreboard

Warm end-to-end image → mesh seconds, HartsyInference vs the upstream Python reference implementation, on a
single RTX 4090. Baseline = Python reference packages `tsr` (TripoSR, F32) and `hy3dgen` (Hunyuan3D-2, fp16),
same GPU. See [`README.md`](README.md) for methodology. This table replaces
the prior `benchmarks/results/threed_genperf_2026-07-15.md` write-up (phase splits, kernel-level fixes,
parity gates) and the README's duplicate copy — that source file has been retired now that its numbers
live here.

| Model | GPU | HartsyInference | Python reference | Ratio | Date | Source |
|---|---|---|---|---|---|---|
| TripoSR (256³ density grid) | RTX 4090 | 2.1 s | 0.58 s (neural) | 0.28× (reference faster; gap = host marching-cubes + small-GEMM occupancy) | 2026-07-15 | threed_genperf_2026-07-15.md |
| Hunyuan3D-2 Shape (30 steps, grid 128, fp16) | RTX 4090 | 9.2 s | 5.76 s | 0.63× (reference faster; DiT per-forward 113 ms vs ref 96 ms) | 2026-07-15 | threed_genperf_2026-07-15.md |

# Flash-attention experiment findings

The July 2026 WMMA experiment is historical; do not use its integration steps as the current backend plan.
Current dispatch is in CudaBackend, shipped sources in HartsyInference.Cuda/Kernels, and open work in
[ROADMAP](../Checklists/ROADMAP.md#2-gpu-kernel-performance).

The tested TF32/F32 online-softmax implementation produced coherent Wan output but lost to the existing
materialized/cuBLAS path. Reducing shared-memory tiles improved occupancy without closing the gap.
Serial softmax, shared-memory output round-trips, and repeated barriers remained costs. This is evidence
against assuming a fused kernel is faster; register-resident output, parallel reductions, and async staging
need a new measured comparison. Preserve F32 softmax/accumulation and test large-magnitude Q/K explicitly.

Use [FLASH_ATTENTION.md](FLASH_ATTENTION.md) for algorithms and
[DEEP_KERNEL_OPTIMIZATION.md](DEEP_KERNEL_OPTIMIZATION.md) for techniques. Historical shell paths,
unbound-API claims, toolkit availability, default switches, and predicted speedup tables were removed.

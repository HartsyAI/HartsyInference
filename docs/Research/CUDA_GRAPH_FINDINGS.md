# CUDA graph replay constraints

Recorded on RTX 4090, driver 580.159.03 (2026-07-04). Current implementation:
[CudaGraph.cs](../../src/HartsyInference.Cuda/CudaGraph.cs); current gaps: [ROADMAP](../Checklists/ROADMAP.md).

High-level backend ops capture stream-ordered allocations as graph memory nodes. Replaying an allocation
node whose prior allocation is still live can fail on the second launch. AUTO_FREE_ON_LAUNCH resolved
that measured case while retaining virtual addresses. Consume a launch's outputs before relaunch;
persistent latents/inputs must remain outside the automatically freed region.

Keep host readback/synchronization outside capture and provide stable input addresses. Streaming weights
that change addresses are incompatible with an unchanged captured graph. Profile first: replay removes
host launch cost, not weight-transfer or kernel bandwidth cost. Graph support and per-model wiring already
exist; the former TryUpdate wrapper and the old generic "wire graphs next" plan are retired.

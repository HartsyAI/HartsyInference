# Phase 3.5 — Deviations from Design Plan

This document tracks every case where the Vulkan backend implementation diverged from the design plan in [VULKAN_COMPUTE_API.md](../Research/VULKAN_COMPUTE_API.md), [SPIRV_COMPUTE_SHADERS.md](../Research/SPIRV_COMPUTE_SHADERS.md), [VULKAN_MEMORY_MANAGEMENT.md](../Research/VULKAN_MEMORY_MANAGEMENT.md), or from the CUDA backend's behavior. It is a debugging journal and a guide for future Vulkan work.

Format mirrors [PHASE_3_DEVIATIONS.md](PHASE_3_DEVIATIONS.md): one entry per issue with **Design assumption → Deviation → How it was found → Fix → Impact**.

---

## (Empty — populate during implementation)

Use this template for each new deviation:

### N. Short Title

**Design assumption**: …

**Deviation**: …

**How it was found**: …

**Fix**: …

**Impact**: …

---

## Anticipated Categories

The CUDA backend's experience suggests the following classes of issues are likely. Check this list when debugging Phase 3.5 problems before assuming a novel bug:

| Category | CUDA precedent | Likely Vulkan equivalent |
|---|---|---|
| 64-bit indexing for 1024+ resolutions | PHASE_3_DEVIATIONS #12 | im2col / spatial kernels need `int64_t` extension |
| Last-dim split for gated activations | PHASE_3_DEVIATIONS #16 | GEGLU/SwiGLU kernels — same bug class |
| Stream / queue race with sync copies | PHASE_3_DEVIATIONS #19 | non-blocking submission ordering |
| Async free deferred cleanup | PHASE_3_DEVIATIONS #18 | timeline-semaphore reclamation timing |
| In-place ops invalidating activation cache | PHASE_3_DEVIATIONS #17 | barrier rules + cache invalidation |
| Weight `DataPointer` access after preload | PHASE_3_DEVIATIONS #14–15 | model code must route through `IBackend` |

Subgroup-size cross-vendor surprises are the most likely **new** category specific to Vulkan.

# CUDA performance constraints

The Phase 0–2 implementation journal is in git. Current measurements:
[scoreboards](../../benchmarks/scoreboards/); open work: [ROADMAP](../Checklists/ROADMAP.md).

The enduring lessons are load-bearing:

- GPU weight-cache hits must work after CPU weight disposal. All model math, including bias, transpose,
  and VAE projections, routes through IBackend rather than directly reading weight.DataPointer.
- Lazy host materialization makes a harmless-looking DataPointer access a stream synchronization point.
  Profile transfers as well as large GEMMs; small per-block host operations can dominate.
- Pair component preloads and releases. Broad cache eviction can discard components still needed later.
- Same-stream ordering removes per-op waits; asynchronous frees still require lifetime accounting and
  capacity retry. Upload and compute on unrelated/nonblocking streams need explicit ordering.
- In-place activation recaching must not invoke an old binding's disposer on the reused device pointer.
- Spatial indexing uses wide products; gated activations split logical dimensions. Finite output is not
  a correctness guarantee.
- FP8 storage, scale metadata, GEMM operands, and accumulation precision are different contracts.
  Preserve per-tensor and input scales through conversion/LoRA. Native FP8 support is GPU/shape-specific;
  current dispatch/defaults live in CudaBackend and EngineKnobs, not the retired EnableNativeFp8Gemm example.
- A local kernel speedup does not predict pipeline quality or total latency. Compare the same real
  checkpoint/inputs and report numerical quality plus memory and end-to-end timing.

See [CUDA_AND_PTX.md](CUDA_AND_PTX.md), [CUDA_GRAPH_FINDINGS.md](CUDA_GRAPH_FINDINGS.md),
[PROFILING_METHODOLOGY.md](PROFILING_METHODOLOGY.md), and
[Troubleshooting](../Checklists/TROUBLESHOOTING.md) for the detailed constraints and reference sources.

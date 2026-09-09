# MiniMax Music 3: performance decisions

Dated August 2026 experiments; current timing belongs in [AUDIO scoreboard](../../benchmarks/scoreboards/AUDIO.md), model coverage in [audio status](MODEL_STATUS_AUDIO.md). Re-measure absolute timings on the same tree, hardware and precision. Old session-specific hardware bookings and implementation phases are retired.

## Keep these findings

- F16 KV originally lost split-K attention and fell back to monolithic attention. Fixed in c81c42a4: 3060/q4 LM stage 16.2–16.3s → 12.4s, with the flow-stage control unchanged. The old “fix not applied” section was obsolete. Reassociation can fork sampled output without proving incorrect math.
- LM graph capture was built and removed: isolated benefit ~1.1ms/frame (~1%), with F32-only fused graph kernels incompatible with quantized F16 caches. The large claimed gain mostly came from fixing split-K, not capture. Shared dual-decode infrastructure and its shape fix remain for other consumers. Depth-graph work was dropped.
- Flow CFG batching was built, measured at −1.63s/−3.7% on a 3060/q4 flow stage (four trials per arm), and left opt-in pending real-weight flow parity. The earlier ~20% estimate was refuted.
- KV grow-and-copy was correct but removed: +~4GB peak and ~20% slower decode from distinct allocation sizes retained by the pool. Do not revive the superseded growth plan without a new discriminator.
- Maximum-length proof: 3060/q4 generated 9000 frames / 360.5s with preallocated F16 KV, peak 11,608/12,288MiB. AR 1016.0s (LM773.9/depth214.0/sampling16.1), flow1273.2s/89windows, vocoder28.3s. Paged F16 is not needed merely to reach this cap; the ~680MiB margin requires a long-song regression after residency changes.
- Pool reservation differs from live bytes. Zero retention previously added ~13s allocation/free overhead on a Krea2 1024² run; inspect used/reserved/cast bytes separately before calling high nvidia-smi usage a leak.
- Semantic-head batching was backed out: +0.2s and different same-seed songs from GEMM last-bit differences. Small vocoder/sampling stages did not justify speculative optimization.

## Precision diagnostic

Dual-row versus two single-row GEMMs used different TF32 heuristics. Relative differences:

| Shape/hardware | Default | HighPrecisionGemm |
|---|---|---|
| 32×32, RTX4090 | 1.494e-4 | 1.241e-7 |
| 32×32, RTX3060 | 1.241e-7 | 1.241e-7 |
| 4096×4096, both | ~2.9e-4 | 6.169e-7 |

GraphDecodeDualEmbedsTests pins high precision to isolate plumbing. BatchedGemmPrecisionProbe owns the precision comparison; AR parity corr0.99999913 and meanAbs8.203e-4 vs8.204e-4 did not justify changing production precision. A split-off control originally proved nothing because that knob was read only by eager attention, not FlashAttentionDev: verify that experimental controls reach the path under test.

## Remaining work

- Run MiniMaxMusic3ArParityTests, MiniMaxMusic3FlowParityTests and MiniMaxMusic3FlowStepParityTests on hardware with enough memory before changing batching defaults; record a fresh Python baseline.
- Profile host-built depth sequences and per-frame readbacks before kernel work. Confirm profiling instrumentation is active, GPU idle and ordinal correct.
- BF16 weight-cast caching can prevent a dtype-matched baseline from fitting. Any budget/lifetime change needs cross-model CUDA parity and throughput checks, not only this model.
- Recheck HeartMuLa on CUDA after shared dual-decode changes. Preserve long-duration memory coverage.

Do not mutate GPU-resident tensors through host reshape views. For memory diagnosis record per-frame free VRAM, cast-cache census and pool used/reserved bytes; distinguish increasing live allocations, reusable casts and retained pool capacity before selecting a fix.

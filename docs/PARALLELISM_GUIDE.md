# Parallelism Guide — which strategy, when

Decision-oriented companion to [`MULTI_GPU.md`](MULTI_GPU.md) (the feature reference). Start from
your **goal**, then check your **topology** row. Every measured number below comes from
[`benchmarks/results/2026-08-05_multigpu_speeds.md`](../benchmarks/results/2026-08-05_multigpu_speeds.md)
on the dev rig (RTX 4090 + RTX 3060, PCIe, **no P2P** — the consumer worst case; NVLink/P2P boxes
only get faster), and every feature names the real-weight test class that verifies it.

The three goals, one line each:

- **FIT** — the model doesn't fit one card → shard it (pooled VRAM, sequential, *not faster*).
- **LATENCY** — one generation, faster → replicate weights and compute concurrently (CFG-parallel,
  context parallelism) or move components (`--te-gpu`/`--vae-gpu`).
- **THROUGHPUT** — more requests/sec → independent engines, one per GPU, behind a queue. No new
  machinery; no numerics caveats.

Conflicts are enforced, not conventions: `EnableDitSharding`, `CfgParallelDevice`, and
`ContextParallelDevices` are mutually exclusive — a second card does one job per engine.

---

## Decision table: goal × topology

| | **Single GPU** | **Heterogeneous pair, no P2P** (e.g. 4090+3060 — the tested rig) | **Homogeneous pair, P2P/NVLink** (datacenter-class) |
|---|---|---|---|
| **FIT** | Quantize (GGUF/fp8) or block-streaming; sharding needs ≥2 devices. Same-GPU dual *backends* is multi-tenancy, not fit. | **The headline case — use sharding.** LLM/VLM/audio-LM layer split or DiT block sharding pools VRAM; byte-weighted planner puts the bigger range on the bigger card automatically. Expect same-or-few-% slower steps, plus the fp8 cross-arch fidelity caveat below. | Same mechanisms, minus the caveats: matched SMs give one GEMM regime (same-device splits measured bit-exact), and P2P skips the host round-trip on boundary copies. |
| **LATENCY** | Nothing here helps — these features all spend a second device. See the single-GPU perf docs instead. | **Case-by-case; check the measured verdicts below.** TE/VAE placement is the safest win (Wan 43.7→32.7 s, LTX-1 16.4→10.2 s). CFG-parallel wins when the replica genuinely fits with headroom (Wan ~1.8-1.9×/step) and *loses* when it doesn't (SDXL 2.6× slower). Context parallelism is verified-correct but **slower at small geometry** on this rig. | The design target for CFG-parallel, context parallelism, and (landing) tensor parallelism: matched per-token speed removes the barrier-waits-on-slow-rank problem, and NVLink/P2P removes the host-staged exchange cost that dominates on PCIe. Unvalidated tier — no such hardware on the dev rig (ROADMAP §1 "Tiers to validate"). |
| **THROUGHPUT** | Two backends on one card (`HARTSY_SAME_GPU_CONCURRENT=1` for true concurrency — opt-in pending soak; serialized default is solid). | **One `InferenceEngine` per GPU, requests split behind a queue** — each request runs the exact single-GPU path, zero cross-GPU traffic. Cards can serve different models, sidestepping the speed mismatch entirely. | Same pattern, plus disaggregated prefill/decode pools as the roadmap item (M5, not built). |

---

## FIT strategies

### LLM / VLM / audio-LM layer split

- **Config**: library `PlacementConfig.ShardDevices = ["cuda:0","cuda:1"]` (+ optional `ShardRatios`);
  CLI `--device "cuda:0+cuda:1"` (text) or `--lm-shard-gpu N`; extension `LmShardGpuId` (or
  `DitShardGpuId`, which feeds the same list). Audio precision policy: `HARTSY_AUDIO_LM_QUANT=q4k|q8|off`
  (auto: un-quantized when sharded).
- **Verified by**: `LlmShardingEngineTests` (Qwen3-32B 11.8 tok/s where one card OOMs; Llama-3.2-1B
  exact token parity), `VlmShardingEngineTests` (mllama + Qwen2.5-VL exact text parity),
  `YueLmShardingEngineTests` (bf16 pooled 8.7+4.3 GB, Whisper recall gate),
  `CosyVoiceLmShardingEngineTests`, `LlmPlacementTests`/`TextServiceShardValidationTests` (planner/validation).
- **Honest caveats**: CUDA-graph + speculative decode disabled while staged (eager decode).
  SSM/Mamba and Gemma-4 per-layer-embedding throw `NotSupportedException`. mllama sharded decode is
  *slower per-token* than unsharded (10.6 vs 14.8 tok/s) on a no-P2P box — the per-token
  vision-state peer copy stages through the host (documented follow-up). YuE bf16-sharded pays
  ~1.8× per-frame vs Q4_K single-card — that's the quality-for-speed trade, chosen deliberately.

### DiT block sharding (diffusion)

- **Config**: `ShardDevices` + `EnableDitSharding = true`; CLI `--dit-shard-gpu N`; extension
  `DitShardGpuId`. 2-way on Krea2/Flux/Chroma/HunyuanImage/MiniMax-H3/SD3.5; N-way on Qwen-Image.
- **Verified by**: per-family `*DitShardingTests` (synthetic bit-parity) + `*DitShardingVramTests`
  (real pooled residency) + `*DitShardingEngineTests` (full engine, SSIM-gated) — e.g.
  `Krea2DitShardingEngineTests`, `QwenImageDitShardingEngineTests`,
  `QwenImageDitSharding3StageTests`/`...3StageVramTests` (N-way), `MiniMaxH3DitShardingEngineTests`,
  `Sd3DitShardingEngineTests`; all in `tests/run-multigpu-campaign.sh`.
- **Honest caveats**:
  - Not faster: step-graphs/step-cache/block-streaming disabled while sharded; boundary copies cost
    a few %/step (Chroma's 49.8→39.0 s *speedup* is the exception — its baseline paid other costs).
  - **fp8 checkpoints with huge activation residuals diverge across mixed-SM pairs even with NO
    sharding**: Qwen-Image's default cross-device SSIM (~0.18) is informational-only after
    root-cause analysis (`QwenImageFp8PrecisionDiagnosticTests` — the *strong* card's native
    per-tensor fp8 GEMM is the lossy side; matching regimes via `HARTSY_FP8_NATIVE=0` recovers
    SSIM 0.9929 at ~1.7× GEMM cost). MiniMax-H3's version of this hid a real bug (scale-blind
    fallback dequant), now fixed — cross-device SSIM 0.9597, a real > 0.90 gate.
    Flux/Chroma/HunyuanImage/SD3.5 stay comfortably above their real 0.75+ bars.
  - On matched-SM cards none of that applies — same-device splits measured bit-exact (Flux 262k
    values, Qwen-Image 0/262144 mismatches).

### TE / VAE component placement (fit mode)

- **Config**: `TextEncoderDevice`/`VaeDevice`; CLI `--te-gpu`/`--vae-gpu`; extension
  `TextEncoderGpuId`/`VaeGpuId`. Composes with DiT sharding.
- **Verified by**: `FluxComponentPlacementEngineTests`, `SdxlComponentPlacementEngineTests`,
  `WanComponentPlacementEngineTests`, `WanVaeComponentPlacementEngineTests`,
  `LtxVideoComponentPlacementEngineTests`, `QwenImageCombinedPlacementShardingEngineTests`
  (placement × sharding, SSIM 0.9876); `LtxVideo2ComponentPlacementEngineTests` written but
  blocked on a ~22 GB download.
- **Honest caveat**: on mixed-SM pairs a placed fp8 T5 takes a different GEMM path (Flux TE
  placement SSIM 0.8126 — legitimate drift, bit-identical on matched cards).

---

## LATENCY strategies

### TE / VAE placement (the safest first move)

Same config as above. Wins when the encoder's evict/re-upload cycle dominates: Wan TI2V-5B
**43.7→32.7 s**, LTX-1 **16.4→10.2 s** (38% — T5-XXL dominates its small geometry). Neutral when
the moved component was never the bottleneck (Wan VAE-only: 42.0→43.3 s, the win is headroom).

### CFG-branch parallelism

- **Config**: `CfgParallelDevice`; CLI `--cfg-parallel-gpu N`; extension `CfgParallelGpuId`.
  Requires the denoiser to fit BOTH cards *with headroom*; falls back observably
  (`[CfgParallel] active/fell-back(...)/inapplicable(no-true-cfg)`).
- **Verified by**: `WanCfgParallelEngineTests`, `FluxCfgParallelFallbackTests`,
  `SdxlCfgParallelEngineTests`, `CfgBranchParallelWanTests` (bit-parity mechanism).
- **Honest caveats (measured)**: per-step concurrency is ~1.8-1.9×, but the double-preload must
  amortize — Wan at only 6 steps was net-neutral (32.9 vs 34.4 s). **SDXL on this rig is the
  cautionary tale**: SSIM 0.9984 correct, yet 2.6× *slower* denoise — the F32-staged ~10 GB UNet
  replica left the 12 GB card at 28.8 MB free, OOM-retry-thrashing every step, and the preload gate
  (exception-only) still logged `active`. If the second card lacks real headroom, don't use this.

### Context parallelism (Wan v1)

- **Config**: `PlacementConfig.ContextParallelDevices = ["cuda:0","cuda:1"]` (entry 0 = primary);
  CLI `--cp-gpu` is landing. Weights REPLICATED per rank; only per-block self-attention K/V is
  exchanged. Observable `[ContextParallel]` decision; v1 gates: 2 ranks, no MoE dual-expert,
  no step-cache.
- **Verified by**: `ContextParallelWanTests` (mechanism byte-exact), `WanContextParallelEngineTests`
  (real cross-device, SSIM 0.9616 > 0.90), `WanCrossGpuRegimeDiagnosticTests` (the 0.7774
  cross-arch control that justifies the 0.90 bar).
- **Honest caveats (measured)**: **slower on this rig at test geometry** — 35.7 s vs 22.4 s at 675
  tokens: per-block two-barrier rendezvous + host-staged K/V dominates, the 2+1 frame split makes
  every barrier wait on the ~3× slower 3060, and no P2P means all exchange traffic stages through
  the host. The win case is long sequences (≥ ~2.7k tokens; 720p-class is 27k) on balanced links —
  exactly what this rig can't demonstrate. The SSIM ceiling is hardware physics on heterogeneous
  pairs: the whole model with no CP code at all scores 0.7774 across these two cards, and regime
  flags could not close it (0.8907) — on matched GPUs expect the same-arch band instead.

### World-model decode overlap

`VaeDevice` on Oasis overlaps frame N's VAE decode with frame N+1's denoise: warm 5.20→**3.85 s**,
SSIM 0.9999, unset = byte-identical original path. Verified by `OasisVaeDeviceOverlapEngineTests`.

### Tensor parallelism (landing) + collectives (shipped)

The transport is built and verified: `NcclApi`/`ICollectiveComm`/`CollectiveComm.Create` with a
host-staged universal fallback and `CudaTopology.ProbeLinks()` topology probing — AllReduce
bit-exact, AllGather **4.79 GB/s** on this no-P2P box (`CollectiveCommTests`). That number is this
box's PCIe ceiling, and it is why GEMM-level tensor parallelism targets NVLink-class links; LLM
tensor parallelism v1 is landing on top of it now (not yet verified — no claims here until its
gates land). Qwen-Image context parallelism is landing on the same foundation.

---

## THROUGHPUT strategies

### Data-parallel engines (one per GPU)

```csharp
using InferenceEngine engine0 = new InferenceEngine("cuda", 0);
using InferenceEngine engine1 = new InferenceEngine("cuda", 1);
// requests[i] → engines[i % 2]; await Task.WhenAll(...)
```

No placement config at all — each engine runs the plain single-GPU path with its own full model
copy, so there is no cross-GPU traffic, no SSIM caveat, and heterogeneous cards just serve at
their own pace (or serve different models). Requires the model to fit ONE card — if it doesn't,
you're back to FIT. Pinned by `DataParallelServingEngineTests`
(`tests/HartsyInference.Diffusion.Tests/` — skeleton awaiting its first real run; no measured
throughput number yet). The underlying concurrency safety is verified: `DeviceGate` scopes
serialization per-ordinal, and the one shared-state race concurrent per-GPU generations ever
surfaced (`Tensor.EnsureCpuData` double-free) is fixed with regression tests
(`TensorConcurrentSyncTests`).

### Same-GPU dual backends (multi-tenant cards)

Two engine instances share one physical GPU with isolated streams/caches/mempools; serialized
per-ordinal by default (solid, 8/8 isolation tests). True concurrency is
`HARTSY_SAME_GPU_CONCURRENT=1` — the near-capacity failure was root-caused (capture-abort VA leak,
not a race) and fixed; `SameGpuConcurrentRealWeightTests` is back in the campaign gate, opt-in
pending longer soak.

---

## Quick reference: feature → config → test

| Feature | Library config | CLI | Extension | Verification class |
|---|---|---|---|---|
| LLM/audio layer split | `ShardDevices` (+`ShardRatios`) | `--device "cuda:0+cuda:1"` / `--lm-shard-gpu` | `LmShardGpuId` | `LlmShardingEngineTests`, `YueLmShardingEngineTests`, `CosyVoiceLmShardingEngineTests` |
| VLM layer split | `ShardDevices` | `--lm-shard-gpu` | `LmShardGpuId` | `VlmShardingEngineTests` |
| DiT block sharding | `ShardDevices` + `EnableDitSharding` | `--dit-shard-gpu` | `DitShardGpuId` | `*DitSharding{,Vram,Engine}Tests` per family |
| TE placement | `TextEncoderDevice` | `--te-gpu` | `TextEncoderGpuId` | `*ComponentPlacementEngineTests` |
| VAE placement | `VaeDevice` | `--vae-gpu` | `VaeGpuId` | `WanVaeComponentPlacementEngineTests`, `OasisVaeDeviceOverlapEngineTests` |
| CFG-parallel | `CfgParallelDevice` | `--cfg-parallel-gpu` | `CfgParallelGpuId` | `WanCfgParallelEngineTests`, `SdxlCfgParallelEngineTests`, `FluxCfgParallelFallbackTests` |
| Context parallel (Wan v1) | `ContextParallelDevices` | `--cp-gpu` (landing) | — | `ContextParallelWanTests`, `WanContextParallelEngineTests` |
| Collectives | automatic (`CollectiveComm.Create`) | — | — | `CollectiveCommTests` |
| Data-parallel engines | none (one engine per ordinal) | run N processes / one process, N engines | two backends, distinct `GPU_ID` | `DataParallelServingEngineTests` (skeleton) |
| Same-GPU tenants | none (same ordinal twice) | — | two backends, same `GPU_ID` | `SameGpuConcurrentRealWeightTests` |

Campaign seal: `tests/run-multigpu-campaign.sh` runs every real-weight class above filter-isolated
with `HARTSY_REQUIRE_REAL_WEIGHTS=1` — a missing checkpoint fails loudly instead of skipping.

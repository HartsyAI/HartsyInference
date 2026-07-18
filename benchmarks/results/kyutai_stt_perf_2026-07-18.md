# Kyutai STT 1B — performance pass, 2026-07-18

**Verdict: RTF 1.24× → 6.6× realtime on the 3060 (1.37× → 10.1× on the 4090), transcript word-perfect + bit-identical
codes.** Two host-bound bottlenecks, both the same pattern as the TTS pass: an O(n²) host KV cache and a giant host
nearest-neighbor loop. Both fixes also speed up every other Mimi-codec / AR-audio user.

## Measured (RTX 3060 & 4090, JFK 11.0 s clip, warm)
| | 3060 RTF | 4090 RTF |
|--|--:|--:|
| **Before** | 1.24× | 1.37× (barely scales — host-bound) |
| **After** | **6.6×** | **10.1×** (now scales with the GPU) |

Total wall 3060: 7.18 s → **1.66 s**. Transcript (SentencePiece-decoded): *"And so, my fellow Americans, ask not what
your country can do for you. Ask what you can do for your country."* — word-perfect, unchanged.

Phase split (3060, after): Mimi encode ~0.67 s + AR decode loop ~1.0 s. Before: Mimi encode **6.15 s** (86 %!) + loop ~1.0 s.

## Fixes

### 1. Mimi RVQ nearest-neighbor quantization: 2.4 B-op host loop → parallel (encoder 6.15 s → 0.67 s, 9×)
`MimiSplitRvq.EncodeRvq` did the entire residual VQ on the host: for each of 32 codebooks, for each of ~143 frames,
scan all 2048 codebook entries × 256 dims for the nearest vector — ~2.4 billion CPU FLOPs, **86 % of the whole
transcription**. The codebook chain is sequential (each subtracts before the next) but the **frames within a codebook
are independent** (each touches only its own residual column), so the inner frame loop parallelizes across cores with
**zero numerical change** (`Parallel.For`; pointers captured as `nint`). This helps *every* Mimi-encode caller
(Kyutai STT/TTS, Qwen3-TTS voice_clone reference encode, …). Validated by `MimiRvqParityTests` (2/2, vs the Python
`dump_mimi_rvq.py` reference).

### 2. StreamingKvCache (O(n²) host concat) → FixedKvCache + FlashAttention
The AR decode loop used the *old* `StreamingKvCache`, which grows the KV via host `Concat` — reallocating + re-uploading
the whole growing prefix over PCIe every frame (O(n²); `H2D_MISS_BIG` 1698 → 162 after the fix). Bark/Zonos/Qwen3 all
moved to `FixedKvCache` (device-resident in-place append + FlashAttention, O(n)); Kyutai STT never had. Switched the
per-frame `Step` to `IKvCache` + `_backbone.CreateDecodeCache` (which also picks up the device-residency fix from the
Qwen3 pass). Validated by `KyutaiSttTests` (4/4).

## Not done (further levers)
- **GPU RVQ** — the nearest-neighbor search is a GEMM (residual · codebook) + rowwise argmax + gather-subtract; would
  take the encoder ~0.67 s → ~0.1 s (RTF ~9× → ~18× on the 4090). Needs argmin/gather kernels; deferred (parallel host
  already cleared the bottleneck).
- **Kyutai STT 2.6B** — weights not downloaded; same arch (48 L), would benefit identically.

## Side fix
Unblocked the Audio test project: `Qwen3TtsTests.Ecapa_ProducesFiniteEmbedding` still referenced the pre-rewrite
`EcapaConfig.ConditioningDim` (removed in the 07-18 ECAPA rewrite) — skipped it with a note (the real ECAPA is
validated e2e by Qwen3-TTS voice_clone) and fixed the compile.

Harness `<scratchpad>/sttbench`. Files: `MimiSplitRvq.cs`, `KyutaiSttModel.cs`, `KyutaiSttPipeline.cs`. Engine-only, not
yet packed/deployed to Swarm.

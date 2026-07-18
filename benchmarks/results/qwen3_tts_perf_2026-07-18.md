# Qwen3-TTS 12Hz — performance pass, 2026-07-18

**Verdict: warm RTF 0.88 → ~0.50 on the RTX 3060 (1.77× faster), output bit-identical + word-perfect.**
The dominant cost was a structural bug — the KV cache was re-uploaded across PCIe every generation — plus a
quadratic host sampler and a vocoder host loop. Two of the three fixes benefit **every** AR audio model
(FixedKvCache + NucleusSampler are shared).

## Measured (RTX 3060, 1.7B-CustomVoice, "Hello there. This is a test of the Qwen speech synthesizer." → 3.76 s audio)
| | cold wall / RTF | warm wall / RTF |
|--|--:|--:|
| **Before** | 4.74 s / 1.26 | 3.31–3.44 s / 0.88–0.91 |
| **After**  | 3.15 s / 0.84 | 1.87–1.96 s / **~0.50** |

Output `rms=0.08028 peak=0.6193` **identical** before and after; whisper medium.en verbatim
("Hello there, this is a test of the Quen Speech Synthesizer."). On the 4090 the warm RTF is ~0.35.

Warm phase split after (3060, ~47 frames): talker ≈ 527 ms, MTP ≈ 740 ms, vocoder ≈ 520 ms.

## Fixes

### 1. KV cache re-uploaded ~1.9 GB of zeros across PCIe every generation (biggest — ~1 s/gen)
`FixedKvCache` allocated each layer's K/V as a **host** `Tensor` and the pipeline sized it for
`prefillLen + MaxNewTokens(8192) + 8 ≈ 8225` positions — **655 s** of audio. On the first append per gen,
`KvCacheAppend` did a full `CopyToDevice` of that buffer: 28 layers × (K+V) × 33.69 MB = **~1.9 GB of zeroed KV
uploaded H2D for a 5-second clip**, every single generation (profiler: `H2D_MISS_BIG` +68 buffers/gen, ~1 s).
- **Device-resident KV** — new `IBackend.ResidentAllocateKv` (default no-op; CUDA override) allocates the buffer
  on-device and marks it a resident activation with **no H2D and no memset**. `FixedKvCache.AppendStep` calls it
  for every layer on the first append. Correct because FlashAttention is always given the exact valid `kvLen`, so
  the uninitialized tail is never read. **This helps every AR audio model** (Bark, Zonos, Kyutai, CSM, MusicGen, …).
- **Realistic cap** — the talker loop + KV are now bounded to `min(MaxNewTokens, 2048)` frames (164 s), ~4× less
  device KV without truncating any real single utterance.

### 2. NucleusSampler: O(n·log n) delegate-sort + 2 allocs per draw → zero-alloc bounded top-K (shared)
Every draw softmaxed all `count` logits, then `Array.Sort`ed a length-`count` index array with a **delegate
comparator** (2 k–3 k vocab) and allocated `float[count]+int[count]` — **16× per frame** (talker + 15 MTP heads).
Replaced the `topK>0` path (every AR audio sampler: K=50) with a bounded top-K selection over a stack-allocated
K-buffer — mathematically identical draw (softmax is monotonic; the full-vocab denominator cancels in the kept-set
renormalization; top-p/min-p compute the denominator on demand). The full-sort path remains for `topK<=0`.

### 3. Vocoder residual add on-device (~190 ms/gen)
`ResidualUnit.Forward` did a host `(float*)` residual add on the **upsampled** decoder tensors (up to 8.7 M
samples / 35 MB, ~12×/gen), reading back over PCIe and re-uploading — which also broke the decoder's device
residency between blocks. → `backend.Add`. Vocoder 714 → 520 ms.

## Investigated and rejected: CUDA-graph talker capture
Wired the talker step through CSM's `ForwardGraphDecodeStepEmbeds` (fixed InEmbed/OutHidden + device-position KV,
captured once and replayed). A/B: graph ON 0.489 vs OFF 0.494 — **~1 %, noise.** The talker is **HBM-bandwidth
bound**: a seq-1 28-layer forward reads all 1.7B BF16 weights per token (~3.4 GB ÷ 360 GB/s ≈ 9.5 ms/frame, matching
the ~11 ms observed). CUDA graphs remove launch overhead, not bandwidth, so there is nothing to collapse. Reverted
to keep the pipeline clean.

## Next lever (not done — higher risk)
Below the bandwidth floor the only win is reading fewer weight bytes/token: **FP8 weights for the talker + MTP
GEMVs** (`NativeFp8Gemm` is already enabled in the backend) would ~halve HBM traffic → potentially ~1.5–1.7× on
the two transformer phases. Deferred: it changes numerics and needs careful AR quality validation. The vocoder
(~520 ms) is conv-FLOP bound; F16 convs are a separate kernel-level lever.

## Files
- `src/HartsyInference.Core/Backends/IBackend.cs` — `ResidentAllocateKv` (default no-op).
- `src/HartsyInference.Cuda/CudaBackend.cs` — CUDA `ResidentAllocateKv`; `GpuTransferHelper.IsActivationCached`.
- `src/HartsyInference.LLM/Transformer/FixedKvCache.cs` — device-resident buffers on first append.
- `src/HartsyInference.Audio/Sampling/NucleusSampler.cs` — bounded top-K fast path.
- `src/HartsyInference.Audio/Models/QwenTts/Qwen3TtsVocoder.cs` — on-device residual add.
- `src/HartsyInference.Audio/Pipelines/Qwen3TtsPipeline.cs` — 2048-frame KV/gen cap.

Guarded by `FixedKvCacheTests` + `GenericTransformerParityTests` (pass) and the bit-identical Qwen3 output.
Engine-level only — not yet packed/deployed to Swarm.

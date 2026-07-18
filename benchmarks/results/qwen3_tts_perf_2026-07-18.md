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

## vs the official Python reference — honest like-for-like (both BF16, same model, same sentence, same speaker `ryan`)
Warm-median RTF (wall ÷ audio seconds), 1 warmup + 3 timed gens each. **Neither side is quantized** — our talker/MTP
run the checkpoint's native **BF16** weights (F16 GEMM compute), the reference loads `dtype=torch.bfloat16`. Same
16-bit model on both. Both outputs whisper word-perfect.

| GPU | **HartsyInference (ours)** | official `qwen_tts` (PyTorch) | **speedup** |
|--|--:|--:|--:|
| RTX 3060 | **0.497** (1.87 s / 3.76 s) | 2.396 (~9.7 s / ~4 s) | **4.8×** |
| RTX 4090 | **0.265** (1.14 s / 4.32 s) | 2.338 (~9.7 s / ~4.2 s) | **8.8×** |

**Key finding — the reference barely scales with the GPU (3060 → 4090: 2.40 → 2.34), while ours nearly halves
(0.50 → 0.27).** The Python reference is **CPU/dispatch-bound**, proven two ways:
- **Measured GPU utilization during a reference gen on the 4090: median 16 %, max 51 % — the card sat ~84 % idle**
  (120 samples @ 0.2 s; 89/120 in the 10–50 % band). It runs on the GPU (weights on `cuda`, matmuls are GPU
  kernels) but spends most of its time waiting.
- **The decode is plain nested Hugging Face `generate()`** — `talker.generate()` (one HF eager AR step per frame),
  and inside *each* frame a *second* nested `code_predictor.generate()` for the 15-codebook MTP
  (`modeling_qwen3_tts.py:2272` + `:1671`). No `torch.compile`, no CUDA graphs, no fused/static-cache decode. So
  every frame is ~16 tiny GPU launches with HF's Python sampling/cache machinery on the CPU between each — a fixed
  per-token host latency that doesn't shrink on a faster card.

Ours is genuinely **GPU-bound** (at the HBM-bandwidth floor), so it scales with the card — the win widens from ~5×
(3060) to ~9× (4090). Caveat for fairness: this is the reference **as shipped** (`pip install qwen-tts`); a
`torch.compile`/static-cache/CUDA-graph build would reclaim some idle time, but that's not what ships, and even fully
launch-optimized PyTorch would then hit the same BF16 HBM floor our C# already sits at — the gap is host overhead we
removed, not a precision advantage.

The reference is ~2.4–2.5× *slower* than realtime on both cards; ours is ~2× (3060) to ~3.8× (4090) *faster* than
realtime. Setup: `~/qwen3ref_venv` (`pip install qwen-tts` → transformers 4.57.3, torch 2.6+cu124, sdpa attn — no
hand-compiled flash-attn, negligible for TTS's short attention); harness `/tmp/claude-1000/qwen3ref_bench.py`
(`CUDA_VISIBLE_DEVICES=0`=4090, `=1`=3060 under fastest-first order). Both are full e2e (talker + MTP + codec → 24 kHz).
Note: frame counts differ per GPU/run (F16 vs BF16 rounding → different sampled length); RTF normalizes for it, and
the reference's RTF held ~2.4 even on a 6.48 s outlier — confirming RTF as the fair metric.

## The ComfyUI custom nodes are the same path (verified)
The popular ComfyUI Qwen3-TTS nodes (`flybirdxx/ComfyUI-Qwen-TTS`, `1038lab/ComfyUI-QwenTTS`, `DarioFT/ComfyUI-Qwen3-TTS`,
…) are thin wrappers around the same `qwen_tts` / `transformers.generate()` — **none ship `torch.compile`, CUDA
graphs, or a static-cache decode**; the only knob they add over vanilla is the attention backend
(`sdpa`/`flash_attn`/`sage_attn`). Since the model is dispatch-bound (GPU ~84 % idle), that knob can't help — and
measurement confirms it: on the 4090, `flash_attention_2` gave RTF **2.77** vs `sdpa` **2.34** — flash attention was
*slightly slower* (its per-step launch cost outweighs the trivial seq-1 attention compute). So the real-world Comfy
node experience ≈ the reference numbers above (~2.3–2.8 RTF on a 4090), and a full ComfyUI graph only adds node/audio
marshaling overhead on top. Our 0.265 (4090) / 0.497 (3060) stands at **~8–10× (4090) / ~5× (3060)** faster than what
these nodes actually deliver. flash-attn 2.7.4 + sageattention installed in `~/qwen3ref_venv`; `ATTN=` env in the harness.

## Ours — both GPUs, warm
| GPU | cold RTF | warm RTF | warm wall |
|--|--:|--:|--:|
| RTX 3060 | 0.845 | **~0.497** | 1.87 s (3.76 s audio) |
| RTX 4090 | 0.580 | **~0.265** | 1.14 s (4.32 s audio) |

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

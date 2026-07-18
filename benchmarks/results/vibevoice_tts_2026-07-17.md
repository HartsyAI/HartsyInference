# VibeVoice-1.5B — perf pass (host VAE → GPU-resident) (2026-07-17)

Multi-speaker next-token-diffusion TTS (Qwen2.5-1.5B LM → per-token 20-step DPM-Solver diffusion head →
64-d acoustic-VAE latent → causal-ConvNeXt acoustic decoder + semantic re-encoder feedback). Correct but
**unusably slow** (~206 s/clip through Swarm; MODEL_STATUS_AUDIO). This is a **perf pass only** — the model was
already Swarm-verified word-correct (2026-07-13); the goal was to make it runnable without changing output.

GPU = RTX 3060 (engine `CudaBackend`, F32 activations, TF32 GEMM — the shipped default). Measured via a standalone
`CudaBackend` harness driving `VibeVoicePipeline.Synthesize` with `backend.Sync()` phase boundaries; an 8.27 s clip
(64 AR tokens → 62 `speech_diffusion` frames).

## Profile located the cost — NOT the diffusion head (the handoff's guess)

Phase-split timing on the baseline showed the bottleneck was the **VAE causal convolutions**, which ran as CPU
`float*` loops (`VibeVoiceOps.CausalConv1d` / `CausalConvTranspose1d` / `RmsNormChannelsFirst` / `LayerScaleApplyCF`),
interleaved with the GPU FFN ops in each ConvNeXt block — so every block also paid a full device↔host round-trip.
The diffusion head was only ~9 %.

| Phase | Before (host VAE) | After (GPU-resident) | Speedup |
|---|---|---|---|
| **prefill** (encode ~6 s voice ref through acoustic VAE) | 43 835 ms | **330 ms** | **133×** |
| **acoustic_decode** (per frame ×62) | 944 ms/fr | **25.5 ms/fr** | 37× |
| **semantic_encode** (per frame ×62) | 394 ms/fr | **11.5 ms/fr** | 34× |
| diffusion head (per frame ×62, 20 steps × 2 CFG) | 88 ms/fr | 88 ms/fr | — (untouched) |
| lm_step (per token) | 31 ms | 31 ms | — |
| **`Synthesize` total (8.27 s clip) — VAE pass only** | **134.4 s (RTF 16.26)** | **9.08 s (RTF 1.10)** | **14.8×** |

(Profiled total with the temporary `Sync()` boundaries was 10.21 s / RTF 1.23; the production run without them is
9.08 s / RTF 1.10.)

### Round 2 — diffusion head (after the VAE pass, it was the new top cost at ~53 %)

Two further wins took it from 9.08 s to **6.47 s (RTF 0.78, faster than real-time; 20.7× cumulative)**:

1. **Batched CFG** (9.08 → 7.21 s): upstream feeds the *same* noisy latent to the conditional and
   unconditional halves, so the head ran twice per step (N=1 each). The head is FFN-only (no cross-frame
   mixing), so cond + uncond stack into **one N=2 forward** — halving the head's tiny-GEMV launch count.
   `DenoiseLatent` builds `[speech;speech]` / `[cond;negCond]` via `backend.Concat` and combines the `[1,2,64]`
   output on the host (`CombineCfgBatched`).
2. **Head host-glue → GPU** (7.21 → 6.47 s): the head's per-forward `SliceAlongLastDim` (×14),
   `AdaLnModulate` (×5), `AdaLnGatedAdd` (×4), and `RmsNormNoAffine` (×1) were CPU `float*` loops — ~25
   device→host syncs per forward × ~1 240 forwards. Ported to resident ops: slice → `SliceLastDim`; modulate
   `x·(1+scale)+shift` → `AddScalar`+`Mul`+`Add`; gated residual → `Mul`+`Add`; affine-free RMSNorm →
   `RmsNorm` with a precomputed all-ones weight. Still zero new kernels.

| `Synthesize` total (8.27 s clip) | 134.4 s | 9.08 s | 7.21 s | **6.47 s** |
|---|---|---|---|---|
| RTF | 16.26 | 1.10 | 0.87 | **0.78** |
| | baseline | +GPU VAE | +batched CFG | +head residency |

After round 2 the four remaining stages are **balanced** (diffusion / LM / acoustic-decode / semantic-encode ≈
31 / 31 / 24 / 11 %); no single dominant cost remains.

## Fix — the whole VAE moved on-device, reusing existing `IBackend` ops (zero new kernels)

Same method as the F5-TTS pass (host Conv1D → `backend.Conv1d`), applied to the shared VibeVoice conv wrappers so
the acoustic encoder, acoustic decoder, and semantic encoder all benefit at once:

- **`SConv1d` / `SConvTranspose1d`** → `backend.Conv1d` / `backend.ConvTranspose1d` (the causal left-pad budget and
  the `trim_right_ratio=1.0` right-trim map directly onto the ops' asymmetric `padLeft/padRight`). The streaming
  path now builds `cache ++ input` with `backend.Concat` and extracts the trailing receptive-field tail with
  `backend.SliceLastDim` — the large activations never leave the GPU (only the tiny per-layer tail syncs to the
  host cache).
- **channels-first RMSNorm** (`RmsNormChannelsFirstGpu`) = `Transpose2D → backend.RmsNorm(last-dim) → Transpose2D`.
  `IBackend.RmsNorm` uses `1/√(meanSq+eps)`, matching the host reference formula bit-for-bit.
- **ConvNeXt layer-scale** (`ChannelScaleGpu`) = a groups=C, kernel-1 `backend.Conv1d` with the per-channel `gamma`
  reshaped to a `[C,1,1]` depthwise kernel at load — exact, resident, and no in-place weight mutation (which would
  have corrupted `EnsureF32`'s shared/mmap F32 tensors).
- Added `VibeVoicePipeline.PreloadWeights` (idempotent bulk H2D of every component's weights before the first gen).

No PTX kernel was added — everything composes from ops that already had CUDA overrides
(`Conv1d`/`ConvTranspose1d`/`Concat`/`SliceLastDim`/`Transpose2D`/`RmsNorm`/`Gelu`/`Add`).

## Correctness — behavior-preserving

Pure residency/precision-equivalent refactor; the math is unchanged. Output waveform vs the pre-optimization host
path (identical seed / text / reference): **corr 0.999624**, identical RMS (0.0524 both), identical Whisper
`medium.en` transcript. The small per-sample `maxAbs` (0.042) is TF32 in the GPU conv/linear path vs the host's
exact F32 — it does not compound over the autoregressive loop (same token count, same length, same words). All 15
VibeVoice unit tests (config / key-map / processor) pass. The conv wrappers are VibeVoice-only (distinct from the
identically-named EnCodec `SConv1d`), so no other model shares this code — no cross-model re-verification needed.
End-to-end word-correctness through the Swarm gallery was established 2026-07-13 and is inherited unchanged; a
gallery re-verify is the recommended final sign-off.

## Round 3 — CUDA-graph capture of the head step: TRIED, REVERTED (no-go)

Captured the shape-invariant N=2 head step (fixed device input buffers refreshed via `CopyInto` outside the
capture window; one capture replayed for all 20 steps × every token — the head step-refactored into a host-read-free
`RunFromSin` core with the sinusoidal timestep embed materialized outside). Capture succeeded cleanly (80 allocs /
79 frees, 1 launch/step via `cuGraphLaunch`), but it was a **double no-go**:

- **Wall-neutral** (6.47 → 6.49 s). After round 2's host-glue→GPU port the N=2 head step is **GPU-compute-bound**,
  not host-launch-bound — collapsing its launches into one graph launch buys nothing, and the per-step `CopyInto`
  refresh of the fixed inputs adds back what it saves. (Same lesson as the image DiTs where the step-graph is
  wall-neutral when GPU-bound.)
- **Diverged the generation.** Graph replay perturbed the TF32-sensitive AR feedback (the diffusion output
  re-encodes into the LM's next-token conditioning) enough to flip the token stream — output length 62 → 64 tokens
  and the audio content changed. Not acceptable.

Reverted to the round-2 eager batched-CFG path (kept the clean `RunFromSin`/`ComputeSinusoidal`/`TimestepMlp`
refactor, which is behavior-identical). **Final: RTF 0.78, corr 0.999585 — a strong stopping point**, well past
CosyVoice's RTF 1.34. The only remaining theoretical lever (a batched multi-KV-cache LM forward for the pos/neg CFG
streams) is high-effort for the ~19% the LM occupies and out of scope here.

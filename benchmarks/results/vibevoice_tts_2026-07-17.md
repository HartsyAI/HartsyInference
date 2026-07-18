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
| **`Synthesize` total (8.27 s clip)** | **134.4 s (RTF 16.26)** | **9.08 s (RTF 1.10)** | **14.8×** |

(Profiled total with the temporary `Sync()` boundaries was 10.21 s / RTF 1.23; the production run without them is
9.08 s / RTF 1.10.)

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

## Remaining headroom (not pursued)

The diffusion head is now the largest single cost (~53 %, 88 ms/frame). It's **launch-bound**: 62 frames × 20 steps
× 2 CFG = 2 480 forwards, each ~60 tiny `Linear` GEMVs (N=1) plus host-glue (`AdaLnModulate`/`AdaLnGatedAdd`/
`RmsNormNoAffine`/`SliceAlongLastDim`/sinusoidal timestep embed). The real lever is a **CUDA-graph capture** of the
per-step head (as noted for CosyVoice's CFM decoder); a timestep-embedding cache is tempting but would pin
activation tensors across the AR loop (GPU-residency use-after-free hazard). RTF 1.10 (warm ~9 s/clip) is a good
stopping point — better than CosyVoice's RTF 1.34.

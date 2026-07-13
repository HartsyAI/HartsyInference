# TTS / STT end-to-end benchmarks — 2026-07-12

First committed **end-to-end** speed benchmarks for the audio TTS/STT models (prior audio results were music-only:
`heartmula_music_e2e_2026-07-11.md`). Measured through the real SwarmUI + AudioLab generation path (the same
in-process C# HartsyInference engine a user hits), on **both** GPUs.

## Method
- Driver: AudioLab `/API/ProcessTTS` and `/API/ProcessSTT` (the production path, not a micro-bench).
- **RTF = generated-audio-seconds ÷ warm-gen-seconds** (higher = faster than real time). Warm = min of 3 runs after a cold warm-up (model resident).
- STT input = a 4.33 s Piper TTS clip (so TTS and STT share one signal). STT correctness spot-checked from the returned transcript.
- Engine: alpha.48 + `HARTSY_AUDIO_CONV_CUDNN=1`. Audio device pinned per GPU via `HARTSY_AUDIO_CUDA_DEVICE` (1=3060, 0=4090).
- **No GPU-Python head-to-head**: the only torch on this box is CPU-only (`2.12.1+cpu`), so a fair GPU baseline
  isn't runnable here. Upstream reference RTF claims are noted where published (they are GPU-Python targets).

## Results (RTF, higher is better)

| Model | Type | 3060 RTF | 4090 RTF | Warm 3060 | Audio | Upstream ref claim | Notes |
|---|---|---|---|---|---|---|---|
| **Piper** (VITS) | TTS | **10.4×** | 7.7× | 0.42 s | 4.33 s | — | fastest; host-bound |
| **Moonshine** | STT | **6.5×** | 6.5× | 0.67 s | 4.33 s | — | transcript correct |
| **Whisper** (base) | STT | **5.1×** | 5.4× | 0.84 s | 4.33 s | — | transcript correct (see en-US bug) |
| **MeloTTS** (en-v3) | TTS | 1.7× | 1.8× | 1.94 s | 3.23 s | — | **slow outlier** (BERT+VITS) |

## Key finding: small audio models are host/launch-bound, NOT compute-bound
The 4090 (≈3–4× the 3060's compute) gives **no meaningful speedup** here — Piper is even slightly *slower* on the
4090 (variance), Whisper/Moonshine/Melo are flat. These models spend their wall time in host orchestration + many
tiny kernel launches, not in GPU math. **Implication (answers "CUDA graphs where appropriate"): the optimization
lever for the small TTS/STT models is CUDA-graph capture / host-glue removal, which removes launch overhead —
buying a faster GPU does nothing.** This mirrors the LLM-decode graph win (dramatic on small models) and is the
opposite of the compute-bound video/music DiTs where graphs are a no-op.

## Outliers / blocked (found while sweeping — need attention)
- **Kokoro** ✅(parity) but **install 401** — loader points at `Hartsy/kokoro-82m-safetensors` which returns 401 (repo missing/gated). Needs the repack uploaded (same pattern as the YuE xcodec repack). Blocks a known-fast model.
- **Whisper `en-US` default bug** — `/API/ProcessSTT` defaults language to `en-US`, which the engine rejects ("Unknown Whisper language code 'en-US'"). Workaround: pass `en`. Fix: normalize `en-US`→`en` or change the default.
- **Spark-TTS** marked ✅ (test parity) but **not runnable through Swarm** — install fails: "SparkTtsConfig token offsets + BiCodec decoder keys checkpoint-reconciliation-pending." The ✅ is parity-harness only.
- **F5-TTS** installs, but needs a voice-reference WAV + reference text (zero-shot) — not benchmarkable with bare text.
- **MeloTTS 1.7×** — the one verified-and-runnable slow model; BERT+VITS path is the optimization target.
- Numerically-verified-but-no-runnable-e2e (do not benchmark yet): Kyutai TTS/STT, FishSpeech, Dia, VibeVoice, NeuTTS, Orpheus, Bark, StyleTTS2 (🔧/🔬); Zonos, PocketTTS (⛔ blocked).

## Remaining work
- Larger verified TTS still to bench (need per-model setup/refs): Chatterbox, CosyVoice 2, Qwen3-TTS, GPT-SoVITS, F5 (with ref).
- CUDA-graph pass on Piper/Whisper/Moonshine/Melo (host-bound → high expected payoff).
- A real GPU-Python baseline needs GPU torch installed (currently CPU-only here).

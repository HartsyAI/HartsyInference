# TTS / STT end-to-end benchmarks — 2026-07-12

First committed **end-to-end** speed benchmarks for the audio TTS/STT models (prior audio results were music-only:
`heartmula_music_e2e_2026-07-11.md`). Measured through the real SwarmUI + AudioLab generation path (the same
in-process C# HartsyInference engine a user hits), on **both** GPUs.

## Method
- **TTS driver: the canonical `/API/GenerateText2Image`** — the same universal generate path all of Swarm uses;
  the audio model is selected (`Audio Models/<Engine>/<variant>`), the text goes in the prompt, and the WAV lands
  in `/Output` exactly like an image gen. (An earlier revision of this file used the secondary `/API/ProcessTTS`;
  those numbers ran ~15–25% faster because they skip Swarm's param-processing + output-file pipeline — the
  `GenerateText2Image` numbers below are the honest end-user figures.)
- STT driver: `/API/ProcessSTT` for now (transcription output is text, not a file in `/Output`; wiring STT through
  the universal path is a separate item). STT input = a Piper TTS clip; transcript spot-checked for correctness.
- **RTF = generated-audio-seconds ÷ warm-gen-seconds** (higher = faster than real time). Warm = min of 3 runs after a cold warm-up (model resident).
- Engine: alpha.48 + Kokoro/pickle fixes (locally-deployed DLLs) + `HARTSY_AUDIO_CONV_CUDNN=1`. Audio device pinned per GPU via `HARTSY_AUDIO_CUDA_DEVICE` (1=3060, 0=4090).
- **No GPU-Python head-to-head**: the only torch on this box is CPU-only (`2.12.1+cpu`), so a fair GPU baseline
  isn't runnable here. Upstream reference RTF claims are noted where published (they are GPU-Python targets).

## Results (RTF, higher is better)

TTS via the canonical `GenerateText2Image` path; STT via `ProcessSTT`.

| Model | Type | 3060 RTF | 4090 RTF | Audio | Notes |
|---|---|---|---|---|---|
| **Piper** (VITS) | TTS | **8.6×** | 8.3× | 5.25 s | fastest; host-bound (4090 = 3060) |
| **Kokoro** (StyleTTS2) | TTS | 4.5× | **5.2×** | 6.45 s | **now works** (canonical-fallback fix); slightly compute-bound |
| **Moonshine** | STT | 6.5× | 6.5× | 4.33 s | transcript correct |
| **Whisper** (base) | STT | 5.1× | 5.4× | 4.33 s | transcript correct (see en-US bug) |
| **MeloTTS** (en-v3) | TTS | 1.4× | 1.4× | 4.45 s | **slow outlier** (BERT+VITS), fully host/GPU-flat |

## Key finding: small audio models are host/launch-bound, NOT compute-bound
The 4090 (≈3–4× the 3060's compute) gives **no meaningful speedup** here — Piper is even slightly *slower* on the
4090 (variance), Whisper/Moonshine/Melo are flat. These models spend their wall time in host orchestration + many
tiny kernel launches, not in GPU math. **Implication (answers "CUDA graphs where appropriate"): the optimization
lever for the small TTS/STT models is CUDA-graph capture / host-glue removal, which removes launch overhead —
buying a faster GPU does nothing.** This mirrors the LLM-decode graph win (dramatic on small models) and is the
opposite of the compute-bound video/music DiTs where graphs are a no-op.

## Outliers / blocked (found while sweeping — need attention)
- **Kokoro** — **FIXED 2026-07-12**. Was install-401 (`KokoroPipeline` only pulled the unpublished `Hartsy/kokoro-82m-safetensors` repack, no fallback). Now: prefer the repack, else download canonical `hexgrad/Kokoro-82M/kokoro-v1_0.pth` and do the flatten + inner-`module.`-strip in-engine (cached once). Installs + generates via the canonical path; 4.5×/5.2× above.
- **Whisper `en-US` default bug** — `/API/ProcessSTT` defaults language to `en-US`, which the engine rejects ("Unknown Whisper language code 'en-US'"). Workaround: pass `en`. Fix: normalize `en-US`→`en` or change the default.
- **Spark-TTS** marked ✅ (test parity) but **not runnable through Swarm** — install fails: "SparkTtsConfig token offsets + BiCodec decoder keys checkpoint-reconciliation-pending." The ✅ is parity-harness only.
- **F5-TTS** installs, but needs a voice-reference WAV + reference text (zero-shot) — not benchmarkable with bare text.
- **MeloTTS 1.7×** — the one verified-and-runnable slow model; BERT+VITS path is the optimization target.
- Numerically-verified-but-no-runnable-e2e (do not benchmark yet): Kyutai TTS/STT, FishSpeech, Dia, VibeVoice, NeuTTS, Orpheus, Bark, StyleTTS2 (🔧/🔬); Zonos, PocketTTS (⛔ blocked).

## Remaining work
- Larger verified TTS still to bench (need per-model setup/refs): Chatterbox, CosyVoice 2, Qwen3-TTS, GPT-SoVITS, F5 (with ref).
- CUDA-graph pass on Piper/Whisper/Moonshine/Melo (host-bound → high expected payoff).
- A real GPU-Python baseline needs GPU torch installed (currently CPU-only here).

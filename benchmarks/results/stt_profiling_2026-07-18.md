# STT profiling — which model needs a perf pass, 2026-07-18

**Verdict: Kyutai STT 1B is the only STT worth a perf pass (~1.24× realtime, barely scales with GPU = host-bound).
Everything else is comfortably fast.** Profiled before optimizing (per the "profile first" rule) rather than guessing.

Method: transcribe the 11.0 s JFK clip (`ComfyUI/input/jfk.wav`), 1 cold + several warm runs, warm = best.
RTF = audio-seconds ÷ wall-seconds (**higher = faster than realtime**). Harness `<scratchpad>/sttbench` (same engine
build as the Qwen3-TTS perf pass, so the FixedKvCache device-residency fix is already included).

| Model | type | 3060 RTF | 4090 RTF | scales w/ GPU? | verdict |
|--|--|--:|--:|--|--|
| **Kyutai STT 1B** (`stt-1b-en_fr-trfs`) | AR delayed-streams (16L, 2048, Mimi codec) | **1.24×** | **1.37×** | **no** (1.24→1.37) | **⟵ perf-pass target** |
| distil-large-v3 | encoder-decoder | 3.28× | 2.79× | no | fine (3× realtime) |
| Moonshine | encoder-decoder (small) | 6.5× | 6.5× | — | fine (from 07-12 bench) |
| whisper-base | encoder-decoder (small) | ~10× | — | — | fine |

All are word-correct (whisper-base/distil verbatim JFK; Kyutai STT verified word-perfect earlier — 54 text tokens here).

## Why Kyutai STT 1B is the target
- **~1.24× realtime is barely above the bar** — 8.9 s to transcribe 11 s of audio (~65 ms per 12.5 Hz frame for a
  16-layer 1B model; the bandwidth floor should be ~10–15 ms/frame).
- **It barely scales with the GPU (3060 1.24× → 4090 1.37×)** — the tell-tale signature of a host/dispatch-bound AR
  loop (same as the Python-reference finding: the GPU sits idle waiting on host work). A 2.7×-faster card buys ~10 %.
- It's an AR delayed-streams decoder (Helium backbone + Mimi *encoder* → text tokens) — the same model class as the
  TTS models just optimized, so it likely has the same removable host-glue (per-op `(float*)` loops, host sampling,
  Mimi-encoder host convs). The FixedKvCache device-residency fix already applies; this 1.24× is *after* that, so the
  remaining cost is elsewhere (Mimi encoder and/or backbone glue) — to be found by profiling.

Next: profile Kyutai STT 1B's decode to locate the host-bound hotspot(s).

## Not measured
- Kyutai STT 2.6B — weights not downloaded locally (`kyutai--stt-2.6b-en-trfs` dir empty). Same arch, 48L/2.6B; would
  be slower still and benefit from the same fixes.

# Model Status — index (by modality)

One concise per-modality status doc tells you, at a glance, which models are **fully complete and
verified end-to-end** versus built-but-pending. Each type doc is the model *list* (status table);
the matching `PHASE_*` doc holds the per-model build detail, deviations, and task plans, and
[`PARITY_VERIFICATION.md`](PARITY_VERIFICATION.md) is the cross-modality source of truth for what has
been *proven correct against real weights*.

## The docs

| Modality | Status doc | Build detail / plan |
|---|---|---|
| **Image** (diffusion T2I) | [MODEL_STATUS_IMAGE.md](MODEL_STATUS_IMAGE.md) | [PHASE_4_MODEL_BREADTH.md](PHASE_4_MODEL_BREADTH.md), [PHASE_3_DEVIATIONS.md](PHASE_3_DEVIATIONS.md) |
| **Audio** (TTS / STT / codec / VC / music / separation) | [MODEL_STATUS_AUDIO.md](MODEL_STATUS_AUDIO.md) | [PHASE_5_AUDIO.md](PHASE_5_AUDIO.md), [MUSIC_MODELS_COMPLETION_PLAN.md](MUSIC_MODELS_COMPLETION_PLAN.md) |
| **Video** (T2V / I2V) | [MODEL_STATUS_VIDEO.md](MODEL_STATUS_VIDEO.md) | [PHASE_9_VIDEO.md](PHASE_9_VIDEO.md) |
| **World / interactive** | [MODEL_STATUS_WORLD.md](MODEL_STATUS_WORLD.md) | [PHASE_10_INTERACTIVE.md](PHASE_10_INTERACTIVE.md) |
| **3D** (image → mesh) | [MODEL_STATUS_3D.md](MODEL_STATUS_3D.md) | [PHASE_11_THREED.md](PHASE_11_THREED.md) |
| **Vision** (CLIP / detection / segmentation) | [MODEL_STATUS_VISION.md](MODEL_STATUS_VISION.md) | [PHASE_6_VISION.md](PHASE_6_VISION.md) |
| **LLM + text encoders + VLMs + embeddings** | [MODEL_STATUS_LLM.md](MODEL_STATUS_LLM.md) | [LLM_MODEL_COVERAGE.md](LLM_MODEL_COVERAGE.md), [PHASE_12_LANGUAGE.md](PHASE_12_LANGUAGE.md); **decode perf:** [LLM_THROUGHPUT_BENCHMARK.md](LLM_THROUGHPUT_BENCHMARK.md) + [LLM_DECODE_PERF_GRIND.md](LLM_DECODE_PERF_GRIND.md) (2026-07-04: 20-54× → 1.94-2.88× off llama.cpp) |

## Shared legend

Every status doc uses the same symbols:

- ✅ **verified end-to-end** — runs on real weights and the output is confirmed correct (clean visual
  output, transcription, bit/spectral parity, or coherent generation).
- 🔬 **numerically parity-verified** — risky components (or the DiT core) match a Python reference to
  tolerance, but the full real-weight end-to-end run is still pending or env-gated.
- 🔧 **built, validation-pending** — implementation is green and structurally tested; awaits checkpoint
  download + a layer-diff pass to reach ✅.
- 🚧 **scaffold only** — types/API reserved, full implementation pending.
- ⛔ **blocked** — gated weights or an external dependency stops verification.
- ❌ **not started**.

## How to read "verified e2e"

A model is counted as ✅ only when it has been run against **real downloaded weights** and the output
checked, not merely "finite floats" from a synthetic structural test. The bar and the per-model
parity evidence (maxAbs, components checked, bugs found) live in
[`PARITY_VERIFICATION.md`](PARITY_VERIFICATION.md).

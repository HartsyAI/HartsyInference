# Checklists — index

This folder is deliberately small. It holds four kinds of doc and nothing else:

1. **Per-modality status + open work** — one doc per modality. Each has the model *status table* (what's
   verified end-to-end vs. built-but-pending) **and** a `## Remaining work` checklist of what's left for
   that modality. This is where per-model work is tracked.
2. **[ROADMAP.md](ROADMAP.md)** — the cross-cutting engineering roadmap: multi-GPU / model-sharding,
   AMD/ROCm + Vulkan, kernel performance, quantization, LLM serving throughput, robotics, new SwarmUI
   extensions, CLI/API, and release/NuGet. Anything that spans modalities lives here, not in a status doc.
3. **[PARITY_VERIFICATION.md](PARITY_VERIFICATION.md)** — the cross-modality source of truth for what has
   been *proven correct against real weights* (maxAbs, components checked, bugs found).
4. **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** — the consolidated model bring-up debugging reference:
   common bugs, model-specific gotchas, notable outliers, and the parity-debugging methodology. **Read
   this first when a new model port is wrong, crashes, or is slow.**

> History note: the old per-phase build logs (`PHASE_*`), perf grinds (`*_GRIND`, `*_BENCHMARK`),
> handoffs, and one-off plans were consolidated into the four buckets above and deleted. The full
> originals remain recoverable from git history.

## Per-modality status docs

| Modality | Status + open work |
|---|---|
| **Image** (diffusion T2I) | [MODEL_STATUS_IMAGE.md](MODEL_STATUS_IMAGE.md) |
| **Audio** (TTS / STT / codec / VC / music / separation) | [MODEL_STATUS_AUDIO.md](MODEL_STATUS_AUDIO.md) |
| **Video** (T2V / I2V) | [MODEL_STATUS_VIDEO.md](MODEL_STATUS_VIDEO.md) |
| **World / interactive** | [MODEL_STATUS_WORLD.md](MODEL_STATUS_WORLD.md) |
| **3D** (image → mesh) | [MODEL_STATUS_3D.md](MODEL_STATUS_3D.md) |
| **Vision** (CLIP / detection / segmentation) | [MODEL_STATUS_VISION.md](MODEL_STATUS_VISION.md) |
| **LLM + text encoders + VLMs + embeddings** | [MODEL_STATUS_LLM.md](MODEL_STATUS_LLM.md) |

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
checked, not merely "finite floats" from a synthetic structural test. The bar and the per-model parity
evidence (maxAbs, components checked, bugs found) live in
[`PARITY_VERIFICATION.md`](PARITY_VERIFICATION.md).

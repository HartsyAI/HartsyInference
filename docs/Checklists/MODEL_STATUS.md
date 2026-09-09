# Model status index

Load only the modality needed. Cross-cutting work: [ROADMAP](ROADMAP.md); numerical evidence: [PARITY_VERIFICATION](PARITY_VERIFICATION.md); failure patterns: [TROUBLESHOOTING](TROUBLESHOOTING.md).

| Modality | Status + open work |
|---|---|
| **Image** (diffusion T2I) | [MODEL_STATUS_IMAGE.md](MODEL_STATUS_IMAGE.md) |
| **Audio** (TTS / STT / codec / VC / music / separation) | [MODEL_STATUS_AUDIO.md](MODEL_STATUS_AUDIO.md) |
| **Video** (T2V / I2V) | [MODEL_STATUS_VIDEO.md](MODEL_STATUS_VIDEO.md) |
| **World / interactive** | [MODEL_STATUS_WORLD.md](MODEL_STATUS_WORLD.md) |
| **3D** (image → mesh) | [MODEL_STATUS_3D.md](MODEL_STATUS_3D.md) |
| **Vision** (CLIP / detection / segmentation) | [MODEL_STATUS_VISION.md](MODEL_STATUS_VISION.md) |
| **LLM + text encoders + VLMs + embeddings** | [MODEL_STATUS_LLM.md](MODEL_STATUS_LLM.md) |


Legend: ✅ real-weight end-to-end output checked; 🔬 scoped numerical parity; 🔧 built, verification incomplete; 🚧 scaffold; ⛔ blocked; ❌ not started; ⚠️ mixed coverage.

Read the evidence beside each symbol: a coherent generation, component comparison, consumer run and full numerical parity establish different things. Synthetic finite tensors or a skipped resource-gated test do not establish real-weight verification. Per-model gaps belong in each modality's Remaining work section.

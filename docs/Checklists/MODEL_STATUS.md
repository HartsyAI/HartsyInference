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

Every symbol in these files was earned on an RTX 4090 or an RTX 3060 unless the row says otherwise. As of
2026-09-25 exactly two models have been run end-to-end on Blackwell (sd15 fp16 and Llama-3.2-1B q8_0, on a
rented RTX PRO 6000) — see [the image scoreboard](../../benchmarks/scoreboards/IMAGE.md); a symbol here is
not a Blackwell claim.

Read the evidence beside each symbol: a coherent generation, component comparison, consumer run and full numerical parity establish different things. Synthetic finite tensors or a skipped resource-gated test do not establish real-weight verification. Per-model gaps belong in each modality's Remaining work section.

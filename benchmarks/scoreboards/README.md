# Benchmark scoreboards

One canonical Markdown table per model type, each comparing HartsyInference against the relevant
reference engine on the same GPU, same request. These files replace the multiple diverging copies of the
same numbers that used to live scattered across `README.md`, `docs/PERFORMANCE.md`, per-modality
`docs/Checklists/MODEL_STATUS_*.md` files, and ~40 dated one-off write-ups in `benchmarks/results/`. If
you're looking for a performance number, start here — not in the README, not in `PERFORMANCE.md`.

| File | Modality | Baseline compared | GPUs measured |
|---|---|---|---|
| [`IMAGE.md`](IMAGE.md) | Image generation (T2I / edit) | ComfyUI (via SwarmUI API); Python `diffusers` for F-Lite | RTX 4090, RTX 3060 |
| [`VIDEO.md`](VIDEO.md) | Video generation (T2V) | ComfyUI (via SwarmUI API) | RTX 4090 |
| [`AUDIO.md`](AUDIO.md) | TTS / STT / Music / VC / Fx | Model-specific Python reference (`moshi`, `qwen_tts`, etc.) or self-comparison — no shared engine exists for audio | RTX 3060, RTX 4090, some CPU |
| [`LLM.md`](LLM.md) | LLM decode throughput | `llama.cpp` / `llama-cpp-python`, same GGUF quant both sides | RTX 3060 |
| [`THREED.md`](THREED.md) | Image → 3D mesh | Python reference (`tsr` for TripoSR, `hy3dgen` for Hunyuan3D-2) | RTX 4090 |

**Not yet benchmarked:** Vision and World-model modalities have no measured end-to-end performance data
yet (see `docs/Checklists/MODEL_STATUS_VISION.md` / `MODEL_STATUS_WORLD.md` for their build/parity
status instead — those docs track correctness, not speed).

**GPUs.** Only RTX 3060 (12 GB) and RTX 4090 (24 GB) have actual measured results anywhere in this repo.
A100/H100/L40S appear only in `../CLOUD_GPU_RUNBOOK.md` as a rental-pricing plan for a future baseline
pass — no data exists for them yet; don't cite them as "supported hardware" until a scoreboard row backs
it up.

**Methodology, the standard performance profile (which optimizations are on by default), and how to
reproduce a number** all live in [`../../docs/PERFORMANCE.md`](../../docs/PERFORMANCE.md) — these
scoreboard files hold the results, that file holds the how/why.

**Model list.** For "what models exist and what's their build/verification status" (as opposed to "how
fast is it"), see `docs/Checklists/MODEL_STATUS.md` and its per-modality children — that's the
canonical model list the top-level `README.md` links to. These scoreboard files only cover models that
have actually been benchmarked; a model can be fully shipped and verified without a row here yet.

**Updating a number.** Edit the row in place, update its Date and Source columns, and don't leave a
second copy anywhere else — that duplication is exactly what these files replaced.

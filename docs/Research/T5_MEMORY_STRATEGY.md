# T5 memory and precision

Pipeline-specific residency decisions belong in Engine/recipes; current coverage is in [image status](../Checklists/MODEL_STATUS_IMAGE.md). GGUF loading and native FP8 paths already exist: the former “future reader” plan was obsolete.

- Budget weights, cast caches, activations, workspace and retained allocations separately. Parameter-count arithmetic gives storage only, not measured peak VRAM.
- Preserve scaled-FP8 metadata through conversion and fallback dequantization. A dtype cast is not a calibrated quantization procedure; validate prompt adherence on saved embeddings and outputs.
- Release text-encoder device weights after encoding when the placement policy needs the memory for denoising; pair stage preload/free and synchronize ownership transitions. Preserve embeddings and any host weights needed for reload/streaming.
- A separate text-encoder GPU can avoid repeated eviction/re-upload; see [placement](../MULTI_GPU.md). Do not prescribe eviction universally when capacity and reuse favor residency.
- Assess encoder and denoiser quantization separately. Lower precision can change spatial/text adherence; the old universal Q4 quality ranking and untraceable ~3% CLIP-score claim are not acceptance evidence.
- Keep VAE precision at the family's validated setting. New lower-precision VAE paths require their own numerical/output gates; a blanket “never below F16” rule conflicts with gated quantization work.

Check exact tensor counts and DType byte layouts for storage estimates. Q8_0 uses block scales, not the old document's claimed zero-point. Do not repeat the former SD3.5 size labels or internally inconsistent half-size weight budgets.

For changes, compare saved text embeddings and full generations, log peak memory per stage, and exercise repeated prompts, eviction/reload and constrained-memory failure paths. [Troubleshooting](../Checklists/TROUBLESHOOTING.md) retains the cache/ownership failure classes.

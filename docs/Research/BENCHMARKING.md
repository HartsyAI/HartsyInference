# Benchmarking SharpInference vs ComfyUI

> **Goal:** SharpInference within 2× of ComfyUI on the same hardware running the same model + same noise + same scheduler config.
> **Status:** procedure documented, not yet collected — needs a paired dual-run on the user's RTX 3060 (or any single-GPU box) once checkpoints are downloaded.

## What to measure

Three deltas matter, in priority order:

1. **Steady-state denoise it/s** — the only number that scales with longer generations. Measure between step 5 and step 15 (skip the first few steps because the GPU memory pool warms up and CUDA caches the kernel sequence).
2. **Time-to-first-pixel (TTFP)** — wall-clock from "user clicks generate" to "VAE decode finishes". Captures load + cast + encode + denoise + decode. Closer to user-perceived latency than it/s alone.
3. **Peak VRAM** — `nvidia-smi --query-gpu=memory.used --format=csv -l 1` while generation runs. Confirms the eviction discipline holds and there are no leaks between generations.

## Reference matrix

| Model | Resolution | Steps | Quality preset | Expected ComfyUI it/s on RTX 3060 |
|---|---|---|---|---|
| SDXL F16 (JuggernautXL) | 1024×1024 | 20 | F16 backbone | ~1.4 it/s |
| Flux Dev FP8 | 512×512 | 10 | FP8 | ~0.6 it/s |
| Flux Schnell FP8 | 512×512 | 4 | FP8 | ~0.7 it/s |
| SD3.5 Medium fp8_scaled | 512×512 | 28 | FP8 | ~1.0 it/s |
| Z-Image Turbo | 512×512 | 8 | FP8 mix | ~0.5 it/s |

Numbers are rough; recapture them on your specific GPU before declaring a regression.

## Procedure

1. Run a ComfyUI workflow with the **same checkpoint, prompt, seed, scheduler, and step count**. Note steady-state it/s from the ComfyUI console.
2. Run the matching SharpInference test (e.g. `Sdxl_GenerateImage_Gpu`, `FluxDev_Generates_WithinSsimThreshold`). The pipeline emits per-step timing via the `onProgress` callback used by every test in this repo — see [`FluxGenerationTests`](../../tests/SharpInference.Diffusion.Tests/FluxGenerationTests.cs) and similar.
3. Compute it/s = `1000 / avg_step_ms` from steps 5..15.
4. Record peak VRAM with `nvidia-smi -l 1` running during the SharpInference run.

## Where the gap will show up

Per [`CUDA_PERFORMANCE.md`](CUDA_PERFORMANCE.md), the remaining ~18× gap to ComfyUI breaks down roughly as:

- ~2× — F16 tensor cores everywhere (mostly already on for Linear/MatMul; Conv2D im2col path can still go faster)
- ~1.5-2× — kernel fusion (GroupNorm+SiLU, Conv2D+bias+SiLU). Multiple kernel launches per ResNet block costs latency.
- ~2-3× — cuDNN Winograd for 3×3 Conv2D
- ~1.5× — memory pooling (eliminate per-op alloc/free)
- ~1.5× — FlashAttention-style tiled SDPA (ours uses a vanilla SDPA backend op)
- ~1.6× — native FP8 GEMM via cublasLtMatmul on Ada+ (already wired behind opt-in `EnableNativeFp8Gemm`; not testable on Ampere)

Stacking these conservatively gets to ~10-15× — the rest is probably driver overhead and our somewhat-naive scheduler/loop structure.

## When to file a regression

If a fresh measurement on the same GPU + same model is **>30% slower** than a previous documented number in a CUDA_PERFORMANCE.md baseline table, that's a regression. PRs should attach a before/after it/s pair.

## Cross-runtime parity check

For numerical (not perf) regression detection, the SSIM tests at [`SdxlSsimTests`](../../tests/SharpInference.Diffusion.Tests/SdxlSsimTests.cs) and [`FluxSsimTests`](../../tests/SharpInference.Diffusion.Tests/FluxSsimTests.cs) compare against diffusers reference images at matched noise. With reference noise loaded from `init_noise_seed42.bin`, the strict gate (0.85-0.90) catches any pipeline-level math drift.

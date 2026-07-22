# Step cache (C1/H1.4) — Qwen-Image warm A/B, RTX 4090

**Item:** INFERENCE_ACCEL_GRIND §H1.4 — across-step First-Block feature cache (`HARTSY_STEP_CACHE`),
device-resident gate (`stepcache.ptx`), reference wiring in Qwen-Image.

**Setup.** Qwen-Image Q4_K_M GGUF (QuantStack, sha256 `6454…bc98`, GPU per-GEMM dequant,
`CacheWeightCasts=false`), Qwen2.5-VL-7B TE, 16-ch VAE. 1024×1024, 20 steps, cfg 4.0, seed 42,
"A photograph of an astronaut riding a horse". Warm same-process A/B: one pipeline instance, knob
flipped between configs via in-process env (read per-`GenerateFromTokens`), 1 warmup + 3 trials/config.
Harness: `tests/HartsyInference.Diffusion.Tests/StepCacheQwenAbTests.cs`. Driver 580.159.03, CUDA 13.0,
process VRAM ≈ 15.3 GB (cache adds prevIndicator+residual per CFG stream — noise vs the resident DiT).
Raw CSV: [`2026-07-22_accel_stepcache_qwen_4090.csv`](2026-07-22_accel_stepcache_qwen_4090.csv).

## Results

| Config | Wall (3 trials) | Mean | Δ vs baseline | SSIM vs baseline | Computes/reuses per stream |
|---|---|---|---|---|---|
| baseline (unset) | 40.89 / 39.75 / 39.86 s | 40.17 s | — | 1.0 (byte-stable ×3 ✓) | 20/0 |
| `HARTSY_STEP_CACHE=0.1` | 35.19 / 35.10 / 35.12 s | 35.14 s | **−12.5% (1.14×)** | **0.9552** | 16/4 † |
| `HARTSY_STEP_CACHE=0.15` | 30.33 / 30.13 / 30.29 s | 30.25 s | −24.7% (1.33×) | 0.9189 | 14/6 |
| `HARTSY_STEP_CACHE=0.2` | 27.30 / 27.23 / 27.12 s | 27.22 s | −32.2% (1.48×) | 0.8744 | 12/8 |

† 0.1-run per-stream counts inferred from the reuse arithmetic (log window truncated); 0.15/0.2 counts
are from the pipeline's logged `Step cache:` lines. Cond and uncond streams gated identically in every
run (their drift indicators track).

Baseline matches the standing 39.4 s reference (warm trials 2–3: 39.75/39.86 s; trial 1 carries
~1 s of residual warm-in). Warmup was 54.2 s. **Cached runs are deterministic**: SSIM identical to 4
decimals across trials per config.

## Correctness gates (all BEFORE speed, per house rule)

- Kernel numerics: `StepCacheKernelTests` (3060) — CUDA vs host reduction rel-err 4.5e-7 (F32) /
  5.4e-7 (F16); identical-tensors → 0; `SupportsDeviceStepCacheGate=true`. 24 CPU tests stay green.
- Baseline byte-stability across 3 trials: **confirmed** (precondition for the SSIM comparisons).
- Eyeball (mandatory): 0.1 ≈ baseline (micro-texture drift only). 0.15 fully coherent, zero artifacts,
  but visible *detail drift* (chest-panel knobs change, tube vanishes) — a different-but-equal image.
  0.2 shows real simplification (instrument panel merges away, saddle-blanket texture fades).

## Decision

- **Shipped default (`HARTSY_STEP_CACHE=1`) = 0.10** — the knee that passes the SSIM ≥ 0.95 acceptance
  (0.9552, 1.14×). Changed in `QwenImagePipeline.ReadStepCacheThreshold` + `DeviceFeatureCache` ctor.
- 0.15 (1.33×) is *eyeball-clean* but fails the pinned gate — available as an explicit opt-in float;
  honest label: "quality-drifting, not quality-degrading, at 0.15 on this model".
- Negative-result note: measured 1.14× at the gate is *below* the literature's 1.4–2× headline. The
  plain accumulated-rel-L1 gate is the limiter (first-block drift is a blunt indicator early in the
  σ-schedule); the documented upgrade path is the TeaCache polynomial-rescaled gate
  (STEP_ACCELERATION §2.3, per-model coefficient fit) — expected to shift the SSIM-vs-speed curve, not
  this wiring. Revisit before replicating to video (H1.5) where the payoff is largest.

## Next (H1.5/H1.6)

Replicate wiring: Chroma (bypass its persistent CFG-pair step graph when armed), HiDream, then
Wan T2V/I2V + LTX-2.3 (TeaCache-class video results are 2–4.4×). Consider the polynomial gate before
the video ports.

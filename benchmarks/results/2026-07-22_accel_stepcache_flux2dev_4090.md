# Step cache on Flux.2 Dev (H1.5) — the biggest cache win in the fleet, RTX 4090

**Item:** INFERENCE_ACCEL_GRIND §H1.5 — First-Block cache on Flux.2 Dev (32B, Q4_K_S GGUF, the
engine catalog pin, sha-verified), poly gate selected by calibration per the fleet rule.

**Setup.** 1024², the production 50 steps, guidance 3.5 embedded (no CFG — ONE cache), seed 42, warm
same-process A/B **driven through the engine recipe** (`Flux2Recipe.Construct` →
`IRecipePipeline.Generate` — after the Ideogram harness lesson, the Mistral-tekken marker-splicing
conditioning is NOT re-implemented in the harness). Q4 GGUF keeps Dev's step graph off by default, so
baseline and armed runs share the eager topology. Harness: `StepCacheFlux2AbTests`. Wiring: original
FBCache shape — double-block-0's img stream gates; a hit replaces doubles 1..7 + concat + all 48
singles with block0Img + the cached img-portion residual (`Flux2Transformer.ForwardWithTemb`).

## Calibration (49 pairs, 50 steps) — a THIRD drift signature

Residual drift is **V-shaped**: 0.52 early → **0.05 mid-schedule valley** → 0.34 rising tail, and the
block-0 indicator TRACKS it (fit R²=0.70; poly `0.616388,-29.5305,456.02,-1890.27`). The late-window
gate would be wrong here (the tail is expensive again); the poly parks reuses in the valley. Fleet
picture: Qwen = informative indicator/monotonic (poly), Ideogram = flat indicator/falling drift
(window), Flux.2 = tracking indicator/V-shaped (poly, mid-schedule reuses).

## Baseline determinism (investigated per house rule)

The A/B flagged baseline NOT byte-stable — unique among the five models measured. Dedicated probe
(`Flux2Dev_Baseline_Stability`): pairwise SSIM 0.999992 / 0.999992 / **1.000000**, differing bytes
0.368% at **max delta ±1 LSB**, and only between the first post-warmup gen and later gens (b–c
byte-identical). One-time weight-promotion settling, not nondeterminism — the SSIM denominators are
valid. (VAE decode: full-res OOMs once on 24 GB beside the resident Q4 DiT → session-sticky tiled
fallback, identical for all runs.)

## Results (baseline 97.5 s warm ×3, eyeball-verified flawless on-prompt)

| Config | Wall | Speedup | SSIM | Reuses (of 50) | Eyeball |
|---|---|---|---|---|---|
| **poly@0.15** | **52.3 s** | **1.86×** | **0.9879 ✓** | 24 | **✓ indistinguishable** |
| **poly@0.25** | **39.1 s** | **2.49×** | **0.9581 ✓** | 31 | **✓ indistinguishable (micro-texture only)** |
| poly@0.4 | 33.6 s | 2.90× | 0.6666 | 34 | ✗ fails gate |

All configs deterministic across trials (SSIM identical to 4 decimals).

## Decision

- **Opt-in ship: `HARTSY_STEP_CACHE=0.25` + `HARTSY_STEP_CACHE_POLY="0.616388,-29.5305,456.02,-1890.27"`
  → 2.49× (97.5→39.1 s)** at the SSIM≥0.95 gate (0.9581 — the knee, same precedent as Qwen/Ideogram).
  Conservative: `0.15` → 1.86× at 0.9879, visually at the reproducibility noise floor.
- Why the fleet-best win: 50 undistilled steps + a deep mid-schedule drift valley = 31 skippable
  block-stacks that genuinely don't change the trajectory. This is the model the FBCache literature
  was written about, and the numbers land in its reported 2–2.5× band — at our stricter per-seed
  SSIM + eyeball gate.
- Fleet standings (all opt-in, all at SSIM≥0.95 + eyeball): **Flux.2 Dev 2.49× / Ideogram 1.39× /
  Qwen 1.20× / Krea2-Turbo 1.13×**; Wan honest-negative at per-seed identity.

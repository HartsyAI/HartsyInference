# Step cache (C1/H1.5) — Wan2.2 TI2V-5B T2V warm A/B, RTX 4090

**Item:** INFERENCE_ACCEL_GRIND §H1.5 — First-Block cache replicated to `WanVideoTransformer` /
`WanVideoPipeline` (plus `IBackend.PinActivation` so the cache's device-only state survives the video
pipelines' per-step `FreeActivations`).

**Setup.** Engine-verified config (CLI ground-truth anchored): 832×480, 33 frames, 50 steps, FlowShift 8,
cfg default, seed 42, engine-exact prompt embeds (full 512-row slice + `ZeroPaddedRows`). Warm
same-process, 1 warmup + 3 trials/config. Harness: `StepCacheWanAbTests`. Baselines byte-stable ✓ in
every run. CSVs: runs 4 (`0.1/0.15/0.2`) and 5 (`0.03/0.05/0.07`) in `Output/stepcache_wan_ab_2026*/`.

## Results (baseline ≈ 68 s warm)

| Threshold | Mean wall | Δ | SSIM mean / min (per-frame vs baseline) | Eyeball |
|---|---|---|---|---|
| 0.03 | 57.7 s | 1.18× | 0.882 / 0.873 | coherent, prompt-faithful, slightly shifted sample |
| 0.05 | 54.6 s | 1.25× | 0.859 / 0.826 | same character |
| 0.07 | 49.1 s | 1.39× | 0.752 / 0.722 | same character, larger drift |
| 0.10 | 47.4 s | 1.44× | 0.650 / 0.615 | different-but-coherent clip (verified frames) |
| 0.15 | 44.9 s | 1.52× | 0.773 / 0.735 † | different-but-coherent |
| 0.20 | 44.1 s | 1.55× | 0.717 / 0.683 † | different-but-coherent |

† SSIM is non-monotonic in threshold — different reuse patterns land in different basins, the signature
of trajectory divergence rather than graceful degradation.

## Verdict (negative result for the pinned gate; recorded per house rule)

1. **No threshold in the plain accumulated-rel-L1 family passes SSIM ≥ 0.95 on Wan at 50 steps** — even
   3% drift budget (0.03, ~24% reuse) lands at 0.88. Video trajectories under UniPC are chaotically
   sensitive: ANY reuse migrates the sample. This is qualitatively different from Qwen-Image (20 steps,
   Euler), where 0.10 held 0.955.
2. **Quality is NOT degraded** — every cached output eyeballs as a clean, prompt-faithful clip; what is
   lost is per-seed reproducibility. Honest framing for users: `HARTSY_STEP_CACHE` on Wan is a
   **"fast non-reproducible sampling" opt-in (1.2–1.55×)**, not a transparent accelerator. Default OFF.
3. Upgrade path: the polynomial-rescaled TeaCache gate (STEP_ACCELERATION §2.3, per-model coefficient
   fit) — worth one attempt, but the divergence mechanism above suggests identity preservation on
   long-schedule video may be fundamentally coarse; distribution-level metrics (FVD) rather than
   per-seed SSIM would be the honest acceptance test for any video step cache.
4. Mechanics all verified: byte-stable baselines, deterministic cached runs, PinActivation survival
   across per-step frees, MoE-boundary reset wired (untested — no A14B pair run yet).

## Pre-existing bugs found & fixed on the way (the standalone Wan test path)

- `Wan22Ti2V5B_*` tests run FlowShift 5 (config default) — engine uses 8. At 5 the output is broken.
- Tests slice prompt embeds to real-token length — engine slices the padded 512 rows and zeroes the
  pad (`VideoRecipeUtils.ZeroPaddedRows`); Wan cross-attends unmasked, so umT5 pad-row garbage drowns
  the prompt. Both produce dark-mush output that still PASSES the tests' frame-count asserts
  (their "NUMERIC OUTPUT VALIDATION-PENDING" labels were accurate). `StepCacheWanAbTests` now mirrors
  the engine exactly; the older test family should be migrated the same way.

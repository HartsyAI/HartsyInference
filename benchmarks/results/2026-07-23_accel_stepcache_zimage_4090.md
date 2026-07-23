# Step cache on Z-Image Turbo (H1.5) — NEGATIVE at the gate, RTX 4090

**Item:** INFERENCE_ACCEL_GRIND §H1.5 — the last flagship without a step-cache measurement. Wired
(`ZImageTransformer.ForwardPacked`/`PackedCore` optional cache on the main-layer loop, refiners always
run; armed cache forces eager — graph can't vary topology; fast-path-only in `ZImagePipeline`),
calibrated, A/B'd through the ENGINE recipe (`ZImageRecipe.Construct`, the Flux.2 harness pattern).

**Setup.** SwarmUI_Z-Image-Turbo-FP8Mix, 1024², the production 8 distilled guidance-free steps,
seed 42, warm same-process A/B ×3, 4090. Harness: `StepCacheZImageAbTests`.

## Calibration (14 pairs, 2 seeds): the drift floor is too high

Residual drift per step: 1.23 early → **floor 0.44–0.51 mid/late** (never lower). Compare the other
distilled 8-stepper Krea2-Turbo (floor 0.146, one safe reuse) — Z-Image's schedule is ~3× drifter
per step: there is **no quality-free skip anywhere in it**. Consistent with Z-Image already being the
fastest flagship (2.7 s warm through Swarm) — its 8 steps are all load-bearing.

## Results (baseline 4.15–4.30 s standalone-warm, byte-stable ×3)

| Config | Wall | Speedup | SSIM | Reuses (of 8) | Verdict |
|---|---|---|---|---|---|
| `LATE=0.5` + `0.15` | 4.22 s | 1.00× | 1.0000 | 0 | no-op (drift trips the gate every step) |
| `LATE=0.5` + `0.3` | 3.94 s | 1.07× | **0.9427 ✗** | 1 | eyeball near-identical but BELOW the 0.95 gate |

## Decision

- **No calibrated profile for Z-Image** — `HARTSY_STEP_CACHE=1` resolves to the generic raw 0.10,
  which produces 0 reuses on this drift scale (harmless no-op). Wiring stays (default-off, correct,
  and ready if a future longer-schedule Z-Image variant appears).
- Gate discipline note: 0.9427 at 1.07× is a near-miss with a clean eyeball, but the Qwen precedent
  (0.9189 eyeball-clean, rejected) holds — the bar does not move for a 7% win.
- Fleet H1.5 final standings (opt-in, SSIM≥0.95 + eyeball): **Flux.2 Dev 2.49× / Ideogram 1.39× /
  Qwen 1.20× / Krea2-Turbo 1.13× / Z-Image NEGATIVE / Wan NEGATIVE (per-seed identity)**. The
  pattern: wins scale with schedule length and drift-valley depth; efficient distilled schedules
  have nothing to skip.

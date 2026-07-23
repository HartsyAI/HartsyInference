# Step cache on Krea 2 Turbo (H1.5 replication) — the distilled-model case, RTX 4090

**Item:** INFERENCE_ACCEL_GRIND §H1.5 — First-Block cache on the 8-step distilled flagship, wired with
the late-window gate. Expected-negative going in (distilled steps each cover 2.5× the schedule of a
20-step model); the calibration + A/B measured the expectation instead of assuming it.

**Setup.** Krea2 Turbo fp8_scaled + Qwen3-VL-4B TE + Qwen-Image VAE, 1024², the production 8
guidance-free steps, seed 42, warm same-process A/B (`StepCacheKrea2AbTests`), 1 warmup + 3
trials/config, 4090. Wiring: optional `stepCache` through `Krea2Transformer.ForwardPatched` →
`ForwardCore` block loop (single unified stream, one cond cache — no CFG); **an armed cache forces
the eager path** (the captured step graph cannot replay per-step-variable topology), so the cached
configs compete against the graph-enabled baseline honestly.

## Calibration (14 pairs, 2 seeds)

Residual drift is **U-shaped**: 0.61 (step 2) → 0.28 (mid) → 0.38 (final step), with the indicator
spiking on the last step (0.28 vs ~0.11 mid). Unlike Ideogram (monotonic fall, flat indicator),
Krea2 Turbo's indicator IS informative at the tail — inside a late window the plain drift gate
correctly places the reuse mid-tail and recomputes the spiking final step.

## Results (baseline 4.43 s warm, byte-stable ×3, graph mode on)

| Config | Wall | Speedup | SSIM | Reuses (of 8 steps) | Eyeball |
|---|---|---|---|---|---|
| **`LATE=0.5` + `0.15`** | **3.91 s** | **1.13×** | **0.9740 ✓** | 1 | **✓ indistinguishable** |
| `LATE=0.25` + `0.15` | 4.43 s | 1.00× | 1.0000 | 0 | ✓ (null check: window too small ⇒ byte-identical) |

Deterministic across trials. The 0-reuse config reproducing baseline exactly is the null check that
the late-window plumbing itself is overhead-free.

## Decision

- **Opt-in ship: `HARTSY_STEP_CACHE=0.15` + `HARTSY_STEP_CACHE_LATE=0.5` → 1.13× (4.43→3.91 s)** at
  SSIM 0.974, eyeball-clean. Default stays OFF (flagship regression bar <6.5 s is comfortably met
  either way).
- Distilled-model verdict, honestly framed: **one** quality-free late skip exists at 8 steps — the
  ceiling is inherently low (each skipped step is 12.5% of the schedule), but it is not zero as
  assumed. The fleet H1.5 trio now spans the whole space: Qwen (poly, 1.20×), Ideogram (window,
  1.39×), Krea2 Turbo (window, 1.13×, distilled floor).

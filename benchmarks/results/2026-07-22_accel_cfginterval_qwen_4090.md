# Limited-interval CFG (C2/H2) — Qwen-Image warm A/B, RTX 4090

**Item:** INFERENCE_ACCEL_GRIND §H2 — `HARTSY_CFG_INTERVAL=lo,hi` (skip the uncond forward outside the
normalized-t band; arXiv:2404.07724), plus the §H2.2 composability run with the step cache.

**Setup.** Identical to the [step-cache A/B](2026-07-22_accel_stepcache_qwen_4090.md) (same model files,
1024², 20 steps, cfg 4, seed 42, warm same-process, 1 warmup + 3 trials/config). Harness:
`StepCacheQwenAbTests.QwenImage_CfgInterval_WarmAb_Gguf` + `QwenImage_CfgIntervalLateOnly_WarmAb_Gguf`.
CSVs: [`cfginterval`](2026-07-22_accel_cfginterval_qwen_4090.csv),
[`cfglate`](2026-07-22_accel_cfglate_qwen_4090.csv).

## Results — paper bands (run 1, baseline 40.07 s byte-stable ✓)

| Config | Mean wall | Δ | Uncond skips | SSIM | Eyeball |
|---|---|---|---|---|---|
| `0.1,0.85` | 33.85 s | −15.5% | 6/20 | **0.354** | **CATEGORY FLIP** — photo → flat stylized illustration |
| `0.15,0.9` | 33.81 s | −15.6% | 6/20 | 0.376 | same flip |
| compose: cache 0.1 + `0.1,0.85` | 27.62 s | −31.1% | 6/20 (+ cond 16c/4r, uncond 10c/4r) | 0.358 | inherits the flip |

## Results — late-only bands (run 2, baseline 40.03 s byte-stable ✓)

| Config | Mean wall | Δ | Uncond skips | SSIM | Eyeball |
|---|---|---|---|---|---|
| `0.05,1` | 38.73 s | −3.2% | 1/20 | 0.9961 | identical |
| `0.1,1` | 38.91 s | −2.8% | 1/20 | 0.9961 | identical |
| `0.15,1` | 37.92 s | −5.3% | 2/20 | 0.9812 | identical (verified) |

## Verdict (negative result recorded per house rule)

1. **The paper's bands are REJECTED for text-to-image on this model.** Skipping guidance on the early
   high-noise steps (normalized t > hi) is what saves most of the wall — and it abandons prompt-style
   adherence: "A photograph of…" renders as an illustration (SSIM 0.35, deterministic across trials).
   The paper's quality-improvement claim is distribution-level (ImageNet class-conditional FID), not
   per-prompt fidelity; it does not transfer to Qwen-Image at cfg 4 / 20 steps.
2. **Late-only bands are quality-safe but marginal** (−3…−5%): this scheduler's normalized-t only drops
   below 0.15 on the last 1–2 of 20 steps, so there is little to skip. `0.15,1` at SSIM 0.981 is a
   reasonable opt-in; anything reaching into early steps is not.
3. **Composability with the step cache works mechanically** (uncond cache self-heals as designed:
   10 computes / 4 reuses on 14 gated steps) — but compounding only makes sense on a quality-safe band.
4. C2 stays **default-off**. Knob semantics unchanged; per-model band tuning remains open for models
   with different σ-schedules (more low-t steps ⇒ bigger safe savings — check Wan/HiDream schedules
   before assuming this result generalizes).

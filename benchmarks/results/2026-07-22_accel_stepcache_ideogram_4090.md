# Step cache on Ideogram 4 (H1.5 replication) — late-window gate, RTX 4090

**Item:** INFERENCE_ACCEL_GRIND §H1.5 — First-Block cache replicated to Ideogram 4, plus the
`HARTSY_STEP_CACHE_LATE` schedule-window gate this model's calibration forced into existence.

**Setup.** Comfy-Org/Ideogram-4 fp8_scaled (both DiTs + Qwen3-VL-8B TE + Flux2 VAE, all
sha256-verified against HF), 1024², Default20 preset, seed 42, warm same-process A/B
(`StepCacheIdeogramAbTests`, both 9.3B DiTs resident, prompt/RoPE caches warm), 1 warmup + 3
trials/config, 4090 (driver 580.159.03, CUDA 13.0). Wiring: optional `stepCache` on
`Ideogram4Transformer.Forward`, **one cache instance per transformer** (cond and uncond are
*different* 9.3B models); regional plans excluded (per-step attention bias under a cached residual).

## Prerequisite: the standalone-harness bug (found and fixed this session)

The first calibration + A/B round produced a deterministic degenerate BASELINE (F16 face-tiles /
F32 gold-pebbles). Root cause: the standalone test family fed the raw `EncodeChat` output —
**padded to 2048 tokens with BOS + a `<think>` block** — into Ideogram's UNMASKED attention;
~2020 pad-token TE rows drowned the ~18 real prompt tokens. The engine recipe
(`Ideogram4RecipePipeline`) always trimmed the pad (`includeThinkBlock:false` + TrimRightPad) —
Swarm/production was never affected. Harnesses now mirror the engine; fixed baseline is an
on-prompt astronaut image at **19.5 s** warm (19.77/19.39/19.41, byte-stable ×3), matching the
19.2 s production/Swarm record. This bug also **retracts the morning's "comfy_quant fused-F16
defect" mechanism claim** — full correction in
[2026-07-22_accel_sageattn_3060.md](2026-07-22_accel_sageattn_3060.md). Same disease as the Wan
standalone pad-row bug: **standalone harnesses must copy the engine conditioning path exactly, and
the baseline gets eyeballed before any A/B is trusted.**

## Calibration (fixed harness, 76 pairs, 2 seeds, both streams)

Indicator (block-0) drift is **schedule-flat** (~0.09–0.13/step, slightly rising) while true
residual drift **falls 0.72 → 0.12** across the schedule (degree-3 fit R²=0.66, explosive
coefficients over the narrow x-range). Conclusion: on this arch the TeaCache-style polynomial
CANNOT place reuses correctly (the indicator carries almost no schedule information) — the damage
of a reuse is a function of *where in the schedule you are*, which the drift gate can't see.
Mechanism-level gate = **late window**: allow reuse only in the last fraction of steps, where the
residual-drift floor lives.

## Results (baseline 19.5 s, byte-stable, eyeball-verified on-prompt)

| Config | Wall | Speedup | SSIM | Reuses/stream (of 20) | Eyeball |
|---|---|---|---|---|---|
| full-schedule `0.15` | 10.6 s | 1.86× | 0.7267 | 10 | ✗ murky/dark, spurious figure |
| full-schedule `0.2` | 11.0 s | 1.79× | 0.6315 | 9–10 | ✗ dark |
| full-schedule `0.3` | 7.9 s | 2.47× | 0.6841 | 13 | ✗ ghost-dark |
| **`LATE=0.5` + `0.15`** | **15.0 s** | **1.31×** | **0.9653 ✓** | 5 | **✓ indistinguishable** |
| **`LATE=0.5` + `0.3`** | **14.1 s** | **1.39×** | **0.9530 ✓** | 6 | **✓ indistinguishable** |
| `LATE=0.3` + `0.15` | 16.8 s | 1.17× | 0.9770 ✓ | 3 | ✓ |

All configs deterministic (SSIM identical to 4 decimals across 3 trials). The late-window images
preserve full identity — same composition, same lighting, same (gibberish) rendered text as
baseline.

## Decision

- **Ship as opt-in** (`HARTSY_STEP_CACHE=<t>` + `HARTSY_STEP_CACHE_LATE=0.5`), default OFF.
  Recommended: `LATE=0.5` + `0.3` (1.39× at the SSIM≥0.95 gate — the knee, same precedent as
  Qwen's 0.9552 ship point); conservative alternative `LATE=0.5` + `0.15` (1.31× at 0.9653).
- The late-window knob is read by `StepCacheEnv.ReadLateWindow()` and currently wired in
  Ideogram 4 only. Fleet note: any model whose calibration shows a flat indicator + falling
  residual (run the observe-mode recipe) should get the same treatment — **check the indicator's
  schedule-informativeness before trusting a polynomial fit** (Qwen: informative, poly wins;
  Ideogram: uninformative, poly inverts, window wins; Wan: per-seed metric unresolved).
- Full-schedule raw gates are a documented NEGATIVE for Ideogram (quality-degrading darkening at
  every threshold tried, despite big speedups).

## Gate summary (per house rule, all before speed)

- Baseline byte-stability ×3 ✓; baseline eyeball on-prompt ✓ (post-fix).
- SSIM ≥ 0.95 at the recommended config ✓ (0.9530 / 0.9653).
- Eyeball of every passing config ✓ (identity preserved).
- Uncached path byte-identical (knob unset ⇒ no cache objects constructed) ✓.

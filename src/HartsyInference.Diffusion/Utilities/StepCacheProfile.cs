namespace HartsyInference.Diffusion.Utilities;

/// <summary>A model's CALIBRATED step-cache gate configuration — the measured ship point from its observe-mode calibration + warm A/B (INFERENCE_ACCEL_GRIND §H1.5; per-model results docs under benchmarks/results/2026-07-22_accel_stepcache_*.md). `HARTSY_STEP_CACHE=1` resolves to this profile when the pipeline supplies one (else the generic raw-0.10 default); an explicit numeric threshold keeps the profile's poly/late gates (the number is a budget in the profile's calibrated units), and the individual env knobs (`HARTSY_STEP_CACHE_POLY`/`_LATE`/`_CAP`) always override.</summary>
/// <param name="Threshold">Accumulated-drift budget at the model's SSIM≥0.95 knee (in calibrated units when <paramref name="Poly"/> is set, raw indicator units otherwise).</param>
/// <param name="Cap">Max consecutive cached steps.</param>
/// <param name="Poly">TeaCache-style calibration polynomial (c0 first), or null when the model's indicator is not schedule-informative (the poly would misplace reuses — use the window instead).</param>
/// <param name="LateWindow">Fraction of the schedule (from the END) where reuse is allowed; 0 = whole schedule. The gate for flat-indicator/falling-drift models (Ideogram) and distilled few-step models (Krea2 Turbo).</param>
public sealed record StepCacheProfile(float Threshold, int Cap, float[]? Poly, float LateWindow);

namespace HartsyInference.Diffusion.Utilities;

/// <summary>Env-knob parsing for the across-step feature cache (`HARTSY_STEP_CACHE`, `HARTSY_STEP_CACHE_CAP`) —
/// shared by every pipeline that wires <see cref="DeviceFeatureCache"/> (Qwen-Image was the reference wiring;
/// Wan-Video and later fleet ports read the same knobs). Malformed values THROW: a silently-ignored perf knob
/// would invalidate an A/B run.</summary>
public static class StepCacheEnv
{
    /// <summary>Reads HARTSY_STEP_CACHE: unset/0 = off (0f); "1"/"true" = the 0.10 default drift threshold
    /// (the measured SSIM-≥0.95 knee on Qwen-Image — benchmarks/results/2026-07-22_accel_stepcache_qwen_4090.md);
    /// any other non-negative float = that threshold.</summary>
    public static float ReadThreshold()
    {
        string? value = Environment.GetEnvironmentVariable("HARTSY_STEP_CACHE");
        if (string.IsNullOrWhiteSpace(value)) return 0f;
        if (value == "0") return 0f;
        if (value == "1" || value.Equals("true", StringComparison.OrdinalIgnoreCase)) return 0.10f;
        if (!float.TryParse(value, System.Globalization.CultureInfo.InvariantCulture, out float threshold) || threshold < 0f)
            throw new ArgumentException($"HARTSY_STEP_CACHE must be a non-negative float or 1/true; got '{value}'.");
        return threshold;
    }

    /// <summary>Resolves the effective step-cache configuration against a model's calibrated
    /// <see cref="StepCacheProfile"/>. Off (unset/0) → threshold 0. "1"/"true" → the profile verbatim
    /// (or the generic raw-0.10 default when the pipeline has none). An explicit numeric threshold keeps
    /// the profile's poly/late/cap — the number is a budget in the profile's calibrated units. The
    /// individual env knobs (HARTSY_STEP_CACHE_POLY / _LATE / _CAP) always override their component,
    /// so A/B harnesses and users can probe alternatives without code changes.</summary>
    public static (float Threshold, int Cap, float[]? Poly, float LateWindow) Resolve(StepCacheProfile? profile)
    {
        string? value = Environment.GetEnvironmentVariable("HARTSY_STEP_CACHE");
        bool isDefaultOn = value == "1" || (value?.Equals("true", StringComparison.OrdinalIgnoreCase) ?? false);
        float threshold = isDefaultOn && profile is not null ? profile.Threshold : ReadThreshold();
        if (threshold <= 0f) return (0f, 0, null, 0f);

        int cap = Environment.GetEnvironmentVariable("HARTSY_STEP_CACHE_CAP") is { Length: > 0 }
            ? ReadCap()
            : profile?.Cap ?? ReadCap();
        float[]? poly = Environment.GetEnvironmentVariable("HARTSY_STEP_CACHE_POLY") is { Length: > 0 }
            ? ReadPoly()
            : profile?.Poly;
        float late = Environment.GetEnvironmentVariable("HARTSY_STEP_CACHE_LATE") is { Length: > 0 }
            ? ReadLateWindow()
            : profile?.LateWindow ?? 0f;
        return (threshold, cap, poly, late);
    }

    /// <summary>Reads HARTSY_STEP_CACHE_POLY ("c0,c1,c2,..." — TeaCache-style gate calibration polynomial,
    /// lowest power first) and HARTSY_STEP_CACHE_CALIB (a CSV path: run uncached and log indicator→residual
    /// drift pairs for fitting). Null when unset; malformed poly THROWS.</summary>
    public static float[]? ReadPoly()
    {
        string? v = Environment.GetEnvironmentVariable("HARTSY_STEP_CACHE_POLY");
        if (string.IsNullOrWhiteSpace(v)) return null;
        string[] parts = v.Split(',');
        float[] c = new float[parts.Length];
        for (int i = 0; i < parts.Length; i++)
            if (!float.TryParse(parts[i], System.Globalization.CultureInfo.InvariantCulture, out c[i]))
                throw new ArgumentException($"HARTSY_STEP_CACHE_POLY malformed at '{parts[i]}'.");
        return c;
    }

    public static string? ReadCalibFile() => Environment.GetEnvironmentVariable("HARTSY_STEP_CACHE_CALIB");

    /// <summary>Reads HARTSY_STEP_CACHE_LATE: fraction of the schedule (measured from the END) where cache
    /// reuse is allowed; steps before the window run uncached. 0 (default) = whole schedule. For models whose
    /// first-block indicator is schedule-flat while true residual drift falls steeply late (Ideogram 4
    /// calibration, 2026-07-22), this is the honest way to place reuses where they are cheap — the drift gate
    /// alone cannot see it.</summary>
    public static float ReadLateWindow()
    {
        string? value = Environment.GetEnvironmentVariable("HARTSY_STEP_CACHE_LATE");
        if (string.IsNullOrWhiteSpace(value)) return 0f;
        if (!float.TryParse(value, System.Globalization.CultureInfo.InvariantCulture, out float late)
            || late < 0f || late > 1f)
            throw new ArgumentException($"HARTSY_STEP_CACHE_LATE must be a float in [0,1]; got '{value}'.");
        return late;
    }

    /// <summary>Reads HARTSY_STEP_CACHE_CAP (max consecutive cached steps), default 3.</summary>
    public static int ReadCap()
    {
        string? value = Environment.GetEnvironmentVariable("HARTSY_STEP_CACHE_CAP");
        if (string.IsNullOrWhiteSpace(value)) return 3;
        if (!int.TryParse(value, out int cap) || cap < 1)
            throw new ArgumentException($"HARTSY_STEP_CACHE_CAP must be a positive integer; got '{value}'.");
        return cap;
    }
}

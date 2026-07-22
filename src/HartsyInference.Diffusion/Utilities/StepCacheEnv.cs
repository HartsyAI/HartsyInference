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

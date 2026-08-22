using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.MemoryManagement;

namespace HartsyInference.Video.Pipelines;

/// <summary>Resolves whether the Wan-Animate-2 driving cache is stored in BF16 (half the bytes, a divergence from
/// the F32 reference forward) or F32 (exact numerics, 2x the bytes — 18.3 GiB at 480x800/61f). The env var
/// <see cref="EnvironmentVariable"/> is the explicit override; unset means <b>auto</b>: the global low-VRAM policy
/// is honoured first, then free VRAM is measured and F32 is kept only where its cache plus the activation reserve
/// still fits beside a streamed DiT. Backends that report no VRAM (CPU) resolve to F32 so parity tests keep exact
/// numerics.</summary>
/// <remarks>Modelled on <see cref="LowVramPolicy"/> (three-state, warn-and-auto on garbage, log dedup) but not a
/// reuse of it: <c>on</c> here means BF16 — a precision trade, not a streaming trade — and the auto branch needs
/// the geometry's byte demands, which the global policy never sees.</remarks>
public static class WanAnimate2DrivingCachePolicy
{
    /// <summary>The env var: <c>1</c>/<c>true</c>/<c>on</c> = BF16, <c>0</c>/<c>false</c>/<c>off</c> = F32,
    /// <c>auto</c>/unset = decide from the low-VRAM policy and measured free VRAM.</summary>
    public const string EnvironmentVariable = "HARTSY_ANIMATE2_BF16_DRIVING_CACHE";

    /// <summary>The last env value logged, so a stable setting announces itself once per process.</summary>
    private static string? _lastLogged;

    /// <summary>Resolves the cache dtype for one generation: true = BF16, false = F32. Trims and queries the
    /// backend only when the decision actually needs a measurement; <paramref name="measuredFreeBytes"/> lets a
    /// caller that already trimmed and queried (the recipe's pre-flight) share its measurement.</summary>
    /// <param name="f32CacheBytes">Exact cache bytes at this geometry in F32 (<c>DrivingCacheBytes(grid, false)</c>).</param>
    /// <param name="bf16CacheBytes">Same in BF16, for the log line.</param>
    /// <param name="activationReserveBytes">What the chunk will reserve for activations and workspace
    /// (<see cref="WanAnimate2Pipeline.ActivationReserveBytes"/>).</param>
    /// <param name="streamedWeightFloorBytes">Weights that must sit on the device even fully streamed: shared
    /// weights plus the prefetch window.</param>
    public static bool Resolve(IBackend backend, long f32CacheBytes, long bf16CacheBytes,
        long activationReserveBytes, long streamedWeightFloorBytes, long? measuredFreeBytes = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        string? env = Environment.GetEnvironmentVariable(EnvironmentVariable);
        bool? forced = ParseEnv(env, ShouldLog(env));
        LowVramMode mode = forced is null ? LowVramPolicy.Resolve(backend) : default;
        long freeBytes = 0;
        if (forced is null && mode == LowVramMode.Auto)
        {
            if (measuredFreeBytes is long measured)
            {
                freeBytes = measured;
            }
            else
            {
                // Pooled frees under-report until trimmed (VramPlanner.TrimBeforeQuery) — and an under-report here
                // silently costs cache precision, not just speed.
                backend.TrimMemoryPool();
                (freeBytes, _) = backend.GetVramInfo();
            }
        }
        long f32DemandBytes = f32CacheBytes + activationReserveBytes + streamedWeightFloorBytes;
        bool bf16 = ResolveCore(forced, mode, freeBytes, f32DemandBytes, out string decidedBy);
        Logs.Info($"[WanAnimate2] Driving cache dtype: {(bf16 ? "BF16" : "F32")} ({decidedBy}) — "
            + $"F32 {Gib(f32CacheBytes)} / BF16 {Gib(bf16CacheBytes)} cache; F32 needs {Gib(f32DemandBytes)} "
            + $"(+{Gib(activationReserveBytes)} activations, +{Gib(streamedWeightFloorBytes)} streamed-weight floor)"
            + (decidedBy == "measured" ? $" vs {Gib(freeBytes)} free." : "."));
        return bf16;
    }

    /// <summary>The decision itself, free of I/O so tests can feed a fake free-VRAM value. Order: env override,
    /// then the global low-VRAM policy, then the measurement — <paramref name="freeBytes"/> ≤ 0 (backends that
    /// don't report VRAM, e.g. CPU) resolves to F32 for exact numerics.</summary>
    internal static bool ResolveCore(bool? envForced, LowVramMode globalMode, long freeBytes, long f32DemandBytes,
        out string decidedBy)
    {
        if (envForced is bool forced)
        {
            decidedBy = $"{EnvironmentVariable} override";
            return forced;
        }
        if (globalMode == LowVramMode.ForceOn)
        {
            decidedBy = $"{LowVramPolicy.EnvironmentVariable}=on";
            return true;
        }
        if (globalMode == LowVramMode.ForceOff)
        {
            decidedBy = $"{LowVramPolicy.EnvironmentVariable}=off";
            return false;
        }
        if (freeBytes <= 0)
        {
            decidedBy = "no VRAM report";
            return false;
        }
        decidedBy = "measured";
        return f32DemandBytes > freeBytes;
    }

    /// <summary>Parses the env value: null = auto (decide from policy + measurement). Accepts the historical
    /// boolean tokens unchanged, so existing machine configs keep their meaning.</summary>
    internal static bool? ParseEnv(string? value, bool log)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return null;
        }
        string normalized = value.Trim();
        if (normalized.Equals("auto", StringComparison.OrdinalIgnoreCase))
        {
            return null;
        }
        if (normalized == "1" || normalized.Equals("true", StringComparison.OrdinalIgnoreCase)
            || normalized.Equals("on", StringComparison.OrdinalIgnoreCase))
        {
            if (log) Logs.Info($"[WanAnimate2] {EnvironmentVariable}={value} — driving cache pinned to BF16.");
            return true;
        }
        if (normalized == "0" || normalized.Equals("false", StringComparison.OrdinalIgnoreCase)
            || normalized.Equals("off", StringComparison.OrdinalIgnoreCase))
        {
            if (log) Logs.Info($"[WanAnimate2] {EnvironmentVariable}={value} — driving cache pinned to F32.");
            return false;
        }
        if (log) Logs.Warning($"[WanAnimate2] Unrecognized {EnvironmentVariable}='{value}' — using auto. "
            + "Valid: auto, on/1/true (BF16), off/0/false (F32).");
        return null;
    }

    /// <summary>True when this env value differs from the last one logged.</summary>
    private static bool ShouldLog(string? value)
    {
        string normalized = value ?? "";
        if (string.Equals(_lastLogged, normalized, StringComparison.Ordinal))
        {
            return false;
        }
        _lastLogged = normalized;
        return true;
    }

    private static string Gib(long bytes) => $"{bytes / (1024.0 * 1024.0 * 1024.0):F2} GiB";
}

using System.Runtime.CompilerServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;

namespace HartsyInference.Core.MemoryManagement;

/// <summary>How aggressively the engine should trade speed for VRAM headroom.</summary>
public enum LowVramMode
{
    /// <summary>Measure free VRAM and stream/evict only when the fully-resident layout would not fit.</summary>
    Auto = 0,

    /// <summary>Stream even when everything would fit — for testing the streamed path, and for sharing a card with another process.</summary>
    ForceOn = 1,

    /// <summary>Never stream and never auto-evict: preload everything and let an oversized model fail. The power-user escape hatch.</summary>
    ForceOff = 2,
}

/// <summary>Resolves <see cref="LowVramMode"/> from the <c>HARTSY_LOWVRAM</c> environment variable.</summary>
/// <remarks>Deliberately three-state rather than the boolean <c>EnvSwitch</c> shape used by the proven-profile
/// features: "measure and decide" is a genuinely different behavior from "always stream", and conflating them
/// would leave no way to exercise the streamed path on a card with headroom. <c>ForceOff</c> exists because
/// operators who size their own workloads want an oversized request to fail loudly rather than silently run at
/// streaming speed.</remarks>
public static class LowVramPolicy
{
    /// <summary>The environment variable read by <see cref="Resolve()"/>.</summary>
    public const string EnvironmentVariable = "HARTSY_LOWVRAM";

    /// <summary>The last value logged, so a stable setting announces itself once instead of every phase.</summary>
    private static string? _lastLogged;

    /// <summary>Per-backend overrides, so two engines in one process (one per GPU) can run different policies. Weak
    /// keys: a disposed backend's entry vanishes with it, no unregistration required on the failure paths.</summary>
    private static readonly ConditionalWeakTable<IBackend, ModeBox> _overrides = new();

    /// <summary>Pins the policy for <paramref name="backend"/>; wins over the environment variable in
    /// <see cref="Resolve(IBackend?)"/>. Hosts with a per-backend low-VRAM setting (the SwarmUI extension) use this
    /// instead of the env var, whose process-wide last-writer-wins semantics broke multi-backend setups.</summary>
    public static void SetOverride(IBackend backend, LowVramMode mode)
    {
        ArgumentNullException.ThrowIfNull(backend);
        _overrides.AddOrUpdate(backend, new ModeBox(mode));
    }

    /// <summary>Removes <paramref name="backend"/>'s override so it falls back to the environment variable.</summary>
    public static void ClearOverride(IBackend backend)
    {
        ArgumentNullException.ThrowIfNull(backend);
        _overrides.Remove(backend);
    }

    /// <summary>The mode governing <paramref name="backend"/>: its override when one is set, else the process-wide
    /// environment resolution. Null backend = environment only.</summary>
    public static LowVramMode Resolve(IBackend? backend)
    {
        if (backend is not null && _overrides.TryGetValue(backend, out ModeBox? box))
        {
            return box.Mode;
        }
        return Resolve();
    }

    private sealed class ModeBox
    {
        public ModeBox(LowVramMode mode) => Mode = mode;

        public LowVramMode Mode { get; }
    }

    /// <summary>The current mode: unset/unrecognized → <see cref="LowVramMode.Auto"/>; <c>1</c>/<c>on</c>/<c>true</c>
    /// → <see cref="LowVramMode.ForceOn"/>; <c>0</c>/<c>off</c>/<c>false</c> → <see cref="LowVramMode.ForceOff"/>.</summary>
    /// <remarks>Deliberately re-read every call rather than cached. A host may set the variable after the process has
    /// already resolved it once — the SwarmUI backend writes it during backend init, which can land after an earlier
    /// backend or a warm-up generation has run — and a cached first answer would silently ignore that, making the
    /// setting appear to do nothing. This is read once per generation phase (not per step), so an environment lookup
    /// is free relative to the weight uploads it governs. Only the log line is de-duplicated.</remarks>
    public static LowVramMode Resolve()
    {
        string? value = Environment.GetEnvironmentVariable(EnvironmentVariable);
        return Parse(value, ShouldLog(value));
    }

    /// <summary>True when this value differs from the last one logged — keeps a stable setting to one log line.</summary>
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

    /// <summary>Clears the log-dedup state. Tests only.</summary>
    internal static void ResetCacheForTests() => _lastLogged = null;

    private static LowVramMode Parse(string? value, bool log)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return LowVramMode.Auto;
        }
        string normalized = value.Trim();
        if (normalized.Equals("auto", StringComparison.OrdinalIgnoreCase))
        {
            return LowVramMode.Auto;
        }
        if (normalized == "1" || normalized.Equals("on", StringComparison.OrdinalIgnoreCase)
            || normalized.Equals("true", StringComparison.OrdinalIgnoreCase))
        {
            if (log) Logs.Info($"[VRAM] {EnvironmentVariable}={value} — forcing the streamed layout even where the resident one would fit.");
            return LowVramMode.ForceOn;
        }
        if (normalized == "0" || normalized.Equals("off", StringComparison.OrdinalIgnoreCase)
            || normalized.Equals("false", StringComparison.OrdinalIgnoreCase))
        {
            if (log) Logs.Info($"[VRAM] {EnvironmentVariable}={value} — low-VRAM handling disabled; oversized models will fail rather than stream.");
            return LowVramMode.ForceOff;
        }
        if (log) Logs.Warning($"[VRAM] Unrecognized {EnvironmentVariable}='{value}' — using auto. Valid: auto, on/1/true, off/0/false.");
        return LowVramMode.Auto;
    }
}

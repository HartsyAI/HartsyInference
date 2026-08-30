using System.Collections.Concurrent;

namespace HartsyInference.Core.Configuration;

/// <summary>Resolves a knob's value: scoped profile → explicit override → declared default.</summary>
/// <remarks>The engine no longer reads its configuration from the process environment. Settings come from
/// <see cref="KnobFile"/>, from <see cref="Set{T}"/> (what a host or extension calls), or from a scoped
/// <see cref="KnobProfile"/> for one generation.
/// <para>The environment was removed rather than kept as a lowest-precedence fallback because a fallback is what
/// let the old surface rot: ~210 names accumulated in six mutually inconsistent value grammars, several
/// documented as working that nothing read, and a doc that was always one commit stale. A second source of truth
/// re-creates that pressure no matter how tidy the first one is.
/// <para>Removing it silently would have been its own trap, so <see cref="ReportStaleEnvironmentVariables"/>
/// names any legacy variable still exported and points at the setting that replaced it.</para></remarks>
public static class KnobStore
{
    private static readonly ConcurrentDictionary<string, object?> _overrides = new(StringComparer.Ordinal);

    /// <summary>Sets an explicit value. Beats the settings file; beaten by a scoped profile.</summary>
    public static void Set<T>(Knob<T> knob, T value) => _overrides[knob.Id] = value;

    /// <summary>Sets by dotted id with an already-parsed value. Used by <see cref="KnobFile"/>, which owns the parsing.</summary>
    internal static void SetByIdRaw(string id, object? value) => _overrides[id] = value;

    /// <summary>Clears an override so the knob falls back to its declared default.</summary>
    public static void Clear<T>(Knob<T> knob) => _overrides.TryRemove(knob.Id, out _);

    /// <summary>Drops every override. For tests and <see cref="KnobFile.Reload"/>.</summary>
    public static void ResetOverrides() => _overrides.Clear();

    public static bool HasOverride<T>(Knob<T> knob) => _overrides.ContainsKey(knob.Id);

    internal static T Resolve<T>(Knob<T> knob)
    {
        // The settings file is discovered rather than injected, because the SwarmUI extension is loaded as a
        // library and never gets a Main to call a loader from.
        KnobFile.EnsureLoaded();
        if (KnobProfileScope.Current is { } profile
            && profile.Values.TryGetValue(knob.Id, out object? scoped) && scoped is T scopedTyped)
        {
            return Coerce(knob, scopedTyped);
        }
        if (_overrides.TryGetValue(knob.Id, out object? o) && o is T typed)
        {
            return Coerce(knob, typed);
        }
        return knob.Default;
    }

    private static string? ReadEnvironmentVariable(string variable) => Environment.GetEnvironmentVariable(variable);

    /// <summary>Applies the knob's range rule to a supplied value. The declared default is trusted as already valid.</summary>
    /// <remarks>Load-bearing for settings that arrive from a file or <c>--set</c>: without it
    /// <c>numerics.gemvWpb = 17</c> would reach a kernel that expects 1..16, and <c>vram.audioEvictBelowGb = -5</c>
    /// would be accepted where the original call site rejected it.</remarks>
    private static T Coerce<T>(Knob<T> knob, T value)
        => knob.Coerce is null ? value : knob.Coerce(value);

    /// <summary>Legacy environment variables that are still exported but are no longer read, paired with the setting that replaced each.</summary>
    /// <remarks>A machine that exported <c>HARTSY_LTX2_TWO_STAGE=1</c> for months would otherwise just quietly
    /// stop two-stage sampling. Hosts call this at startup and log the result.</remarks>
    public static IReadOnlyList<(string Variable, string Setting)> ReportStaleEnvironmentVariables()
    {
        List<(string, string)> stale = [];
        foreach (object knob in KnobRegistry.All)
        {
            (string id, string? legacy, _, _, _, _, _) = KnobRegistry.Describe(knob);
            if (legacy is null || id.StartsWith("test.", StringComparison.Ordinal))
            {
                continue;
            }
            if (!string.IsNullOrEmpty(ReadEnvironmentVariable(legacy)))
            {
                stale.Add((legacy, id));
            }
        }
        stale.Sort((a, b) => string.CompareOrdinal(a.Item1, b.Item1));
        return stale;
    }
}

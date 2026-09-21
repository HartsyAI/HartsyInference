using System.Collections.Concurrent;

namespace HartsyInference.Core.Configuration;

/// <summary>Resolves a knob's value: scoped profile → explicit override → declared default.</summary>
/// <remarks>The engine no longer reads its configuration from the process environment. Settings come from
/// <see cref="KnobFile"/>, from <see cref="Set{T}"/> (what a host or extension calls), or from a scoped
/// <see cref="KnobProfile"/> for one generation.
/// <para>The environment was removed rather than kept as a lowest-precedence fallback because a fallback is what
/// let the old surface rot: ~210 names accumulated in six mutually inconsistent value grammars, several
/// documented as working that nothing read, and a doc that was always one commit stale. A second source of truth
/// re-creates that pressure no matter how tidy the first one is.</para></remarks>
public static class KnobStore
{
    /// <summary>Each override with the layer that set it, so "where did this value come from" is recorded rather than guessed.</summary>
    private static readonly ConcurrentDictionary<string, (object? Value, string Source)> _overrides = new(StringComparer.Ordinal);

    /// <summary>Sets an explicit value. Beats the settings file; beaten by a scoped profile.</summary>
    public static void Set<T>(Knob<T> knob, T value) => _overrides[knob.Id] = (value, "host");

    /// <summary>Sets by dotted id with an already-parsed value. Used by <see cref="KnobFile"/>, which owns the parsing.</summary>
    internal static void SetByIdRaw(string id, object? value, string source) => _overrides[id] = (value, source);

    /// <summary>Which layer supplies <paramref name="id"/>'s current value: a per-request profile, the settings file or a host, else the declared default.</summary>
    /// <remarks>Recorded when the override is set. Inferring it instead cannot work: a host override and a file
    /// value land in the same dictionary, so after the file supplies a value a later host Set is invisible.</remarks>
    public static string SourceOf(string id)
    {
        if (KnobProfileScope.Current is { } profile && profile.Values.ContainsKey(id))
        {
            return "request profile";
        }
        KnobFile.EnsureLoaded();
        return _overrides.TryGetValue(id, out (object? Value, string Source) entry) ? entry.Source : "default";
    }

    /// <summary>The value currently overriding <paramref name="id"/>, or null when nothing does. Used by <see cref="KnobFile.Save"/> to persist exactly what the loader parsed.</summary>
    internal static object? Raw(string id) => _overrides.TryGetValue(id, out (object? Value, string Source) entry) ? entry.Value : null;

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
            && profile.Values.TryGetValue(knob.Id, out object? scoped))
        {
            // A pinned null means retain the caller's contextual default, not inherit a machine override.
            if (scoped is null && default(T) is null)
                return default!;
            if (scoped is T scopedTyped)
                return Coerce(knob, scopedTyped);
        }
        if (_overrides.TryGetValue(knob.Id, out (object? Value, string Source) entry) && entry.Value is T typed)
        {
            return Coerce(knob, typed);
        }
        return knob.Default;
    }


    /// <summary>Applies the knob's range rule to a supplied value. The declared default is trusted as already valid.</summary>
    /// <remarks>Load-bearing for settings that arrive from a file or <c>--set</c>: without it
    /// <c>numerics.gemvWpb = 17</c> would reach a kernel that expects 1..16, and <c>vram.audioEvictBelowGb = -5</c>
    /// would be accepted where the original call site rejected it.</remarks>
    private static T Coerce<T>(Knob<T> knob, T value)
        => knob.Coerce is null ? value : knob.Coerce(value);

}

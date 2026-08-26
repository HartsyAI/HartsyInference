using System.Collections.Concurrent;
using System.Globalization;

namespace HartsyInference.Core.Configuration;

/// <summary>How a boolean knob's legacy environment spelling maps to a value. Preserved per knob because the two grammars genuinely disagree.</summary>
/// <remarks><see cref="Exact"/> covers the historic <c>== "1"</c> and <c>!= "0"</c> call sites: only the exact
/// opposite-of-default spelling flips it, anything else is the default — so <c>HARTSY_X=false</c> on a
/// default-ON <c>!= "0"</c> knob resolves to <b>true</b>, which is what that code did.
/// <see cref="TriState"/> is the <c>EnvSwitch.IsEnabled</c> convention, where <c>false</c> also means false.
/// Unifying them is deliberately deferred: C2 changes where knobs are declared, not what they mean.</remarks>
public enum BoolGrammar
{
    /// <summary>Only the exact opposite-of-default spelling (<c>"1"</c> or <c>"0"</c>) flips it; every other value is the default.</summary>
    Exact,

    /// <summary><c>1</c>/<c>true</c> → true, <c>0</c>/<c>false</c> → false, anything else → default.</summary>
    TriState,
}

/// <summary>The single place the engine reads process environment. Resolution order is override → legacy environment → declared default.</summary>
/// <remarks>Every other file gets its settings from a <see cref="Knob{T}"/>, so the C1 allowlist can be driven to
/// this one entry and stay monotonic. When C7 retires the legacy names, the environment lookup here is deleted and
/// the allowlist reaches zero.
/// <para>Deliberately does not cache: call sites that want a value bound once already hold it in a
/// <c>static readonly</c> field, so caching here would only add a second, staler layer.</para></remarks>
public static class KnobStore
{
    private static readonly ConcurrentDictionary<string, object?> _overrides = new(StringComparer.Ordinal);

    /// <summary>Reads the legacy environment name. The only environment access in the engine.</summary>
    private static string? ReadLegacy(string? name)
        => string.IsNullOrEmpty(name) ? null : Environment.GetEnvironmentVariable(name);

    /// <summary>Sets an explicit value, beating the environment. Used by CLI <c>--set</c> and the API in C4.</summary>
    public static void Set<T>(Knob<T> knob, T value) => _overrides[knob.Id] = value;

    /// <summary>Clears an override so the knob falls back to environment then default.</summary>
    public static void Clear<T>(Knob<T> knob) => _overrides.TryRemove(knob.Id, out _);

    /// <summary>Drops every override. For tests.</summary>
    public static void ResetOverrides() => _overrides.Clear();

    public static bool HasOverride<T>(Knob<T> knob) => _overrides.ContainsKey(knob.Id);

    internal static T Resolve<T>(Knob<T> knob)
    {
        if (_overrides.TryGetValue(knob.Id, out object? o) && o is T typed)
        {
            return typed;
        }
        string? raw = ReadLegacy(knob.LegacyEnv);
        if (raw is null)
        {
            return knob.Default;
        }
        // Set-but-empty is NOT unset for a string knob. Presence-only call sites test `is null`, so folding ""
        // into the default would make `HARTSY_MUSICGEN_GRAPH_OFF=` stop disabling graph decode. Consumers that
        // want empty to mean absent apply their own IsNullOrEmpty, as DebugDumpSink does.
        if (raw.Length == 0 && typeof(T) != typeof(string))
        {
            return knob.Default;
        }
        return Parse(knob, raw);
    }

    /// <remarks>Dispatches on the DECLARED type, not the default's runtime type: an override knob declares
    /// <c>bool?</c> with a null default, and switching on the value would send every one of them to the string arm.</remarks>
    private static T Parse<T>(Knob<T> knob, string raw)
    {
        Type t = typeof(T);
        object? parsed;
        if (t == typeof(bool))
        {
            parsed = ParseBool(raw, (bool)(object)knob.Default!, knob.Grammar);
        }
        else if (t == typeof(bool?))
        {
            // No opinion when unrecognized, so the call site keeps its contextual default.
            parsed = ParseBoolOverride(raw);
        }
        else if (t == typeof(int) || t == typeof(int?))
        {
            parsed = int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out int v) ? v : null;
        }
        else if (t == typeof(long) || t == typeof(long?))
        {
            parsed = long.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out long v) ? v : null;
        }
        else if (t == typeof(float) || t == typeof(float?))
        {
            parsed = float.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out float v) ? v : null;
        }
        else
        {
            parsed = raw;
        }
        if (parsed is not T typed)
        {
            return knob.Default;
        }
        return knob.Coerce is null ? typed : knob.Coerce(typed);
    }

    /// <summary>Tri-state override: only a recognized spelling takes a position, anything else defers to the caller.</summary>
    private static bool? ParseBoolOverride(string raw)
    {
        if (raw == "1" || raw.Equals("true", StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }
        if (raw == "0" || raw.Equals("false", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }
        return null;
    }

    private static bool ParseBool(string raw, bool defaultValue, BoolGrammar grammar)
    {
        if (grammar == BoolGrammar.TriState)
        {
            if (raw == "1" || raw.Equals("true", StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
            if (raw == "0" || raw.Equals("false", StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }
            return defaultValue;
        }
        // Exact: only the opposite-of-default spelling flips it.
        return defaultValue ? raw != "0" : raw == "1";
    }
}

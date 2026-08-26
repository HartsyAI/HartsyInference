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
        if (string.IsNullOrEmpty(raw))
        {
            return knob.Default;
        }
        return Parse(knob, raw);
    }

    private static T Parse<T>(Knob<T> knob, string raw)
    {
        object? parsed = knob.Default switch
        {
            bool d => ParseBool(raw, d, knob.Grammar),
            int d => int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out int v) ? v : d,
            long d => long.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out long v) ? v : d,
            float d => float.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out float v) ? v : d,
            _ => raw,
        };
        // A string knob declared with a null default lands in the default arm above, which already yields raw.
        return parsed is T typed ? typed : knob.Default;
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

using System.Collections.Immutable;

namespace HartsyInference.Core.Configuration;

/// <summary>A named set of knob values that can be pushed for the duration of one generation.</summary>
/// <remarks>Built with <see cref="Create"/> and <see cref="With{T}"/>, which are typed per knob, so a profile
/// cannot pin a bool knob to a string. Values are stored by knob id and resolved through <see cref="KnobStore"/>.
/// <para>Deliberately not a record with one field per knob. There are 210 knobs; a record would need editing
/// every time one is added, and a profile that pins four of them would still have to state the other 206.</para></remarks>
public sealed class KnobProfile
{
    private KnobProfile(string name, ImmutableDictionary<string, object?> values)
    {
        Name = name;
        Values = values;
    }

    /// <summary>Human-readable name, used in logs and by CLI <c>--profile</c>.</summary>
    public string Name { get; }

    internal ImmutableDictionary<string, object?> Values { get; }

    public int Count => Values.Count;

    /// <summary>Starts an empty profile.</summary>
    public static KnobProfile Create(string name)
        => new(name, ImmutableDictionary<string, object?>.Empty.WithComparers(StringComparer.Ordinal));

    /// <summary>Pins one knob. Typed, so the value must match the knob's declared type.</summary>
    public KnobProfile With<T>(Knob<T> knob, T value)
        => new(Name, Values.SetItem(knob.Id, value));

    /// <summary>Pins a knob by dotted id, for CLI <c>--set</c> where the id arrives as text.</summary>
    /// <remarks>Returns false rather than throwing on an unknown id or an unparsable value, so the caller can
    /// report the operator's typo with the list of valid ids instead of a stack trace.</remarks>
    public bool TrySet(string id, string rawValue, out KnobProfile updated, out string? error)
    {
        updated = this;
        object? knob = KnobRegistry.Find(id);
        if (knob is null)
        {
            error = $"Unknown setting '{id}'.";
            return false;
        }
        Type declared = knob.GetType().GetGenericArguments()[0];
        if (!TryParseAs(declared, rawValue, out object? parsed, out error))
        {
            return false;
        }
        updated = new KnobProfile(Name, Values.SetItem(id, parsed));
        return true;
    }

    private static bool TryParseAs(Type declared, string raw, out object? parsed, out string? error)
    {
        parsed = null;
        error = null;
        Type t = Nullable.GetUnderlyingType(declared) ?? declared;
        if (t == typeof(string))
        {
            parsed = raw;
            return true;
        }
        if (t == typeof(bool))
        {
            // Accepts both grammars' spellings, since at this layer the operator is being explicit.
            if (raw is "1" or "0" || bool.TryParse(raw, out _))
            {
                parsed = raw is "1" || (bool.TryParse(raw, out bool b) && b);
                return true;
            }
            error = $"Expected a boolean (1/0/true/false), got '{raw}'.";
            return false;
        }
        if (t == typeof(int) && int.TryParse(raw, out int i)) { parsed = i; return true; }
        if (t == typeof(long) && long.TryParse(raw, out long l)) { parsed = l; return true; }
        if (t == typeof(float) && float.TryParse(raw, System.Globalization.NumberStyles.Float,
            System.Globalization.CultureInfo.InvariantCulture, out float f)) { parsed = f; return true; }
        error = $"Expected {t.Name}, got '{raw}'.";
        return false;
    }

    /// <summary>Makes this profile current until the returned scope is disposed.</summary>
    public IDisposable Push() => KnobProfileScope.Push(this);

    public override string ToString() => $"{Name} ({Values.Count} setting(s))";
}

/// <summary>The knob profile in force for the generation running on this async flow.</summary>
/// <remarks>Mirrors <c>VramPolicyScope</c>, and for the same reason: a request-level override that only reached
/// construction would silently do nothing for knobs read during the generation, because pipelines are cached.
/// <para><see cref="AsyncLocal{T}"/> rather than a static, so two engines generating concurrently on two devices
/// cannot have one request's profile decide the other's numerics.</para></remarks>
public static class KnobProfileScope
{
    private static readonly AsyncLocal<KnobProfile?> _current = new();

    /// <summary>The current profile, or null outside a scoped generation.</summary>
    public static KnobProfile? Current => _current.Value;

    /// <summary>Pushes <paramref name="profile"/> until disposed. A null profile pushes nothing, so an unscoped request costs no behavior change.</summary>
    public static IDisposable Push(KnobProfile? profile) => new Scope(profile);

    private sealed class Scope : IDisposable
    {
        private readonly KnobProfile? _previous;
        private bool _disposed;

        internal Scope(KnobProfile? profile)
        {
            _previous = _current.Value;
            _current.Value = profile ?? _previous;
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }
            _disposed = true;
            _current.Value = _previous;
        }
    }
}

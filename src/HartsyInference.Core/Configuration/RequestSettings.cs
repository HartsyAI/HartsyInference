namespace HartsyInference.Core.Configuration;

/// <summary>Per-request engine settings, as they arrive over the API.</summary>
/// <remarks>The wire shape deliberately mirrors the CLI: a named <see cref="Profile"/> applied first, then
/// individual <see cref="Set"/> entries on top, so <c>--profile reference --set numerics.sageAttn=1</c> and its
/// JSON equivalent mean the same thing.
/// <code>
/// { "settings": { "profile": "reference", "set": { "numerics.ditF16": "0" } } }
/// </code>
/// <para>Values are strings because the wire has no types; <see cref="KnobProfile.TrySet"/> parses each against
/// the knob's declared type and reports the operator's typo rather than throwing.</para></remarks>
public sealed record RequestSettings
{
    /// <summary>Named profile to apply first, e.g. <c>reference</c>. Null applies none.</summary>
    public string? Profile { get; init; }

    /// <summary>Individual overrides by dotted knob id, applied after <see cref="Profile"/>.</summary>
    public IReadOnlyDictionary<string, string>? Set { get; init; }

    /// <summary>True when this carries nothing, so a request without settings costs no scope push.</summary>
    public bool IsEmpty => string.IsNullOrWhiteSpace(Profile) && Set is not { Count: > 0 };

    /// <summary>Resolves to a profile, or null when empty. Throws <see cref="ArgumentException"/> naming the bad id or value.</summary>
    public KnobProfile? Resolve()
    {
        if (IsEmpty)
        {
            return null;
        }
        KnobProfile profile = KnobProfiles.Default;
        if (!string.IsNullOrWhiteSpace(Profile))
        {
            profile = KnobProfiles.ByName(Profile)
                ?? throw new ArgumentException(
                    $"Unknown settings profile '{Profile}'. Known profiles: {string.Join(", ", KnobProfiles.Names)}.");
        }
        if (Set is not { Count: > 0 })
        {
            return profile;
        }
        foreach ((string id, string value) in Set)
        {
            if (!profile.TrySet(id, value, out KnobProfile updated, out string? error))
            {
                throw new ArgumentException(error);
            }
            profile = updated;
        }
        return profile;
    }
}

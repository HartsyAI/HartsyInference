namespace HartsyInference.Core.Configuration;

/// <summary>Per-request engine settings, as they arrive over the API.</summary>
/// <remarks>The wire shape deliberately mirrors the CLI: a named <see cref="Profile"/> applied first, then
/// individual <see cref="Set"/> entries on top, so <c>--profile reference --set numerics.sageAttn=1</c> and its
/// JSON equivalent mean the same thing.
/// <code>
/// { "settings": { "profile": "reference", "set": { "numerics.ditF16": "0" } } }
/// </code>
/// <para>Values are strings because the wire has no types; <see cref="KnobProfile.TrySet"/> parses each against
/// the knob's declared type and reports the operator's typo rather than throwing.</para>
/// <para>⚠️ <b>Per-request settings reach per-call knobs only.</b> Anything bound while the engine, backend or
/// pipeline is built is already fixed by the time a request arrives on a long-lived server — including the
/// backend's TF32, F16-GEMM and native-FP8 decisions, which are assigned in <c>CudaBackend</c>'s constructor.
/// Those need process-level configuration instead: the CLI's <c>--profile</c> (pushed before the engine is
/// constructed) or the server's own startup settings. <see cref="UnreachablePerRequest"/> names the ones a
/// request cannot move, so the caller can refuse rather than silently ignore them.</para></remarks>
public sealed record RequestSettings
{
    /// <summary>Named profile to apply first, e.g. <c>reference</c>. Null applies none.</summary>
    public string? Profile { get; init; }

    /// <summary>Individual overrides by dotted knob id, applied after <see cref="Profile"/>.</summary>
    public IReadOnlyDictionary<string, string>? Set { get; init; }

    /// <summary>True when this carries nothing, so a request without settings costs no scope push.</summary>
    public bool IsEmpty => string.IsNullOrWhiteSpace(Profile) && Set is not { Count: > 0 };

    /// <summary>Ids in <paramref name="profile"/> whose knob binds when the engine or a model is built, so a per-request pin cannot take effect.</summary>
    /// <remarks>Every <see cref="KnobScope.Construction"/> id is reported, whatever its value: the value the engine was built
    /// with is not recorded, and the live knob value changes on a runtime settings write without a rebuild, so a comparison
    /// against it can accept a pin that changes nothing and refuse one that matches the built configuration.
    /// <para>Measured, not theorised: an API generation with <c>profile: reference</c> returned a byte-identical image to
    /// one without it, because the GEMM knobs it pins are read once in <c>CudaBackend</c>'s constructor. Those knobs are
    /// declared <see cref="KnobScope.Construction"/> for that reason. The CLI does not have this problem because it
    /// pushes the profile in <c>Main</c>, before the engine is constructed.</para></remarks>
    public static IReadOnlyList<string> UnreachablePerRequest(KnobProfile profile)
    {
        List<string> unreachable = [];
        foreach ((string id, _) in profile.Values)
        {
            if (KnobRegistry.Find(id) is { } knob && KnobRegistry.Describe(knob).Scope == KnobScope.Construction)
            {
                unreachable.Add(id);
            }
        }
        unreachable.Sort(StringComparer.Ordinal);
        return unreachable;
    }

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
        if (Set is { Count: > 0 })
        {
            foreach ((string id, string value) in Set)
            {
                if (!profile.TrySet(id, value, out KnobProfile updated, out string? error))
                {
                    throw new ArgumentException(error);
                }
                profile = updated;
            }
        }
        // Rejected rather than ignored: these bind while the engine and pipeline are built, so on a long-lived
        // server they are already fixed when the request arrives. Accepting them would report success and change
        // nothing, which is the failure mode this whole surface exists to remove. The named profile is checked
        // too, not only the explicit entries: a profile that pins a construction knob is just as unreachable.
        IReadOnlyList<string> unreachable = UnreachablePerRequest(profile);
        if (unreachable.Count > 0)
        {
            throw new ArgumentException(
                $"These settings bind when the engine or a model is built and cannot be changed per request: "
                + $"{string.Join(", ", unreachable)}. Configure them where the server starts, or use the CLI's "
                + "--set, which applies before the engine is constructed.");
        }
        return profile;
    }
}

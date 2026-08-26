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

    /// <summary>Ids in <paramref name="profile"/> that bind before a request exists, so setting them per-request cannot take effect.</summary>
    /// <remarks>Two kinds, and the second is why this check exists at all. Knobs declared
    /// <see cref="KnobScope.Construction"/> enter a model at load, and the pipeline is cached. But several knobs
    /// declared <see cref="KnobScope.Runtime"/> are also unreachable per-request because their consumer assigns
    /// them once — a <c>static readonly</c> field bound at type-init, or a backend field assigned in the
    /// constructor. <c>_allowTf32</c>, <c>_gemmFast16</c> and <c>EnableNativeFp8Gemm</c> are all set in
    /// <c>CudaBackend</c>'s constructor, so on a long-lived server they are fixed before the first request.
    /// <para>Measured, not theorised: an API generation with <c>profile: reference</c> returned a byte-identical
    /// image to one without it. The CLI does not have this problem because it pushes the profile in <c>Main</c>,
    /// before the engine is constructed.</para></remarks>
    public static IReadOnlyList<string> UnreachablePerRequest(KnobProfile profile)
    {
        List<string> unreachable = [];
        foreach (string id in profile.Values.Keys)
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
        if (Set is not { Count: > 0 })
        {
            return profile;
        }
        List<string> unreachable = [];
        foreach ((string id, string value) in Set)
        {
            if (!profile.TrySet(id, value, out KnobProfile updated, out string? error))
            {
                throw new ArgumentException(error);
            }
            if (KnobRegistry.Find(id) is { } k && KnobRegistry.Describe(k).Scope == KnobScope.Construction)
            {
                unreachable.Add(id);
            }
            profile = updated;
        }
        // Rejected rather than ignored: these bind while the engine and pipeline are built, so on a long-lived
        // server they are already fixed when the request arrives. Accepting them would report success and change
        // nothing, which is the failure mode this whole surface exists to remove.
        if (unreachable.Count > 0)
        {
            throw new ArgumentException(
                $"These settings bind when a model is loaded and cannot be changed per request: "
                + $"{string.Join(", ", unreachable)}. Configure them where the server starts, or use the CLI's "
                + "--set, which applies before the engine is constructed.");
        }
        return profile;
    }
}

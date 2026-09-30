namespace HartsyInference.Core.Backends;

/// <summary>Routing recipe for <see cref="IBackend.MoeRoute"/>; the fields mirror the HF gate variants the host router covers.</summary>
/// <param name="NumExperts">Router width E (at most <see cref="MaxExperts"/>).</param>
/// <param name="TopK">Experts kept per token.</param>
/// <param name="Scoring">Logit to score function.</param>
/// <param name="GroupCount">Expert groups for node-limited routing; 0 or 1 disables group limiting.</param>
/// <param name="GroupsKept">Groups whose experts stay eligible (group score is the sum of its top-2 selection scores).</param>
/// <param name="MaskedGroupValue">Selection score given to experts of dropped groups; 0 reproduces HF <c>masked_fill(0)</c>.</param>
/// <param name="Renormalize">Divide the gathered weights by their sum plus <paramref name="RenormEpsilon"/>.</param>
/// <param name="RenormEpsilon">0 for the legacy flat routers, 1e-20 for grouped and V4.1 gates.</param>
/// <param name="Scale">Routed scaling factor applied after renormalisation.</param>
/// <param name="LogitDivisor">Logits are divided by this before scoring (V4.1 <c>gate_temp</c>); 1 is a no-op.</param>
public readonly record struct MoeRouteArgs(
    int NumExperts, int TopK, MoeRouteScoring Scoring, int GroupCount = 0, int GroupsKept = 0,
    float MaskedGroupValue = 0f, bool Renormalize = false, float RenormEpsilon = 0f, float Scale = 1f,
    float LogitDivisor = 1f)
{
    /// <summary>Largest router width the fused kernel keeps in shared memory.</summary>
    public const int MaxExperts = 1024;

    /// <summary>True when experts are partitioned and only the best groups stay eligible.</summary>
    public bool IsGrouped => GroupCount > 1;
}

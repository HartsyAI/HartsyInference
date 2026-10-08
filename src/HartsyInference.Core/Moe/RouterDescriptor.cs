using HartsyInference.Core.Backends;

namespace HartsyInference.Core.Moe;

/// <summary>
/// Semantic router: scoring, group limiting, top-k per phase and weight normalization. Lowers to a
/// <see cref="MoeRouteArgs"/> backend fast path whenever the router fits <see cref="MoeRouteArgs.MaxExperts"/>.
/// </summary>
/// <param name="NumExperts">Routed experts this router selects among.</param>
/// <param name="TopKDecode">Experts kept per token during decode.</param>
/// <param name="TopKPrefill">Experts kept per token during prefill; equal to <paramref name="TopKDecode"/> unless the model differs.</param>
/// <param name="Scoring">Logit to score function.</param>
/// <param name="GroupCount">Expert groups for node-limited routing; 0 or 1 disables group limiting.</param>
/// <param name="GroupsKept">Groups that stay eligible when group limiting is on.</param>
/// <param name="Renormalize">Divide the gathered weights by their sum.</param>
/// <param name="RenormEpsilon">Epsilon added to the sum (0 for flat routers, 1e-20 for grouped and V4.1 gates).</param>
/// <param name="Scale">Routed scaling factor after renormalization; must be finite.</param>
/// <param name="LogitDivisor">Logits are divided by this before scoring (V4.1 gate temperature); 1 is a no-op.</param>
/// <param name="HasSelectionBias">A learned per-expert bias affects selection only, not the weights.</param>
/// <param name="HasTokenKindBias">A second bias is selected per token kind (V4.1 image-span tokens).</param>
/// <param name="BiasSpace">Where the selection bias is added. <see cref="SelectionBiasSpace.Logit"/> needs a flat router with a bias.</param>
public sealed record RouterDescriptor(
    int NumExperts, int TopKDecode, int TopKPrefill, MoeRouteScoring Scoring,
    int GroupCount = 0, int GroupsKept = 0, bool Renormalize = false, float RenormEpsilon = 0f,
    float Scale = 1f, float LogitDivisor = 1f, bool HasSelectionBias = false, bool HasTokenKindBias = false,
    SelectionBiasSpace BiasSpace = SelectionBiasSpace.Score)
{
    /// <summary>Validates the router and returns it.</summary>
    /// <exception cref="ArgumentException">The router is internally inconsistent.</exception>
    public RouterDescriptor Validated()
    {
        if (NumExperts <= 0) throw new ArgumentException("A router needs at least one expert.", nameof(NumExperts));
        if (TopKDecode <= 0 || TopKDecode > NumExperts) throw new ArgumentException($"Decode top-k {TopKDecode} must be in [1, {NumExperts}].", nameof(TopKDecode));
        if (TopKPrefill <= 0 || TopKPrefill > NumExperts) throw new ArgumentException($"Prefill top-k {TopKPrefill} must be in [1, {NumExperts}].", nameof(TopKPrefill));
        if (GroupCount == 1) throw new ArgumentException("One expert group is not a grouping; use GroupCount 0 for flat routing.", nameof(GroupCount));
        if (GroupCount > 1)
        {
            if (NumExperts % GroupCount != 0) throw new ArgumentException($"{GroupCount} groups do not divide {NumExperts} experts.", nameof(GroupCount));
            int groupSize = NumExperts / GroupCount;
            if (groupSize < 2) throw new ArgumentException($"Groups of {groupSize} expert(s) cannot be group-limited; the backend needs at least 2 per group.", nameof(GroupCount));
            if (GroupsKept <= 0 || GroupsKept > GroupCount) throw new ArgumentException($"Groups kept {GroupsKept} must be in [1, {GroupCount}].", nameof(GroupsKept));
            long eligible = (long)groupSize * GroupsKept;
            if (TopKDecode > eligible || TopKPrefill > eligible)
                throw new ArgumentException($"Top-k cannot exceed the {eligible} experts in the kept groups.", nameof(TopKDecode));
        }
        if (RenormEpsilon < 0f || !float.IsFinite(RenormEpsilon)) throw new ArgumentOutOfRangeException(nameof(RenormEpsilon));
        if (!float.IsFinite(Scale)) throw new ArgumentOutOfRangeException(nameof(Scale), Scale, "The routed scale must be finite.");
        if (!(LogitDivisor > 0f) || !float.IsFinite(LogitDivisor)) throw new ArgumentOutOfRangeException(nameof(LogitDivisor));
        if (BiasSpace == SelectionBiasSpace.Logit && (!HasSelectionBias || GroupCount > 1))
            throw new ArgumentException("Logit-space selection needs a flat router with a selection bias.", nameof(BiasSpace));
        return this;
    }

    /// <summary>
    /// True when the backend routing kernel can run this router: its width fits the fused kernel and selection happens on
    /// the score. Logit-space selection has no backend form yet, so such a router stays on the host.
    /// </summary>
    public bool CanLowerToBackend => NumExperts <= MoeRouteArgs.MaxExperts && BiasSpace == SelectionBiasSpace.Score;

    /// <remarks>
    /// Every scoring function produces non-negative scores (softmax, sigmoid, sqrt-softplus), so the masked-group value
    /// of 0 always ranks below a kept group. A future scoring function that can go negative must revisit this.
    /// </remarks>
    /// <summary>Backend routing arguments for the given phase.</summary>
    /// <exception cref="InvalidOperationException">The router is wider than the backend kernel supports; routing stays on the host.</exception>
    public MoeRouteArgs ToRouteArgs(bool prefill)
    {
        if (!CanLowerToBackend) throw new InvalidOperationException($"{NumExperts} experts exceed the backend router limit of {MoeRouteArgs.MaxExperts}.");
        return new MoeRouteArgs(NumExperts, prefill ? TopKPrefill : TopKDecode, Scoring, GroupCount, GroupsKept,
            MaskedGroupValue: 0f, Renormalize: Renormalize, RenormEpsilon: RenormEpsilon, Scale: Scale, LogitDivisor: LogitDivisor);
    }
}

using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Moe;

/// <summary>
/// Capabilities the runtime discovers from a topology, so generic code asks "does this topology have X" and never
/// checks a model name.
/// </summary>
public sealed record SparseCapabilities(
    bool HasSparseExperts,
    bool HasSharedExperts,
    bool HasDenseLayers,
    bool HasMixedDenseAndSparseLayers,
    bool VariableExpertsPerLayer,
    bool HeterogeneousExpertShapes,
    bool VariableTopK,
    bool PhaseDependentRouting,
    bool HasGroupRouting,
    bool HasTokenKindRouting,
    bool HasDraftLayers,
    bool HasClampedPrograms,
    bool RequiresHostRouting,
    IReadOnlySet<SequenceStateKind> StateKinds,
    IReadOnlySet<DType> ExpertDTypes,
    IReadOnlySet<MoeRouteScoring> ScoringKinds)
{
    /// <summary>Derives the capability set from a topology.</summary>
    public static SparseCapabilities From(SparseModelTopology topology)
    {
        ArgumentNullException.ThrowIfNull(topology);
        IReadOnlyList<SparseLayerDescriptor> layers = topology.Layers;
        List<MoeLayerDescriptor> sparse = layers.Where(static l => l.Moe is not null).Select(static l => l.Moe!).ToList();
        bool anySparse = sparse.Count > 0;
        bool anyDense = layers.Any(static l => l.Moe is null);

        HashSet<SequenceStateKind> states = new(layers.Select(static l => l.StateKind));
        HashSet<DType> dtypes = new();
        HashSet<MoeRouteScoring> scoring = new();
        foreach (MoeLayerDescriptor moe in sparse)
        {
            dtypes.Add(moe.Routed.Shape.WeightDType);
            if (moe.Shared is not null) dtypes.Add(moe.Shared.Shape.WeightDType);
            foreach (KeyValuePair<int, ExpertDescriptor> pair in moe.Routed.Overrides ?? new Dictionary<int, ExpertDescriptor>()) dtypes.Add(pair.Value.WeightDType);
            scoring.Add(moe.Router.Scoring);
        }

        int distinctCounts = sparse.Select(static m => m.ExpertCount).Distinct().Count();
        bool heterogeneous = sparse.Any(static m => m.Routed.Overrides is { Count: > 0 })
            || sparse.Select(static m => m.Routed.Shape.IntermediateSize).Distinct().Count() > 1;
        bool variableTopK = sparse.Select(static m => m.Router.TopKDecode).Distinct().Count() > 1;

        return new SparseCapabilities(
            HasSparseExperts: anySparse,
            HasSharedExperts: sparse.Any(static m => m.Shared is not null),
            HasDenseLayers: anyDense,
            HasMixedDenseAndSparseLayers: anyDense && anySparse,
            VariableExpertsPerLayer: distinctCounts > 1,
            HeterogeneousExpertShapes: heterogeneous,
            VariableTopK: variableTopK,
            PhaseDependentRouting: sparse.Any(static m => m.Router.TopKPrefill != m.Router.TopKDecode),
            HasGroupRouting: sparse.Any(static m => m.Router.GroupCount > 1),
            HasTokenKindRouting: sparse.Any(static m => m.Router.HasTokenKindBias),
            HasDraftLayers: layers.Any(static l => l.IsDraft),
            HasClampedPrograms: sparse.Any(static m => m.Program.IsClamped),
            RequiresHostRouting: sparse.Any(static m => !m.Router.CanLowerToBackend),
            StateKinds: states,
            ExpertDTypes: dtypes,
            ScoringKinds: scoring);
    }
}

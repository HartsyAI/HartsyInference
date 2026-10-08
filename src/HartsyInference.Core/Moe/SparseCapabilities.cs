using System.Collections.Frozen;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Moe;

/// <summary>
/// Capabilities the runtime discovers from a topology, so generic code asks "does this topology have X" and never
/// checks a model name. Every flag is defined in its own comment and is computed only from the topology.
/// </summary>
/// <param name="HasSparseExperts">At least one layer is sparse.</param>
/// <param name="HasSharedExperts">At least one sparse layer has a shared expert group.</param>
/// <param name="HasDenseLayers">At least one layer is dense.</param>
/// <param name="HasMixedDenseAndSparseLayers">Dense and sparse layers both occur.</param>
/// <param name="VariableExpertsPerLayer">Sparse layers do not all have the same routed-expert count.</param>
/// <param name="HeterogeneousExpertShapes">Experts differ in intermediate width, dtype or layout, across routed and shared groups including overrides.</param>
/// <param name="VariableTopK">The decode or the prefill top-k differs between sparse layers.</param>
/// <param name="PhaseDependentRouting">Some sparse layer keeps a different top-k in prefill than in decode.</param>
/// <param name="HasGroupRouting">Some router applies node-limited group selection.</param>
/// <param name="HasTokenKindRouting">Some router selects a bias per token kind.</param>
/// <param name="HasDraftLayers">At least one layer is a draft (speculative) layer.</param>
/// <param name="HasClampedPrograms">Some expert program clamps a branch.</param>
/// <param name="RequiresHostRouting">Some router is wider than the backend routing kernel supports.</param>
/// <param name="StateKinds">Sequence-state kinds used by any layer.</param>
/// <param name="ExpertDTypes">Storage dtypes used by any expert.</param>
/// <param name="ScoringKinds">Router scoring functions used.</param>
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

        HashSet<SequenceStateKind> states = new();
        foreach (SparseLayerDescriptor layer in layers)
            foreach (SequenceStateKind kind in Enum.GetValues<SequenceStateKind>())
                if (kind != SequenceStateKind.None && layer.StateKind.HasFlag(kind)) states.Add(kind);
        HashSet<DType> dtypes = new();
        HashSet<MoeRouteScoring> scoring = new();
        HashSet<(int Intermediate, DType DType, ExpertWeightLayout Layout)> shapeSignatures = new();
        foreach (MoeLayerDescriptor moe in sparse)
        {
            scoring.Add(moe.Router.Scoring);
            foreach (ExpertGroupDescriptor group in moe.Shared is null ? [moe.Routed] : new[] { moe.Routed, moe.Shared })
            {
                AddShape(group.Shape);
                if (group.Overrides is not null)
                    foreach (KeyValuePair<int, ExpertDescriptor> pair in group.Overrides) AddShape(pair.Value);
            }
            void AddShape(ExpertDescriptor shape)
            {
                dtypes.Add(shape.WeightDType);
                shapeSignatures.Add((shape.IntermediateSize, shape.WeightDType, shape.Layout));
            }
        }

        HashSet<int> decodeTopK = new(sparse.Select(static m => m.Router.TopKDecode));
        HashSet<int> prefillTopK = new(sparse.Select(static m => m.Router.TopKPrefill));

        return new SparseCapabilities(
            HasSparseExperts: anySparse,
            HasSharedExperts: sparse.Any(static m => m.Shared is not null),
            HasDenseLayers: anyDense,
            HasMixedDenseAndSparseLayers: anyDense && anySparse,
            VariableExpertsPerLayer: sparse.Select(static m => m.ExpertCount).Distinct().Count() > 1,
            HeterogeneousExpertShapes: shapeSignatures.Count > 1,
            VariableTopK: decodeTopK.Count > 1 || prefillTopK.Count > 1,
            PhaseDependentRouting: sparse.Any(static m => m.Router.TopKPrefill != m.Router.TopKDecode),
            HasGroupRouting: sparse.Any(static m => m.Router.GroupCount > 1),
            HasTokenKindRouting: sparse.Any(static m => m.Router.HasTokenKindBias),
            HasDraftLayers: layers.Any(static l => l.IsDraft),
            HasClampedPrograms: sparse.Any(static m => m.Program.IsClamped),
            RequiresHostRouting: sparse.Any(static m => !m.Router.CanLowerToBackend),
            StateKinds: states.ToFrozenSet(),
            ExpertDTypes: dtypes.ToFrozenSet(),
            ScoringKinds: scoring.ToFrozenSet());
    }
}

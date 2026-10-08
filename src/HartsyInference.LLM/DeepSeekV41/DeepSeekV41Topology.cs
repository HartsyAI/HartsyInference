using HartsyInference.Core.Backends;
using HartsyInference.Core.Moe;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>Describes a DeepSeek-V4.1 checkpoint with the generic sparse topology, from its parsed config.</summary>
/// <remarks>
/// Target backbone layers are verified against the host loader: every layer routes through a gate with a required
/// <c>gate.bias</c>, one shared expert of width <c>moe_intermediate_size</c>, and the clamped SwiGLU. Draft (MTP/DSpark)
/// layers are described from their expert counts only. Their gate-bias and shared-expert layout is not loaded yet, so they
/// carry no selection bias and no shared expert. They do carry the backbone's clamped SwiGLU: <c>swiglu_limit</c> is a
/// model-wide config key, but the draft path is not verified yet and must be checked when it lands.
/// </remarks>
public static class DeepSeekV41Topology
{
    /// <summary>Builds the topology for <paramref name="config"/>.</summary>
    /// <param name="config">Parsed V4.1 configuration.</param>
    /// <param name="expertDType">Storage type of routed experts; the official checkpoint ships MXFP4 (E2M1 with E8M0 block scales).</param>
    /// <exception cref="ArgumentException">The config's scoring function is not one the router knows.</exception>
    public static SparseModelTopology Build(DeepSeekV41Config config, DType? expertDType = null)
    {
        ArgumentNullException.ThrowIfNull(config);
        DType dtype = expertDType ?? DType.F4E2M1;
        MoeRouteScoring scoring = ParseScoring(config.ScoringFunc);
        ExpertProgram program = config.SwigluLimit > 0 ? ExpertProgram.SwigluClamped((float)config.SwigluLimit) : ExpertProgram.Swiglu;
        List<SparseLayerDescriptor> layers = new List<SparseLayerDescriptor>(config.TotalLayerCount);
        for (int i = 0; i < config.TotalLayerCount; i++)
        {
            DeepSeekV41LayerPlan plan = config.LayerPlans[i];
            SequenceStateKind state = plan.CompressRatio > 0 ? SequenceStateKind.CompressedKv : SequenceStateKind.SlidingWindowKv;
            layers.Add(plan.IsDraft ? DraftLayer(i, config, scoring, dtype, program, state) : TargetLayer(i, config, scoring, dtype, program, state));
        }
        return new SparseModelTopology(config.HiddenSize, layers, "deepseek-v4.1");
    }

    private static SparseLayerDescriptor TargetLayer(int index, DeepSeekV41Config config, MoeRouteScoring scoring, DType dtype,
        ExpertProgram program, SequenceStateKind state)
    {
        RouterDescriptor router = new RouterDescriptor(
            NumExperts: config.NRoutedExperts,
            TopKDecode: config.NumExpertsPerTok,
            TopKPrefill: config.NumExpertsPerTok,
            Scoring: scoring,
            Renormalize: config.NormTopkProb && config.NumExpertsPerTok > 1,
            RenormEpsilon: 1e-20f,
            Scale: (float)config.RoutedScalingFactor,
            HasSelectionBias: true,
            HasTokenKindBias: config.Vision is not null).Validated();
        ExpertGroupDescriptor routed = new ExpertGroupDescriptor(config.NRoutedExperts, new ExpertDescriptor(config.HiddenSize, config.MoeIntermediateSize, dtype)).Validated();
        ExpertGroupDescriptor shared = new ExpertGroupDescriptor(config.NSharedExperts, new ExpertDescriptor(config.HiddenSize, config.MoeIntermediateSize, dtype)).Validated();
        return new SparseLayerDescriptor(index, new MoeLayerDescriptor(router, routed, shared, SharedIsGated: false, program).Validated(), state);
    }

    private static SparseLayerDescriptor DraftLayer(int index, DeepSeekV41Config config, MoeRouteScoring scoring, DType dtype,
        ExpertProgram program, SequenceStateKind state)
    {
        RouterDescriptor router = new RouterDescriptor(
            NumExperts: config.DsparkNRoutedExperts,
            TopKDecode: config.DsparkNumExpertsPerTok,
            TopKPrefill: config.DsparkNumExpertsPerTok,
            Scoring: scoring,
            Renormalize: config.NormTopkProb && config.DsparkNumExpertsPerTok > 1,
            RenormEpsilon: 1e-20f,
            Scale: (float)config.RoutedScalingFactor).Validated();
        ExpertGroupDescriptor routed = new ExpertGroupDescriptor(config.DsparkNRoutedExperts, new ExpertDescriptor(config.HiddenSize, config.MoeIntermediateSize, dtype)).Validated();
        return new SparseLayerDescriptor(index, new MoeLayerDescriptor(router, routed, Shared: null, SharedIsGated: false, program).Validated(), state, IsDraft: true);
    }

    private static MoeRouteScoring ParseScoring(string name) => name switch
    {
        "sqrtsoftplus" => MoeRouteScoring.SqrtSoftplus,
        "sigmoid" => MoeRouteScoring.Sigmoid,
        "softmax" => MoeRouteScoring.Softmax,
        _ => throw new ArgumentException($"Unknown V4.1 scoring function '{name}'.", nameof(name)),
    };
}

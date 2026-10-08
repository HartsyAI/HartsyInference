using HartsyInference.Core.Backends;
using HartsyInference.Core.Moe;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer;

/// <summary>Builds the generic sparse topology for a <see cref="TransformerConfig"/> whose feed-forward is MoE.</summary>
public static class MoeTopologyFactory
{
    /// <summary>
    /// Describes a transformer's layers. Layers before <see cref="MoeConfig.FirstDenseLayers"/> stay dense; the rest route
    /// through <see cref="MoeConfig"/>.
    /// </summary>
    /// <param name="config">Parsed transformer configuration.</param>
    /// <param name="expertDType">Storage type of the expert projections; F32 when the model is dequantized at load.</param>
    /// <param name="sharedExpertGated">Shared output is scaled by a sigmoid gate (Qwen2-MoE). The caller knows whether the gate tensor was loaded.</param>
    /// <param name="stateKindForLayer">Sequence state per layer; standard KV when null.</param>
    /// <param name="family">Diagnostic label.</param>
    public static SparseModelTopology FromTransformer(TransformerConfig config, DType? expertDType = null, bool sharedExpertGated = false,
        Func<int, SequenceStateKind>? stateKindForLayer = null, string family = "transformer")
    {
        ArgumentNullException.ThrowIfNull(config);
        DType dtype = expertDType ?? DType.F32;
        MoeConfig? moe = config.Moe;
        List<SparseLayerDescriptor> layers = new List<SparseLayerDescriptor>(config.NumLayers);
        for (int i = 0; i < config.NumLayers; i++)
        {
            SequenceStateKind state = stateKindForLayer?.Invoke(i) ?? SequenceStateKind.StandardKv;
            MoeLayerDescriptor? sparse = moe is null || i < moe.FirstDenseLayers ? null : BuildLayer(config, moe, dtype, sharedExpertGated);
            layers.Add(new SparseLayerDescriptor(i, sparse, state));
        }
        return new SparseModelTopology(config.HiddenSize, layers, family);
    }

    /// <summary>
    /// Qwen3.5 hybrid rule: every <paramref name="fullAttentionInterval"/>-th layer is full attention (standard KV), the rest are
    /// Gated DeltaNet (recurrent state). <paramref name="recurrentOverride"/> wins when the GGUF carries an explicit list.
    /// </summary>
    public static Func<int, SequenceStateKind> Qwen35States(int fullAttentionInterval, bool[]? recurrentOverride = null)
    {
        if (fullAttentionInterval <= 0) throw new ArgumentOutOfRangeException(nameof(fullAttentionInterval), fullAttentionInterval, "The full-attention interval must be positive.");
        return layer =>
        {
            if (layer < 0) throw new ArgumentOutOfRangeException(nameof(layer), layer, "Layer index must not be negative.");
            if (recurrentOverride is not null && layer >= recurrentOverride.Length)
                throw new ArgumentOutOfRangeException(nameof(layer), layer, $"The recurrent-layer list covers {recurrentOverride.Length} layers.");
            bool recurrent = recurrentOverride is { Length: > 0 } ? recurrentOverride[layer] : (layer + 1) % fullAttentionInterval != 0;
            return recurrent ? SequenceStateKind.RecurrentState : SequenceStateKind.StandardKv;
        };
    }

    private static MoeLayerDescriptor BuildLayer(TransformerConfig config, MoeConfig moe, DType dtype, bool sharedGated)
    {
        bool grouped = moe.ExpertGroupCount > 1;
        RouterDescriptor router = new RouterDescriptor(
            NumExperts: moe.NumExperts,
            TopKDecode: moe.NumExpertsPerTok,
            TopKPrefill: moe.NumExpertsPerTok,
            Scoring: ToScoring(moe.Scoring),
            GroupCount: moe.ExpertGroupCount,
            GroupsKept: moe.ExpertGroupUsedCount,
            Renormalize: moe.NormTopKProb,
            RenormEpsilon: grouped ? 1e-20f : 0f,
            Scale: moe.RoutedScalingFactor,
            // SigmoidLogitAdd scores by sigmoid but adds e_score_correction_bias to the logit for selection only.
            HasSelectionBias: moe.Scoring == MoeScoring.SigmoidLogitAdd,
            // Production flat sigmoid routing biases the logit before scoring; grouped routing biases the score.
            BiasSpace: moe.Scoring == MoeScoring.SigmoidLogitAdd && !grouped ? SelectionBiasSpace.Logit : SelectionBiasSpace.Score).Validated();

        ExpertGroupDescriptor routed = new ExpertGroupDescriptor(moe.NumExperts, new ExpertDescriptor(config.HiddenSize, moe.MoeIntermediateSize, dtype)).Validated();
        ExpertGroupDescriptor? shared = moe.SharedExpertIntermediateSize > 0
            ? new ExpertGroupDescriptor(1, new ExpertDescriptor(config.HiddenSize, moe.SharedExpertIntermediateSize, dtype)).Validated()
            : null;
        ExpertProgram program = new ExpertProgram(ToActivation(moe.Activation), float.PositiveInfinity, float.NegativeInfinity, float.PositiveInfinity);
        return new MoeLayerDescriptor(router, routed, shared, shared is not null && sharedGated, program).Validated();
    }

    private static MoeRouteScoring ToScoring(MoeScoring scoring) => scoring switch
    {
        MoeScoring.Softmax => MoeRouteScoring.Softmax,
        MoeScoring.Sigmoid or MoeScoring.SigmoidLogitAdd => MoeRouteScoring.Sigmoid,
        _ => throw new ArgumentOutOfRangeException(nameof(scoring), scoring, "Unknown MoE scoring."),
    };

    private static ExpertActivation ToActivation(ActivationKind kind) => kind switch
    {
        ActivationKind.Silu => ExpertActivation.Silu,
        ActivationKind.GeluTanh => ExpertActivation.GeluTanh,
        ActivationKind.Relu => ExpertActivation.Relu,
        ActivationKind.ReluSquared => ExpertActivation.ReluSquared,
        _ => throw new ArgumentOutOfRangeException(nameof(kind), kind, "Unknown activation."),
    };
}

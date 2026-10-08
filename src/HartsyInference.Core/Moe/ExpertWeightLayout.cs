using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Moe;

/// <summary>How an expert's gate, up and down projections are stored.</summary>
public enum ExpertWeightLayout
{
    /// <summary>Separate gate, up and down tensors.</summary>
    SplitGateUp,

    /// <summary>One fused <c>[2·I, H]</c> gate/up tensor plus down.</summary>
    FusedGateUp,
}

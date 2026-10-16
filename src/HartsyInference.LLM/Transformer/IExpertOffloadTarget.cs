using HartsyInference.Core.Moe;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer;

/// <summary>A mixture-of-experts model whose routed experts can be served by a <see cref="MoeExpertOffload"/>: a device cache of
/// the hottest experts, with the rest run on the CPU.</summary>
public interface IExpertOffloadTarget
{
    /// <summary>Model width each routed expert reads and writes.</summary>
    int ExpertHiddenSize { get; }

    /// <summary>Inner width of one routed expert.</summary>
    int ExpertIntermediateSize { get; }

    /// <summary>Every routed expert projection of every MoE layer.</summary>
    IEnumerable<Tensor> EnumerateExpertWeights();

    /// <summary>Routes every MoE layer's experts through <paramref name="offload"/>; returns each MoE layer with its expert count,
    /// for seeding the cache.</summary>
    IReadOnlyList<(int Layer, int ExpertCount)> AttachExpertOffload(MoeExpertOffload offload);
}

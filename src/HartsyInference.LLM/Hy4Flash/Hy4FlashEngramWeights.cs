using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Hy4Flash;

/// <summary>The dense weights of one Hy4 Flash Engram layer; the embedding table is read through <see cref="Hy4FlashRowTable"/>. Linear weights are
/// <c>[out, in]</c> in any dtype <c>IBackend.Linear</c> takes; norm gains, biases and scales are F32.</summary>
public sealed record Hy4FlashEngramWeights
{
    /// <summary><c>lookup_per_head_norms.i.weight</c>: one gain of the row width per hash head.</summary>
    public required IReadOnlyList<Tensor> PerHeadNorms { get; init; }

    /// <summary><c>value_proj.weight</c>, <c>[hidden, orders * embedPerNgram]</c>.</summary>
    public required Tensor ValueProj { get; init; }

    /// <summary><c>value_proj.bias</c>, <c>[hidden]</c>.</summary>
    public required Tensor ValueBias { get; init; }

    /// <summary><c>key_projs.i.weight</c>, one per residual stream.</summary>
    public required IReadOnlyList<Tensor> KeyProjs { get; init; }

    /// <summary><c>key_projs.i.bias</c>.</summary>
    public required IReadOnlyList<Tensor> KeyBiases { get; init; }

    /// <summary><c>norm_key.i.weight</c>.</summary>
    public required IReadOnlyList<Tensor> NormKey { get; init; }

    /// <summary><c>norm_query.i.weight</c>.</summary>
    public required IReadOnlyList<Tensor> NormQuery { get; init; }

    /// <summary><c>unified_pre_gamma</c>, <c>[hidden]</c>.</summary>
    public required Tensor UnifiedPreGamma { get; init; }

    /// <summary><c>unified_per_hc_scales.i</c>, one <c>[hidden]</c> scale per residual stream.</summary>
    public required IReadOnlyList<Tensor> PerHcScales { get; init; }
}

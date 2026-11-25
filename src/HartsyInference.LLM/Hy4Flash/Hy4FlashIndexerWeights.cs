using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Hy4Flash;

/// <summary>Weights of a Hy4 Flash compressed indexer, <c>[out, in]</c> Linear layout. The pooling tensors and norm gains are F32.</summary>
public sealed record Hy4FlashIndexerWeights
{
    /// <summary><c>query_projection</c>: query latent to <c>Heads * HeadDim</c>.</summary>
    public required Tensor QueryProjection { get; init; }

    /// <summary><c>head_weight_projection</c>: hidden state to one weight per head.</summary>
    public required Tensor HeadWeightProjection { get; init; }

    /// <summary><c>token_key_projection</c>: hidden state to one key per token.</summary>
    public required Tensor TokenKeyProjection { get; init; }

    /// <summary><c>token_key_norm</c> gain, F32 <c>[HeadDim]</c>.</summary>
    public required Tensor TokenKeyNorm { get; init; }

    /// <summary><c>key_value_projection</c>, F32 <c>[HeadDim, HeadDim]</c>.</summary>
    public required Tensor KeyValueProjection { get; init; }

    /// <summary><c>key_gate_projection</c>, F32 <c>[HeadDim, HeadDim]</c>.</summary>
    public required Tensor KeyGateProjection { get; init; }

    /// <summary><c>intra_group_ape</c>, F32 <c>[Kpool, HeadDim]</c>.</summary>
    public required Tensor IntraGroupApe { get; init; }

    /// <summary><c>key_norm</c> gain applied to the pooled group key, F32 <c>[HeadDim]</c>.</summary>
    public required Tensor KeyNorm { get; init; }
}

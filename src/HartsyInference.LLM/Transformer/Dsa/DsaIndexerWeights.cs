using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer.Dsa;

/// <summary>Weights of a DSA lightning indexer, in <c>[out, in]</c> Linear layout and any dtype <c>IBackend.Linear</c> takes.</summary>
public sealed record DsaIndexerWeights
{
    /// <summary><c>wq_b</c>: query latent to <c>Heads * HeadDim</c> queries.</summary>
    public required Tensor WqB { get; init; }

    /// <summary><c>wk</c>: hidden state to the shared key.</summary>
    public required Tensor Wk { get; init; }

    /// <summary>Key LayerNorm gain, F32 <c>[HeadDim]</c>.</summary>
    public required Tensor KeyNormWeight { get; init; }

    /// <summary>Key LayerNorm bias, F32 <c>[HeadDim]</c>; null is zero.</summary>
    public Tensor? KeyNormBias { get; init; }

    /// <summary><c>weights_proj</c>: hidden state to one weight per indexer head.</summary>
    public required Tensor WeightsProj { get; init; }
}

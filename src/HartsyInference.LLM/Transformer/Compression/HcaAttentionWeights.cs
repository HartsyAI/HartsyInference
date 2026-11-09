using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer.Compression;

/// <summary>Weights of one HCA layer. Projections are <c>[out, in]</c> in any dtype <c>IBackend.Linear</c> takes; the grouped output
/// projection, the compressor's positional bias and norm, and the sink are F32.</summary>
public sealed record HcaAttentionWeights
{
    /// <summary><c>wq_a</c>: hidden state to the query latent, <c>[QLoraRank, Dim]</c>.</summary>
    public required Tensor WqA { get; init; }

    /// <summary><c>q_norm</c> (RMS) gain, <c>[QLoraRank]</c>.</summary>
    public required Tensor QNorm { get; init; }

    /// <summary><c>wq_b</c>: query latent to <c>Heads * HeadDim</c> queries.</summary>
    public required Tensor WqB { get; init; }

    /// <summary><c>wkv</c>: hidden state to the shared key/value row, <c>[HeadDim, Dim]</c>.</summary>
    public required Tensor Wkv { get; init; }

    /// <summary><c>kv_norm</c> (RMS) gain, <c>[HeadDim]</c>.</summary>
    public required Tensor KvNorm { get; init; }

    /// <summary>Compressor <c>wkv</c>, <c>[HeadDim, Dim]</c>.</summary>
    public required Tensor CompressorWkv { get; init; }

    /// <summary>Compressor <c>wgate</c> (pooling scores), <c>[HeadDim, Dim]</c>.</summary>
    public required Tensor CompressorWgate { get; init; }

    /// <summary>Compressor positional bias added to the scores per slot, <c>[CompressRatio * HeadDim]</c>; null for none.</summary>
    public float[]? CompressorApe { get; init; }

    /// <summary>Compressor output RMS norm gain, <c>[HeadDim]</c>.</summary>
    public required float[] CompressorNorm { get; init; }

    /// <summary>Per-head sink logits joining the softmax denominator, F32 <c>[Heads]</c>.</summary>
    public required Tensor Sink { get; init; }

    /// <summary>Grouped <c>wo_a</c>, F32 <c>[OGroups * OLoraRank * GroupDim]</c>.</summary>
    public required float[] WoA { get; init; }

    /// <summary><c>wo_b</c>: <c>OGroups * OLoraRank</c> to the hidden width.</summary>
    public required Tensor WoB { get; init; }

    /// <summary>Sigmoid gate on the grouped output, query latent to <c>OGroups * OLoraRank</c> logits; null means no gate.</summary>
    public Tensor? OGate { get; init; }

    /// <summary>Dense head mix <c>[Heads, Heads]</c> applied before the inverse RoPE (ERNIE VHA); null means none.</summary>
    public Tensor? HeadMix { get; init; }
}

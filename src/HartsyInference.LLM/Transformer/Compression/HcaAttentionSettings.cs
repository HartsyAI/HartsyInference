using HartsyInference.Core.Backends.Rotary;

namespace HartsyInference.LLM.Transformer.Compression;

/// <summary>Shape of one heavily compressed attention (HCA) layer: low-rank queries, one shared key/value row per token (MQA), a sliding
/// window plus every completed compressed latent, a per-head sink, and a grouped low-rank output.</summary>
public sealed record HcaAttentionSettings
{
    /// <summary>Hidden width.</summary>
    public required int Dim { get; init; }

    /// <summary>Query heads.</summary>
    public required int Heads { get; init; }

    /// <summary>Per-head width; also the width of the shared key/value row.</summary>
    public required int HeadDim { get; init; }

    /// <summary>Rotated width at the end of each query and key/value row.</summary>
    public required int RopeDim { get; init; }

    /// <summary>Query latent width (<c>q_lora_rank</c>).</summary>
    public required int QLoraRank { get; init; }

    /// <summary>Head groups of the grouped output projection (<c>o_groups</c>).</summary>
    public required int OGroups { get; init; }

    /// <summary>Per-group width of the grouped output projection (<c>o_lora_rank</c>).</summary>
    public required int OLoraRank { get; init; }

    /// <summary>Sliding window: the newest positions every query attends, itself included.</summary>
    public required int Window { get; init; }

    /// <summary>Tokens pooled into one compressed latent.</summary>
    public required int CompressRatio { get; init; }

    /// <summary>Epsilon of every RMS norm in the layer.</summary>
    public required float NormEps { get; init; }

    /// <summary>Normalizes each query head with an unweighted RMS norm before RoPE (ERNIE 5.5 <c>qk_norm_type: per_head</c>).</summary>
    public bool QueryHeadNorm { get; init; }

    /// <summary>Pairing inside the rotated slice; ERNIE 5.5 and DeepSeek V4 pair adjacent values.</summary>
    public RotaryPairing RopePairing { get; init; } = RotaryPairing.Interleaved;

    /// <summary>Input width of one output group, <c>Heads * HeadDim / OGroups</c>.</summary>
    public int GroupDim => Heads * HeadDim / OGroups;

    /// <summary>Softmax scale <c>HeadDim^-0.5</c>.</summary>
    public float Scale => 1f / MathF.Sqrt(HeadDim);

    /// <summary>Fails fast on a shape the layer cannot run.</summary>
    public void Validate()
    {
        if (Dim < 1 || Heads < 1 || HeadDim < 1 || QLoraRank < 1 || OGroups < 1 || OLoraRank < 1 || Window < 1)
            throw new ArgumentException("HCA widths, heads, groups and window must be positive.");
        if (RopeDim < 2 || (RopeDim & 1) != 0 || RopeDim > HeadDim)
            throw new ArgumentException("RopeDim must be even, positive and at most HeadDim.");
        if (Heads * HeadDim % OGroups != 0) throw new ArgumentException("Heads x HeadDim must divide into OGroups.");
        if (CompressRatio < 2) throw new ArgumentException("An HCA layer pools at least 2 tokens per latent.");
        if (CompressRatio == 4)
            throw new ArgumentException("compress_ratio 4 needs the overlapping C4 compressor, which HcaAttention does not implement.");
    }
}

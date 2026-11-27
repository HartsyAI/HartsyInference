using HartsyInference.Core.Backends.Latent;
using HartsyInference.Core.Backends.Rotary;

namespace HartsyInference.LLM.Transformer.Dsa;

/// <summary>Shape and numerics of a DSA lightning indexer: per-head queries from the query latent, one shared key per token, and a
/// weighted ReLU score summed over heads (see <see cref="DsaIndexer"/>).</summary>
public sealed record DsaIndexerSettings
{
    /// <summary>Indexer heads (<c>index_n_heads</c>).</summary>
    public required int Heads { get; init; }

    /// <summary>Width of each indexer query and key (<c>index_head_dim</c>); a power of two when <see cref="Hadamard"/> is set.</summary>
    public required int HeadDim { get; init; }

    /// <summary>Rotated width inside each indexer vector.</summary>
    public required int RopeDim { get; init; }

    /// <summary>Keys kept per query (<c>index_topk</c>).</summary>
    public required int TopK { get; init; }

    /// <summary>Where the rotated slice sits: the leading values (ERNIE 5.5, rope first then the rest) or the trailing ones.</summary>
    public bool RopeLeading { get; init; } = true;

    /// <summary>Pairing inside the rotated slice; ERNIE 5.5's indexer is NeoX style.</summary>
    public RotaryPairing RopePairing { get; init; } = RotaryPairing.RotateHalf;

    /// <summary>Applies the orthonormal Walsh-Hadamard rotation to queries and keys after RoPE.</summary>
    public bool Hadamard { get; init; } = true;

    /// <summary>Epsilon of the key LayerNorm.</summary>
    public float KeyNormEps { get; init; } = 1e-6f;

    /// <summary>Quantize-dequantize round trip of the queries before scoring; <see cref="LatentEncoding.F32"/> skips it.</summary>
    public LatentEncoding QueryEncoding { get; init; } = LatentEncoding.Fp8E4M3Ue8m0x128;

    /// <summary>Quantize-dequantize round trip of the keys as the indexer cache stores them; <see cref="LatentEncoding.F32"/> skips it.</summary>
    public LatentEncoding KeyEncoding { get; init; } = LatentEncoding.Fp8E4M3Ue8m0x128;

    /// <summary>Fails fast on a shape the indexer cannot run.</summary>
    public void Validate()
    {
        if (Heads < 1 || HeadDim < 2 || TopK < 1) throw new ArgumentException("Indexer Heads, HeadDim and TopK must be positive.");
        if (RopeDim < 2 || (RopeDim & 1) != 0 || RopeDim > HeadDim) throw new ArgumentException("Indexer RopeDim must be even and fit in HeadDim.");
        if (Hadamard && (HeadDim & (HeadDim - 1)) != 0) throw new ArgumentException("A Hadamard indexer needs a power-of-two HeadDim.");
        foreach (LatentEncoding e in (ReadOnlySpan<LatentEncoding>)[QueryEncoding, KeyEncoding])
        {
            int group = LatentEncodings.GroupSize(e);
            if (group != 0 && HeadDim % group != 0) throw new ArgumentException($"Indexer HeadDim must be a multiple of the {e} group.");
        }
    }
}

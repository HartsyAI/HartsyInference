using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Latent;
using HartsyInference.LLM.Transformer;
using HartsyInference.LLM.Transformer.Dsa;
using HartsyInference.LLM.Transformer.Kda;

namespace HartsyInference.LLM.Kimi;

/// <summary>The shape of a Kimi K3.1 text decoder: KDA linear-attention layers and NoPE multi-head latent attention layers with a DSA
/// indexer, a dense first layer and latent-MoE layers after it, and block attention residuals around every sublayer. Dimensions are
/// the checkpoint's; the synthetic tests use smaller ones with the same structure.</summary>
public sealed record KimiK31Config
{
    /// <summary>Model width.</summary>
    public required int Hidden { get; init; }

    /// <summary>Vocabulary size.</summary>
    public required int Vocab { get; init; }

    /// <summary>Decoder layers.</summary>
    public required int Layers { get; init; }

    /// <summary>Leading dense layers (<c>first_k_dense_replace</c>); the rest are MoE.</summary>
    public required int DenseLayers { get; init; }

    /// <summary>Dense FFN width of the leading layers.</summary>
    public required int DenseIntermediate { get; init; }

    /// <summary>Heads of the KDA and MLA layers.</summary>
    public required int Heads { get; init; }

    /// <summary>Per-head width of the KDA layers (<c>head_dim</c>).</summary>
    public required int KdaHeadDim { get; init; }

    /// <summary>Low-rank width of the KDA forget gate.</summary>
    public required int KdaForgetRank { get; init; }

    /// <summary>KDA's causal conv taps on q, k and v.</summary>
    public int ConvKernel { get; init; } = 4;

    /// <summary>Lower bound of the KDA forget gate (<c>gate_lower_bound</c>).</summary>
    public float GateLowerBound { get; init; } = -5f;

    /// <summary>Layers that use MLA with the DSA indexer; the others use KDA.</summary>
    public required IReadOnlySet<int> FullAttentionLayers { get; init; }

    /// <summary>MLA query latent width.</summary>
    public required int QLoraRank { get; init; }

    /// <summary>MLA key/value latent width.</summary>
    public required int KvLoraRank { get; init; }

    /// <summary>MLA per-head key/query width without position.</summary>
    public required int QkNopeDim { get; init; }

    /// <summary>Width of the unrotated shared key slice (NoPE: never rotated).</summary>
    public required int QkRopeDim { get; init; }

    /// <summary>MLA per-head value width.</summary>
    public required int VDim { get; init; }

    /// <summary>DSA indexer heads.</summary>
    public required int IndexHeads { get; init; }

    /// <summary>DSA indexer head width.</summary>
    public required int IndexHeadDim { get; init; }

    /// <summary>Keys kept per query by the DSA indexer (<c>index_topk</c>).</summary>
    public required int IndexTopK { get; init; }

    /// <summary>Routed experts.</summary>
    public required int Experts { get; init; }

    /// <summary>Experts each token routes to.</summary>
    public required int ExpertsPerToken { get; init; }

    /// <summary>Routed expert width.</summary>
    public required int MoeIntermediate { get; init; }

    /// <summary>Shared experts.</summary>
    public required int SharedExperts { get; init; }

    /// <summary>Latent width the routed experts run in.</summary>
    public required int LatentSize { get; init; }

    /// <summary>SiTU gate soft-cap (<c>activation_situ_beta</c>).</summary>
    public float SituBeta { get; init; } = 4f;

    /// <summary>SiTU value soft-cap (<c>activation_situ_linear_beta</c>).</summary>
    public float SituLinearBeta { get; init; } = 25f;

    /// <summary>Layers per attention-residual block (<c>attn_res_block_size</c>).</summary>
    public int AttnResBlock { get; init; } = 12;

    /// <summary>RMSNorm epsilon.</summary>
    public float RmsNormEps { get; init; } = 1e-6f;

    /// <summary>Whether layer <paramref name="index"/> is an MLA layer.</summary>
    public bool IsMla(int index) => FullAttentionLayers.Contains(index);

    /// <summary>Whether layer <paramref name="index"/> has a dense FFN.</summary>
    public bool IsDense(int index) => index < DenseLayers;

    /// <summary>KDA layer shape.</summary>
    public KdaSettings Kda => new(Hidden, Heads, KdaHeadDim, ConvKernel, KdaForgetRank, GateLowerBound);

    /// <summary>NoPE MLA layer shape on the shared DSA layer: no rotation, no sliding window, no Hadamard, F32 indexer caches.</summary>
    public DsaAttentionSettings Mla => new()
    {
        Dim = Hidden, Heads = Heads, QLoraRank = QLoraRank, KvLoraRank = KvLoraRank, QkNopeDim = QkNopeDim, QkRopeDim = QkRopeDim,
        VDim = VDim, Window = 0, NormEps = RmsNormEps,
        Indexer = new DsaIndexerSettings
        {
            Heads = IndexHeads, HeadDim = IndexHeadDim, RopeDim = 2, TopK = IndexTopK, Hadamard = false,
            QueryEncoding = LatentEncoding.F32, KeyEncoding = LatentEncoding.F32,
        },
    };
}

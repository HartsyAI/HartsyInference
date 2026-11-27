namespace HartsyInference.LLM.Glm5Next;

/// <summary>The shape of a <c>glm5_next</c> text decoder (GLM-5.3-Flash, Sage S2.1 Flash): KDA and NoPE compressed-pool DSA layers under
/// manifold hyper-connections, dense then clamped-SwiGLU sigmoid-routed MoE layers, and an optional MTP layer. Dimensions are the
/// checkpoint's; tiny test models keep the structure with smaller ones.</summary>
public sealed record Glm5NextConfig
{
    /// <summary>Model width.</summary>
    public required int Hidden { get; init; }

    /// <summary>Vocabulary size.</summary>
    public required int Vocab { get; init; }

    /// <summary>Decoder layers, the MTP layer excluded.</summary>
    public required int Layers { get; init; }

    /// <summary>Multi-token-prediction layers after the decoder (<c>num_nextn_predict_layers</c>); each is stored as layer <see cref="Layers"/> + i.</summary>
    public required int MtpLayers { get; init; }

    /// <summary>Attention heads (KDA, MLA and the key/value heads alike).</summary>
    public required int Heads { get; init; }

    /// <summary>Token mixer per decoder layer.</summary>
    public required IReadOnlyList<Glm5NextAttentionKind> LayerKinds { get; init; }

    /// <summary>Per decoder layer: the feed-forward block is MoE (<c>sparse</c>) rather than dense.</summary>
    public required IReadOnlyList<bool> SparseMlp { get; init; }

    /// <summary>Dense feed-forward inner width.</summary>
    public required int DenseIntermediate { get; init; }

    /// <summary>Clamp of every SwiGLU, dense, shared and routed (<c>swiglu_limit</c>).</summary>
    public required float SwigluLimit { get; init; }

    /// <summary>KDA head width (<c>linear_attn_config.head_dim</c>); the forget and output gates' low rank equals it.</summary>
    public required int KdaHeadDim { get; init; }

    /// <summary>KDA causal conv taps on q, k and v.</summary>
    public required int ConvKernel { get; init; }

    /// <summary>Forget gate bound: the log-decay lies in (bound, 0).</summary>
    public required float GateLowerBound { get; init; }

    /// <summary>MLA query latent width.</summary>
    public required int QLoraRank { get; init; }

    /// <summary>MLA key/value latent width.</summary>
    public required int KvLoraRank { get; init; }

    /// <summary>MLA per-head query/key width (no rotated slice: NoPE).</summary>
    public required int QkDim { get; init; }

    /// <summary>MLA per-head value width.</summary>
    public required int VDim { get; init; }

    /// <summary>Indexer heads.</summary>
    public required int IndexHeads { get; init; }

    /// <summary>Indexer head width.</summary>
    public required int IndexHeadDim { get; init; }

    /// <summary>Raw keys one query keeps from the pooled selection (<c>index_topk</c>); a multiple of <see cref="IndexPool"/>.</summary>
    public required int IndexTopK { get; init; }

    /// <summary>Tokens per key pool (<c>index_kpool</c>).</summary>
    public required int IndexPool { get; init; }

    /// <summary>Whether the query's incomplete pool is appended as raw tokens (<c>index_kpool_always_select_tail</c>).</summary>
    public required bool IndexAlwaysSelectTail { get; init; }

    /// <summary>Routed experts per MoE layer.</summary>
    public required int Experts { get; init; }

    /// <summary>Experts each token routes to.</summary>
    public required int ExpertsPerToken { get; init; }

    /// <summary>Routed and shared expert inner width per expert.</summary>
    public required int MoeIntermediate { get; init; }

    /// <summary>Shared experts (their widths add up).</summary>
    public required int SharedExperts { get; init; }

    /// <summary>Multiplier on the normalized routing weights.</summary>
    public required float RoutedScalingFactor { get; init; }

    /// <summary>Whether the selected routing weights are renormalized to sum to 1 before scaling.</summary>
    public required bool NormTopKProb { get; init; }

    /// <summary>Hyper-connection streams per token.</summary>
    public required int HcMult { get; init; }

    /// <summary>Epsilon of the hyper-connection gates and Sinkhorn normalizers.</summary>
    public required float HcEps { get; init; }

    /// <summary>Sinkhorn rounds on the hyper-connection mixing matrix.</summary>
    public required int HcSinkhornIters { get; init; }

    /// <summary>RMSNorm epsilon, also the hyper-connection mix norm's.</summary>
    public required float RmsNormEps { get; init; }

    /// <summary>Token ids that end a reply.</summary>
    public required IReadOnlyList<int> EosTokenIds { get; init; }

    /// <summary>Padding token id.</summary>
    public required int PadTokenId { get; init; }

    /// <summary>The vision tower, or null for a text-only config.</summary>
    public Glm5NextVisionConfig? Vision { get; init; }

    /// <summary>Id of the <c>&lt;|image|&gt;</c> placeholder token.</summary>
    public int ImageTokenId { get; init; }

    /// <summary>Id of the <c>&lt;|video|&gt;</c> placeholder token.</summary>
    public int VideoTokenId { get; init; }

    /// <summary>Width of the KDA q, k or v projections: <c>Heads · KdaHeadDim</c>.</summary>
    public int KdaKeyDim => Heads * KdaHeadDim;

    /// <summary>Pools the indexer selects per query: <c>IndexTopK / IndexPool</c>.</summary>
    public int IndexSelectedPools => IndexTopK / IndexPool;

    /// <summary>Layer index of MTP layer <paramref name="i"/>.</summary>
    public int MtpLayerIndex(int i) => Layers + i;

    /// <summary>Whether <paramref name="layer"/> is an MTP layer.</summary>
    public bool IsMtpLayer(int layer) => layer >= Layers && layer < Layers + MtpLayers;

    /// <summary>Whether the feed-forward block of <paramref name="layer"/> is MoE. An MTP layer always is: this checkpoint's MTP layer is DSA + MoE
    /// (the index test pins it), which is read from the checkpoint's layout rather than from a config field.</summary>
    public bool IsSparse(int layer) => IsMtpLayer(layer) || SparseMlp[layer];

    /// <summary>Whether <paramref name="layer"/> runs DSA; an MTP layer always does.</summary>
    public bool IsDsa(int layer) => IsMtpLayer(layer) || LayerKinds[layer] == Glm5NextAttentionKind.Dsa;
}

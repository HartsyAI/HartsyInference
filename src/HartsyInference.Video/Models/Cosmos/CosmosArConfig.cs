namespace HartsyInference.Video.Models.Cosmos;

/// <summary>Architecture config for <see cref="CosmosArTransformer"/> — the Llama3-shape discrete-token
/// autoregressive backbone of Cosmos-Predict1 Video2World. A distinct type (not an LLM
/// <c>TransformerConfig</c>): the V2W combination — 3D RoPE over <c>(T,H,W)</c>, per-layer non-causal T5
/// cross-attention, and an additive 3D absolute-position table — is not expressible as a generic decoder config.
/// Designed as a reusable backbone so the Phase-10 action-conditioned world models instantiate the same class.
///
/// <para>Constants confirmed against <c>docs/Research/COSMOS_PREDICT1_VIDEO2WORLD_ARCHITECTURE.md</c> §3–6.
/// Items flagged below resolve from the real <c>model.pt</c> key dump (research-doc Open Questions §2–5).</para></summary>
public sealed record CosmosArConfig
{
    /// <summary>Number of transformer layers (5B: 16, 13B: 40).</summary>
    public required int NumLayers { get; init; }

    /// <summary>Hidden dimension (5B: 4096, 13B: 5120).</summary>
    public required int Dim { get; init; }

    /// <summary>Query attention heads (both sizes: 32).</summary>
    public int NumHeads { get; init; } = 32;

    /// <summary>Key/value heads for GQA (both sizes: 8 → group 4).</summary>
    public int NumKvHeads { get; init; } = 8;

    /// <summary>Per-head dimension (128 for both — 4096/32 and the explicit 12B value).</summary>
    public int HeadDim { get; init; } = 128;

    /// <summary>SwiGLU feed-forward intermediate size (both: 14336 — Llama3-8B's).</summary>
    public int FfnHiddenSize { get; init; } = 14336;

    /// <summary>Vocabulary = DV codebook size (64,000 = prod([8,8,8,5,5,5])). No text vocab share.</summary>
    public int VocabSize { get; init; } = 64000;

    /// <summary>RMSNorm epsilon (1e-5).</summary>
    public float NormEps { get; init; } = 1e-5f;

    /// <summary>RoPE base θ. Research-doc Open Q §4: <c>BASE_CONFIG</c> doesn't set it; Llama3 convention (500,000)
    /// is the leaning, but the dataclass fallback is 10,000 and the small latent grid would tolerate it — VERIFY
    /// against <c>model.rope.rope_theta</c> at integration.</summary>
    public float RopeTheta { get; init; } = 500000f;

    /// <summary>Contiguous head_dim/2 partition <c>(T, H, W)</c> for 3D RoPE (must sum to <c>HeadDim/2</c>). The exact
    /// Cosmos split is integration-pending (research-doc §5 describes contiguous per-axis ranges); default 16/24/24
    /// for head_dim 128.</summary>
    public (int T, int H, int W) RopeSection { get; init; } = (16, 24, 24);

    /// <summary>T5-11B cross-attention context dim (== T5-11B hidden = 1024).</summary>
    public int ContextDim { get; init; } = 1024;

    /// <summary>QK per-head RMSNorm before RoPE. ON for V2W (5B/13B); the base 4B/12B leave it off.</summary>
    public bool QkNorm { get; init; } = true;

    /// <summary>Insert a T5 cross-attention block (V2W adapter). True for 5B/13B.</summary>
    public bool InsertCrossAttn { get; init; } = true;

    /// <summary>Cross-attention every k layers. V2W = 1 (every layer).</summary>
    public int InsertCrossAttnEveryKLayers { get; init; } = 1;

    /// <summary>Additive 3D absolute-position table added to embeddings before the stack (V2W-only). True for 5B/13B.</summary>
    public bool ApplyAbsPosEmb { get; init; } = true;

    /// <summary>Whether Q/K/V share one fused projection. Research-doc Open Q §2: <c>fuse_qkv=False</c> for V2W —
    /// unfused is the shipped path (fused is a later optimization).</summary>
    public bool FuseQkv { get; init; } = false;

    /// <summary>GQA group factor.</summary>
    public int KvGroup => NumHeads / NumKvHeads;

    /// <summary>Layer <paramref name="i"/> carries a cross-attention block.</summary>
    public bool HasCrossAttn(int i) => InsertCrossAttn && (i % InsertCrossAttnEveryKLayers == 0);

    /// <summary>Cosmos-Predict1-5B-Video2World (4B base + per-layer cross-attn).</summary>
    public static CosmosArConfig Cosmos5BV2W => new() { NumLayers = 16, Dim = 4096 };

    /// <summary>Cosmos-Predict1-13B-Video2World (12B base + per-layer cross-attn).</summary>
    public static CosmosArConfig Cosmos13BV2W => new() { NumLayers = 40, Dim = 5120 };
}

namespace SharpInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for Ideogram 4 (<c>ideogram-oss/ideogram4</c>, 9.3B, non-commercial license). Verified verbatim against upstream <c>src/ideogram4/modeling_ideogram4.py</c> (<c>Ideogram4Config</c> + <c>Ideogram4Transformer</c>).
///
/// Ideogram 4 is a **single-stream unified-sequence DiT**: text tokens (Qwen3-VL multi-layer hidden states) and image-latent tokens occupy ONE length-L sequence, kept apart by 3D MRoPE + an <c>image_indicator</c> embedding + (for packed multi-prompt batches) a block-diagonal <c>segment_ids</c> mask. Distinctive vs every other DiT in this codebase:
/// <list type="bullet">
///   <item>**Scale-only AdaLN (NO shift)** + <c>tanh</c>-gated residuals + **sandwich norms** (a norm before AND after each sublayer). Per-block modulation is <c>Linear(512 → 4*hidden)</c> → <c>chunk(4)</c> = <c>(scale_msa, gate_msa, scale_mlp, gate_mlp)</c>.</item>
///   <item>**3D sectioned MRoPE** <c>(24, 20, 20)</c>, θ=5e6, standard HF <c>rotate_half</c> on <c>cat(freqs, freqs)</c> — see <see cref="DiTBlocks.Ideogram4Mrope"/>.</item>
///   <item>**Fused QKV** <c>Linear(hidden → 3*hidden, bias=False)</c>, per-head QK-RMSNorm, <c>o</c> proj bias=False, SwiGLU <c>w2(silu(w1)·w3)</c> bias=False.</item>
///   <item>Text/image features placed into the unified sequence by **masked addition** (text positions zero <c>x</c>, image positions zero <c>llm_features</c>), not concatenation.</item>
///   <item>Final layer: <c>LayerNorm(no affine)</c> + scale-only AdaLN (<c>Linear(512 → hidden)</c>, double-SiLU input) → <c>Linear(hidden → 128)</c>.</item>
/// </list>
///
/// Conditioning is Qwen3-VL-8B-Instruct (language tower) with 13 layers tapped — see <see cref="LlamaStyleEncoderConfig.Qwen3_VL_8B"/> and <see cref="QwenActivationLayers"/>. VAE is the Flux.2 KL autoencoder (<c>VaeConfig.Flux2</c>), but latent un-normalization uses fixed constants (<see cref="Ideogram4LatentNorm"/>), NOT the VAE BatchNorm.</summary>
public sealed record Ideogram4Config
{
    /// <summary>Model hidden dim (<c>emb_dim</c>). 4608 = 18 heads × 256 head_dim.</summary>
    public int EmbDim { get; init; } = 4608;

    /// <summary>Number of single-stream transformer blocks.</summary>
    public int NumLayers { get; init; } = 34;

    /// <summary>Number of attention heads.</summary>
    public int NumHeads { get; init; } = 18;

    /// <summary>Per-head dimension (<c>EmbDim / NumHeads</c>). 256 — unusually large; the SDPA path must support 256-wide heads.</summary>
    public int HeadDim => EmbDim / NumHeads;

    /// <summary>SwiGLU FFN intermediate size.</summary>
    public int IntermediateSize { get; init; } = 12288;

    /// <summary>AdaLN bottleneck dim (<c>adanln_dim</c>) — the shared timestep→modulation width.</summary>
    public int AdalnDim { get; init; } = 512;

    /// <summary>Transformer I/O channels = VAE z_channels (32) × patch² (4) = 128.</summary>
    public int InChannels { get; init; } = 128;

    /// <summary>Qwen3-VL hidden size (4096) × number of tapped layers (13) = 53248.</summary>
    public int LlmFeaturesDim { get; init; } = 4096 * 13;

    /// <summary>MRoPE base frequency (θ).</summary>
    public int RopeTheta { get; init; } = 5_000_000;

    /// <summary>MRoPE per-axis sections <c>(temporal, height, width)</c>. Sum × 2 must equal head_dim.</summary>
    public (int Temporal, int Height, int Width) MropeSection { get; init; } = (24, 20, 20);

    /// <summary>Stream/QK RMSNorm epsilon (1e-5 in the block; <c>llm_cond_norm</c> uses 1e-6 internally).</summary>
    public float NormEps { get; init; } = 1e-5f;

    /// <summary>VAE spatial downsample factor (Flux.2 VAE: 8×).</summary>
    public int VaeScaleFactor { get; init; } = 8;

    /// <summary>Pipeline-side patch size (2×2 channel-fold on the VAE latent: 32 → 128).</summary>
    public int PatchSize { get; init; } = 2;

    /// <summary>Max text tokens fed to the transformer.</summary>
    public int MaxTextTokens { get; init; } = 2048;

    /// <summary>Qwen3-VL decoder-layer indices whose hidden states are concatenated, in upstream (post-decoder-layer) convention. Maps to <c>EncodeMultiLayer</c> HF indices via <c>+1</c> (see <see cref="QwenActivationLayersHf"/>).</summary>
    public static readonly int[] QwenActivationLayers = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35];

    /// <summary>The same tap layers in <see cref="LlamaStyleEncoder.EncodeMultiLayer"/> HF convention (0=embeddings, k=post-layer-(k−1)). Upstream captures the output of decoder layer <c>l</c>, which is HF hidden-state index <c>l+1</c>.</summary>
    public static readonly int[] QwenActivationLayersHf = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 36];

    /// <summary>The single Ideogram 4 preset. Both the conditional and unconditional transformers share this architecture.</summary>
    public static Ideogram4Config V4 => new();
}

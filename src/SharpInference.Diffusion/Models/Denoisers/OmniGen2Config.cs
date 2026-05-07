namespace SharpInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for the OmniGen2 unified diffusion transformer (<c>OmniGen2Transformer2DModel</c> from
/// <c>VectorSpaceLab/OmniGen2</c>). The architecture is a Lumina-2-style joint transformer: text tokens are embedded
/// from a Qwen2.5-VL-3B MLLM (last hidden state, dim 2048), refined through <see cref="NumRefinerLayers"/>
/// non-modulated <c>context_refiner</c> blocks; image patch tokens are embedded via <c>x_embedder</c> (a 2×2 patchify
/// of the 16-channel latent) and passed through <see cref="NumRefinerLayers"/> modulated <c>noise_refiner</c> blocks;
/// the two streams are then concatenated <c>[txt, img]</c> along the sequence dim and processed jointly through
/// <see cref="NumLayers"/> modulated <c>layers</c> blocks. Every block is a single-stream block with shared self-attention
/// (no separate text/image attention paths inside the block — the joining happens once before the joint stack).
/// <para>Defaults below match the public <c>OmniGen2/OmniGen2</c> safetensors at HuggingFace (verified against
/// <c>transformer/config.json</c>): hidden=2520, 32 layers, 21 attention heads, 7 KV heads (GQA), patch=2,
/// in_channels=16, axes_dim_rope=[40,40,40] (3-axis RoPE summing to head_dim=120), text_feat_dim=2048
/// (matching Qwen2.5-VL-3B hidden), timestep_scale=1000.</para></summary>
public record OmniGen2Config
{
    /// <summary>Hidden dimension of the joint transformer (2520 for the public release).</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Number of attention heads (21).</summary>
    public required int NumAttentionHeads { get; init; }

    /// <summary>Number of key/value heads for grouped-query attention (7 = 21/3 groups).</summary>
    public required int NumKvHeads { get; init; }

    /// <summary>Per-head dimension. Computed as <c>HiddenSize / NumAttentionHeads</c>; must equal <c>sum(AxesDimRope)</c>.</summary>
    public int HeadDim => HiddenSize / NumAttentionHeads;

    /// <summary>Number of joint transformer blocks (32 for the public release).</summary>
    public required int NumLayers { get; init; }

    /// <summary>Number of refiner blocks for both <c>context_refiner</c> (text) and <c>noise_refiner</c> (image). Defaults to 2.</summary>
    public int NumRefinerLayers { get; init; } = 2;

    /// <summary>Patch size for latent → token packing (2×2).</summary>
    public int PatchSize { get; init; } = 2;

    /// <summary>Number of latent channels (16, Flux-VAE compatible).</summary>
    public int InChannels { get; init; } = 16;

    /// <summary>Number of output channels. Null means same as <see cref="InChannels"/>.</summary>
    public int? OutChannels { get; init; }

    /// <summary>FFN inner-dim multiplier rounding granularity (Lumina convention: round
    /// <c>4 * hidden</c> up to a multiple of this value). 256 in the reference config.</summary>
    public int MultipleOf { get; init; } = 256;

    /// <summary>Optional explicit FFN dimension multiplier. When null the default inner dim is <c>4 * hidden</c>
    /// (still rounded by <see cref="MultipleOf"/>); when set the inner dim becomes
    /// <c>ffn_dim_multiplier * (4 * hidden)</c> per Lumina's <c>LuminaFeedForward</c> rule.</summary>
    public float? FfnDimMultiplier { get; init; }

    /// <summary>RMSNorm epsilon for in-block norms. 1e-5 in the reference config.</summary>
    public float NormEps { get; init; } = 1e-5f;

    /// <summary>QK-norm epsilon for the per-head Q/K RMSNorm. 1e-5 in the reference config (matches the <c>eps=1e-5</c> in the diffusers <c>Attention(qk_norm="rms_norm", eps=1e-5)</c> call).</summary>
    public float QkNormEps { get; init; } = 1e-5f;

    /// <summary>Per-axis RoPE dim split <c>(time, height, width)</c>. Sum must equal <see cref="HeadDim"/>. <c>[40, 40, 40]</c> for the public release.</summary>
    public int[] AxesDimRope { get; init; } = [40, 40, 40];

    /// <summary>Per-axis maximum position used by the RoPE precompute. <c>[1024, 1664, 1664]</c> for the public release —
    /// supports very long captions (1024 text tokens) and 3328-pixel-wide images at patch_size=2 (1664 tokens × 2 patches × 1 pixel/token = 3328).</summary>
    public int[] AxesLens { get; init; } = [1024, 1664, 1664];

    /// <summary>RoPE base frequency. 10000 for OmniGen2 (the diffusers code hardcodes this).</summary>
    public int RopeTheta { get; init; } = 10000;

    /// <summary>Text feature dimension fed into <c>caption_embedder</c>. 2048 for the public release (matches Qwen2.5-VL-3B hidden).</summary>
    public int TextFeatDim { get; init; } = 2048;

    /// <summary>Timestep scale used inside <c>time_caption_embed.time_proj</c>. 1000 for the public release —
    /// scales the [0,1] flow-match timestep into the 0–1000 sinusoidal frequency range.</summary>
    public float TimestepScale { get; init; } = 1000.0f;

    /// <summary>Number of slots in the learnable <c>image_index_embedding</c> parameter. 5 in the reference config —
    /// only used by the editing / multi-image-input path which is intentionally <b>not</b> implemented here. The
    /// table is loaded as a passthrough weight so checkpoints don't fail key validation.</summary>
    public int MaxRefImages { get; init; } = 5;

    /// <summary>Conditioning embedding dim used inside the modulation linears (<c>min(hidden_size, 1024)</c> per Lumina).</summary>
    public int ConditioningDim => Math.Min(HiddenSize, 1024);

    /// <summary>OmniGen2 v1 preset (<c>OmniGen2/OmniGen2</c>): 32 joint layers + 2 refiner layers each side, hidden=2520,
    /// 21 attention heads, 7 KV heads, head_dim=120, patch=2, in_channels=16, axes_dim_rope=[40,40,40],
    /// axes_lens=[1024,1664,1664], text_feat_dim=2048, timestep_scale=1000. Pairs with Qwen2.5-VL-3B as the text
    /// encoder and the standard 16-channel Flux-style VAE.</summary>
    public static OmniGen2Config V1 => new()
    {
        HiddenSize = 2520,
        NumAttentionHeads = 21,
        NumKvHeads = 7,
        NumLayers = 32,
        NumRefinerLayers = 2,
        PatchSize = 2,
        InChannels = 16,
        MultipleOf = 256,
        NormEps = 1e-5f,
        AxesDimRope = [40, 40, 40],
        AxesLens = [1024, 1664, 1664],
        RopeTheta = 10000,
        TextFeatDim = 2048,
        TimestepScale = 1000.0f,
    };
}

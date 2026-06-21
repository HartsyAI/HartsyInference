namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for the Boogu-Image diffusion transformer (<c>BooguImageTransformer2DModel</c> from
/// <c>github.com/boogu-project/Boogu-Image</c>). An OmniGen2 / Lumina-Image-2.0 lineage joint transformer with a
/// dual-stream (double-stream) stage layered before the single-stream stack and a reference-image editing path.
///
/// <para>Forward topology: <c>context_refiner</c> (text) + <c>noise_refiner</c> (noise image) + <c>ref_image_refiner</c>
/// (reference image, edit only) → <see cref="NumDoubleStreamLayers"/> dual-stream blocks (instruction stream + image
/// stream with joint cross-attention) → <c>NumLayers − NumDoubleStreamLayers</c> single-stream blocks on the fused
/// <c>[instruct, ref_img, noise_img]</c> sequence → <c>LuminaLayerNormContinuous</c> output norm + unpatchify.</para>
///
/// <para>Defaults match the released <c>Boogu-Image-0.1-Base</c>/<c>-Edit</c> <c>transformer/config.json</c>:
/// hidden=3360, 40 total layers (8 double-stream → 32 single-stream), 2 refiner layers per refiner, 28 attention
/// heads, 7 KV heads (GQA group 4), head_dim=120, patch=2, in_channels=16, axes_dim_rope=[40,40,40],
/// axes_lens=[2048,1664,1664], instruction_feat_dim=4096 (Qwen3-VL-8B hidden), timestep_scale=1000.</para></summary>
public sealed record BooguImageConfig
{
    /// <summary>Hidden dimension of the joint transformer (3360 for the public release).</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Total number of transformer layers (40). The first <see cref="NumDoubleStreamLayers"/> are dual-stream;
    /// the remainder are single-stream.</summary>
    public required int NumLayers { get; init; }

    /// <summary>Number of leading dual-stream (double-stream) blocks (8).</summary>
    public required int NumDoubleStreamLayers { get; init; }

    /// <summary>Number of single-stream blocks (<c>NumLayers − NumDoubleStreamLayers</c>).</summary>
    public int NumSingleStreamLayers => NumLayers - NumDoubleStreamLayers;

    /// <summary>Number of refiner blocks per refiner stack (context / noise / ref-image). Defaults to 2.</summary>
    public int NumRefinerLayers { get; init; } = 2;

    /// <summary>Number of attention heads (28).</summary>
    public required int NumAttentionHeads { get; init; }

    /// <summary>Number of key/value heads for grouped-query attention (7 = 28/4 groups).</summary>
    public required int NumKvHeads { get; init; }

    /// <summary>Per-head dimension. <c>HiddenSize / NumAttentionHeads</c>; must equal <c>sum(AxesDimRope)</c> (120).</summary>
    public int HeadDim => HiddenSize / NumAttentionHeads;

    /// <summary>Patch size for latent → token packing (2×2).</summary>
    public int PatchSize { get; init; } = 2;

    /// <summary>Number of latent channels (16, FLUX-VAE compatible).</summary>
    public int InChannels { get; init; } = 16;

    /// <summary>Output channels. Null means same as <see cref="InChannels"/>.</summary>
    public int? OutChannels { get; init; }

    /// <summary>FFN inner-dim rounding granularity (round <c>4·hidden</c> up to a multiple of this). 256 in the config.</summary>
    public int MultipleOf { get; init; } = 256;

    /// <summary>Optional FFN dimension multiplier applied to <c>4·hidden</c> before rounding. Null in the release.</summary>
    public float? FfnDimMultiplier { get; init; }

    /// <summary>Lumina SwiGLU FFN inner dim: <c>round_up(ffn_dim_multiplier? · 4·hidden, multiple_of)</c>.</summary>
    public int FfnInnerDim
    {
        get
        {
            int target = 4 * HiddenSize;
            if (FfnDimMultiplier is float mult) target = (int)(mult * target);
            int rem = target % MultipleOf;
            return rem == 0 ? target : target + (MultipleOf - rem);
        }
    }

    /// <summary>RMSNorm epsilon for in-block norms (1e-5).</summary>
    public float NormEps { get; init; } = 1e-5f;

    /// <summary>QK-norm epsilon for the per-head Q/K RMSNorm (1e-5).</summary>
    public float QkNormEps { get; init; } = 1e-5f;

    /// <summary>Output-norm LayerNorm epsilon (1e-6 in <c>LuminaLayerNormContinuous</c>).</summary>
    public float OutNormEps { get; init; } = 1e-6f;

    /// <summary>Per-axis RoPE dim split <c>(time, height, width)</c>. Sum must equal <see cref="HeadDim"/>. <c>[40,40,40]</c>.</summary>
    public int[] AxesDimRope { get; init; } = [40, 40, 40];

    /// <summary>Per-axis maximum RoPE position. <c>[2048, 1664, 1664]</c> in the release.</summary>
    public int[] AxesLens { get; init; } = [2048, 1664, 1664];

    /// <summary>RoPE base frequency (10000).</summary>
    public int RopeTheta { get; init; } = 10000;

    /// <summary>Instruction feature dimension fed into <c>caption_embedder</c> (4096, Qwen3-VL-8B hidden).</summary>
    public int InstructionFeatDim { get; init; } = 4096;

    /// <summary>Timestep scale inside the sinusoidal time embedding (1000).</summary>
    public float TimestepScale { get; init; } = 1000.0f;

    /// <summary>Slots in the learnable <c>image_index_embedding</c> table — distinguishes up to this many reference
    /// images in the edit path (5 in the release).</summary>
    public int MaxRefImages { get; init; } = 5;

    /// <summary>Conditioning embedding dim used by the modulation linears (<c>min(hidden, 1024)</c> per Lumina) = 1024.</summary>
    public int ConditioningDim => Math.Min(HiddenSize, 1024);

    /// <summary>Boogu-Image-0.1 release preset (Base / Edit / Turbo share this transformer geometry): hidden=3360,
    /// 40 layers (8 double + 32 single), 2 refiner layers, 28 heads / 7 KV heads, head_dim=120, patch=2,
    /// in_channels=16, axes_dim_rope=[40,40,40], axes_lens=[2048,1664,1664], instruction_feat_dim=4096,
    /// timestep_scale=1000. Pairs with Qwen3-VL-8B and the FLUX.1 VAE.</summary>
    public static BooguImageConfig V01 => new()
    {
        HiddenSize = 3360,
        NumLayers = 40,
        NumDoubleStreamLayers = 8,
        NumRefinerLayers = 2,
        NumAttentionHeads = 28,
        NumKvHeads = 7,
        PatchSize = 2,
        InChannels = 16,
        MultipleOf = 256,
        NormEps = 1e-5f,
        QkNormEps = 1e-5f,
        AxesDimRope = [40, 40, 40],
        AxesLens = [2048, 1664, 1664],
        RopeTheta = 10000,
        InstructionFeatDim = 4096,
        TimestepScale = 1000.0f,
        MaxRefImages = 5,
    };
}

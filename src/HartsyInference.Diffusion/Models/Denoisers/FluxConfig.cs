namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for Flux DiT transformers (Flux.1 Dev, Flux.1 Schnell). Architecture is defined by depth (double-stream blocks) and depth_single_blocks (single-stream blocks).</summary>
public sealed record FluxConfig
{
    /// <summary>Hidden dimension of the transformer (3072 for Flux.1).</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Number of attention heads (24 for Flux.1).</summary>
    public required int NumHeads { get; init; }

    /// <summary>Per-head dimension (128 for Flux.1, = HiddenSize / NumHeads).</summary>
    public int HeadDim { get; init; } = 128;

    /// <summary>Number of double-stream blocks (19 for Flux.1).</summary>
    public required int Depth { get; init; }

    /// <summary>Number of single-stream blocks (38 for Flux.1).</summary>
    public required int DepthSingleBlocks { get; init; }

    /// <summary>Input channels per packed patch (64 = 16 latent channels * 2x2 patch).</summary>
    public int InChannels { get; init; } = 64;

    /// <summary>Output channels per packed patch (64).</summary>
    public int OutChannels { get; init; } = 64;

    /// <summary>T5 text encoder context dimension (4096 for T5-XXL).</summary>
    public int ContextInDim { get; init; } = 4096;

    /// <summary>CLIP pooled vector dimension (768 for CLIP-L).</summary>
    public int VecInDim { get; init; } = 768;

    /// <summary>MLP ratio (4.0 for Flux.1: mlp_dim = hidden * 4).</summary>
    public float MlpRatio { get; init; } = 4.0f;

    /// <summary>RoPE axis dimensions. Must sum to HeadDim. Default: [16, 56, 56].</summary>
    public int[] AxesDim { get; init; } = [16, 56, 56];

    /// <summary>RoPE base frequency (10000).</summary>
    public int Theta { get; init; } = 10000;

    /// <summary>Whether Q/K/V projections have bias (true for Flux.1, false for Flux.2).</summary>
    public bool QkvBias { get; init; } = true;

    /// <summary>Whether to embed guidance scale via MLP (true for Dev, false for Schnell).</summary>
    public bool GuidanceEmbed { get; init; } = true;

    /// <summary>QK-norm RMSNorm epsilon.</summary>
    public float QkNormEps { get; init; } = 1e-6f;

    /// <summary>Flux.1 Dev preset (12B params, guidance embedding, 19+38 blocks).</summary>
    public static FluxConfig Dev => new()
    {
        HiddenSize = 3072,
        NumHeads = 24,
        HeadDim = 128,
        Depth = 19,
        DepthSingleBlocks = 38,
        GuidanceEmbed = true,
    };

    /// <summary>Flux.1 Schnell preset (12B params, no guidance, 19+38 blocks).</summary>
    public static FluxConfig Schnell => new()
    {
        HiddenSize = 3072,
        NumHeads = 24,
        HeadDim = 128,
        Depth = 19,
        DepthSingleBlocks = 38,
        GuidanceEmbed = false,
    };

    /// <summary>FLUX.1 Tools (Canny / Depth / Fill) preset. Same shape as Dev (12B params, 19+38 blocks, guidance embedding) BUT with a wider <c>x_embedder</c> input — 32 latent channels (16 noise + 16 control, concatenated along the channel dim) instead of 16, packed as <c>32×4 = 128</c> instead of <c>16×4 = 64</c>. The pipeline detects Tools mode by inspecting <see cref="FluxTransformer.XEmbedInputDim"/> at load time and routes through the control-image-concat path.
    /// <para>The three Tools variants (Canny / Depth / Fill) all share the same architectural shape — the differentiator is purely how the "control" image is preprocessed (canny edges, depth map, masked-image+mask). Loaders pick the preprocessor based on filename or metadata.</para></summary>
    public static FluxConfig Flux1Tools => new()
    {
        HiddenSize = 3072,
        NumHeads = 24,
        HeadDim = 128,
        Depth = 19,
        DepthSingleBlocks = 38,
        GuidanceEmbed = true,
        InChannels = 128,
    };

    /// <summary>FLUX.1 Fill preset. Same 12B Dev shape (19+38 blocks, guidance embedding) BUT with a 384-channel
    /// <c>x_embedder</c> input: 64 packed noise + 320 packed conditioning (masked-image latent 64 + mask 256),
    /// vs Canny/Depth's 128 (64 + 64). The pipeline detects Fill by <see cref="FluxTransformer.XEmbedInputDim"/>
    /// ≥ 384 at load time. The masked-image + mask conditioning prep differs from the single-control-image concat
    /// used by Canny/Depth.</summary>
    public static FluxConfig Flux1Fill => new()
    {
        HiddenSize = 3072,
        NumHeads = 24,
        HeadDim = 128,
        Depth = 19,
        DepthSingleBlocks = 38,
        GuidanceEmbed = true,
        InChannels = 384,
    };

    /// <summary>Auto-detects config from weight key count. Counts transformer_blocks.* and single_transformer_blocks.* prefixes.</summary>
    public static FluxConfig FromWeights(IReadOnlyDictionary<string, object> keys)
    {
        int doubleBlocks = 0;
        int singleBlocks = 0;
        bool hasGuidance = false;

        foreach (string key in keys.Keys)
        {
            if (key.StartsWith("transformer_blocks.", StringComparison.Ordinal))
            {
                int dotIdx = key.IndexOf('.', 20);
                if (dotIdx > 0 && int.TryParse(key.AsSpan(19, dotIdx - 19), out int blockIdx))
                {
                    if (blockIdx + 1 > doubleBlocks)
                        doubleBlocks = blockIdx + 1;
                }
            }
            else if (key.StartsWith("single_transformer_blocks.", StringComparison.Ordinal))
            {
                int dotIdx = key.IndexOf('.', 27);
                if (dotIdx > 0 && int.TryParse(key.AsSpan(26, dotIdx - 26), out int blockIdx))
                {
                    if (blockIdx + 1 > singleBlocks)
                        singleBlocks = blockIdx + 1;
                }
            }
            else if (key.StartsWith("time_text_embed.guidance_embedder", StringComparison.Ordinal))
            {
                hasGuidance = true;
            }
        }

        return new FluxConfig
        {
            HiddenSize = 3072,
            NumHeads = 24,
            HeadDim = 128,
            Depth = doubleBlocks,
            DepthSingleBlocks = singleBlocks,
            GuidanceEmbed = hasGuidance,
        };
    }
}

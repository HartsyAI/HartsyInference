namespace SharpInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for Flux.2 transformers (Flux.2 Dev 32B, Flux.2 Klein 4B/9B). Next-gen MMDiT evolution of Flux.1 with 16×16 VAE, Mistral/Qwen text encoder, and SOTA quality. Core transformer architecture is similar to Flux.1 but with different block counts, hidden dims, and no Q/K/V bias.</summary>
public record Flux2Config
{
    /// <summary>Hidden dimension of the transformer.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Number of attention heads.</summary>
    public required int NumHeads { get; init; }

    /// <summary>Per-head dimension.</summary>
    public int HeadDim { get; init; } = 128;

    /// <summary>Number of double-stream blocks.</summary>
    public required int Depth { get; init; }

    /// <summary>Number of single-stream blocks.</summary>
    public required int DepthSingleBlocks { get; init; }

    /// <summary>Input channels per packed patch. Flux.2 uses 16×16 VAE with different latent packing.</summary>
    public int InChannels { get; init; } = 64;

    /// <summary>Output channels per packed patch.</summary>
    public int OutChannels { get; init; } = 64;

    /// <summary>Text encoder context dimension. Flux.2 Dev uses Mistral (4096), Klein uses Qwen (3584).</summary>
    public int ContextInDim { get; init; } = 4096;

    /// <summary>Pooled vector dimension from secondary text encoder.</summary>
    public int VecInDim { get; init; } = 768;

    /// <summary>MLP ratio.</summary>
    public float MlpRatio { get; init; } = 4.0f;

    /// <summary>RoPE axis dimensions.</summary>
    public int[] AxesDim { get; init; } = [16, 56, 56];

    /// <summary>RoPE base frequency.</summary>
    public int Theta { get; init; } = 10000;

    /// <summary>Whether Q/K/V projections have bias (false for Flux.2, unlike Flux.1).</summary>
    public bool QkvBias { get; init; } = false;

    /// <summary>Whether to embed guidance scale via MLP.</summary>
    public bool GuidanceEmbed { get; init; } = true;

    /// <summary>QK-norm epsilon.</summary>
    public float QkNormEps { get; init; } = 1e-6f;

    /// <summary>VAE downscale factor (16 for Flux.2 vs 8 for Flux.1).</summary>
    public int VaeDownscaleFactor { get; init; } = 16;

    /// <summary>Text encoder type used by this variant.</summary>
    public required Flux2TextEncoderType TextEncoderType { get; init; }

    /// <summary>Flux.2 Dev preset (32B params, Mistral text encoder, guidance embedding).</summary>
    public static Flux2Config Dev => new()
    {
        HiddenSize = 3072,
        NumHeads = 24,
        HeadDim = 128,
        Depth = 24,
        DepthSingleBlocks = 48,
        GuidanceEmbed = true,
        TextEncoderType = Flux2TextEncoderType.Mistral,
        ContextInDim = 4096,
    };

    /// <summary>Flux.2 Klein 9B preset (Qwen text encoder).</summary>
    public static Flux2Config Klein9B => new()
    {
        HiddenSize = 3072,
        NumHeads = 24,
        HeadDim = 128,
        Depth = 12,
        DepthSingleBlocks = 24,
        GuidanceEmbed = false,
        TextEncoderType = Flux2TextEncoderType.Qwen,
        ContextInDim = 3584,
    };

    /// <summary>Flux.2 Klein 4B preset (Qwen text encoder, smallest variant).</summary>
    public static Flux2Config Klein4B => new()
    {
        HiddenSize = 2048,
        NumHeads = 16,
        HeadDim = 128,
        Depth = 8,
        DepthSingleBlocks = 16,
        GuidanceEmbed = false,
        TextEncoderType = Flux2TextEncoderType.Qwen,
        ContextInDim = 3584,
    };
}

/// <summary>Text encoder type used by Flux.2 variants.</summary>
public enum Flux2TextEncoderType
{
    /// <summary>Mistral-based text encoder (Flux.2 Dev 32B).</summary>
    Mistral,

    /// <summary>Qwen-based text encoder (Flux.2 Klein 4B/9B).</summary>
    Qwen,
}

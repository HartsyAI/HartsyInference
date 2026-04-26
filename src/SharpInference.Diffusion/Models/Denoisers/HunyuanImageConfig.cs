namespace SharpInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for Hunyuan Image 2.1 MMDiT transformer (17B by Tencent). Features a 32×32 VAE downscale, native 2048×2048 resolution, and includes distilled + refiner variants. Uses dual text encoders and a unique double/single-stream MMDiT architecture.</summary>
public record HunyuanImageConfig
{
    /// <summary>Hidden dimension of the transformer.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Number of attention heads.</summary>
    public required int NumHeads { get; init; }

    /// <summary>Per-head dimension.</summary>
    public int HeadDim { get; init; } = 128;

    /// <summary>Number of double-stream blocks (joint image+text attention).</summary>
    public required int NumDoubleBlocks { get; init; }

    /// <summary>Number of single-stream blocks (image-only after joint processing).</summary>
    public required int NumSingleBlocks { get; init; }

    /// <summary>Patch size for latent→token embedding.</summary>
    public int PatchSize { get; init; } = 2;

    /// <summary>Number of latent channels (32 for Hunyuan Image's 32×32 VAE).</summary>
    public int InChannels { get; init; } = 32;

    /// <summary>Text context dimension from the text encoder.</summary>
    public int ContextDim { get; init; } = 4096;

    /// <summary>Pooled projection dimension from CLIP-like encoder.</summary>
    public int PooledProjectionDim { get; init; } = 1024;

    /// <summary>RoPE base frequency for positional encoding.</summary>
    public int RopeTheta { get; init; } = 10000;

    /// <summary>Whether to embed guidance scale via MLP (true for full model, false for distilled).</summary>
    public bool GuidanceEmbed { get; init; } = true;

    /// <summary>QK-norm epsilon.</summary>
    public float QkNormEps { get; init; } = 1e-6f;

    /// <summary>Whether to use QK-norm (RMSNorm on Q/K).</summary>
    public bool UseQkNorm { get; init; } = true;

    /// <summary>MLP ratio (typically 4.0).</summary>
    public float MlpRatio { get; init; } = 4.0f;

    /// <summary>Hunyuan Image 2.1 full preset (17B params).</summary>
    public static HunyuanImageConfig V21 => new()
    {
        HiddenSize = 3072,
        NumHeads = 24,
        HeadDim = 128,
        NumDoubleBlocks = 20,
        NumSingleBlocks = 40,
        InChannels = 32,
        GuidanceEmbed = true,
    };

    /// <summary>Hunyuan Image 2.1 distilled preset (faster, fewer steps).</summary>
    public static HunyuanImageConfig V21Distilled => new()
    {
        HiddenSize = 3072,
        NumHeads = 24,
        HeadDim = 128,
        NumDoubleBlocks = 20,
        NumSingleBlocks = 40,
        InChannels = 32,
        GuidanceEmbed = false,
    };
}

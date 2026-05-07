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

    /// <summary>Number of latent channels (64 for Hunyuan Image, the 32× VAE outputs 64-channel latents).</summary>
    public int InChannels { get; init; } = 64;

    /// <summary>Number of refiner layers in the MLLM context embedder (<c>HunyuanImageTokenRefiner</c>). Default 2.</summary>
    public int NumRefinerLayers { get; init; } = 2;

    /// <summary>Primary text context dimension (Qwen2.5-VL MLLM hidden size).</summary>
    public int TextEmbedDim { get; init; } = 4096;

    /// <summary>Optional secondary text context dimension (ByT5 glyph encoder). Null when only the primary encoder is used.</summary>
    public int? TextEmbedDim2 { get; init; }

    /// <summary>RoPE base frequency for positional encoding.</summary>
    public float RopeTheta { get; init; } = 256.0f;

    /// <summary>Per-axis RoPE dim split (height, width). Must sum to <see cref="HeadDim"/>.</summary>
    public int[] RopeAxesDim { get; init; } = [64, 64];

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
        NumRefinerLayers = 2,
        PatchSize = 1,
        InChannels = 64,
        TextEmbedDim = 4096,
        TextEmbedDim2 = 1472,
        GuidanceEmbed = true,
        RopeTheta = 256.0f,
        RopeAxesDim = [64, 64],
    };

    /// <summary>Hunyuan Image 2.1 distilled preset (faster, fewer steps).</summary>
    public static HunyuanImageConfig V21Distilled => new()
    {
        HiddenSize = 3072,
        NumHeads = 24,
        HeadDim = 128,
        NumDoubleBlocks = 20,
        NumSingleBlocks = 40,
        NumRefinerLayers = 2,
        PatchSize = 1,
        InChannels = 64,
        TextEmbedDim = 4096,
        TextEmbedDim2 = 1472,
        GuidanceEmbed = false,
        RopeTheta = 256.0f,
        RopeAxesDim = [64, 64],
    };
}

namespace SharpInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for Qwen-Image / Qwen-Image 2.0 MMDiT transformers (7B-20B by Alibaba). Features unified generation and editing (inpaint, outpaint, relighting, style transfer) through a single model. Uses Qwen VL text encoder and MMDiT architecture with AdaLN-Zero modulation.</summary>
public record QwenImageConfig
{
    /// <summary>Hidden dimension of the transformer.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Number of attention heads.</summary>
    public required int NumHeads { get; init; }

    /// <summary>Per-head dimension.</summary>
    public int HeadDim { get; init; } = 128;

    /// <summary>Number of joint transformer blocks.</summary>
    public required int Depth { get; init; }

    /// <summary>Patch size for latent→token embedding.</summary>
    public int PatchSize { get; init; } = 2;

    /// <summary>Number of latent channels.</summary>
    public int InChannels { get; init; } = 16;

    /// <summary>Qwen VL text encoder context dimension.</summary>
    public int ContextDim { get; init; } = 4096;

    /// <summary>Pooled projection dimension.</summary>
    public int PooledProjectionDim { get; init; } = 2048;

    /// <summary>MLP ratio for feed-forward blocks.</summary>
    public float MlpRatio { get; init; } = 4.0f;

    /// <summary>RoPE base frequency.</summary>
    public int RopeTheta { get; init; } = 10000;

    /// <summary>QK-norm epsilon.</summary>
    public float QkNormEps { get; init; } = 1e-6f;

    /// <summary>Whether to use QK-norm.</summary>
    public bool UseQkNorm { get; init; } = true;

    /// <summary>Whether this model supports unified editing tasks (inpaint, outpaint, etc.).</summary>
    public bool SupportsEditing { get; init; }

    /// <summary>Qwen-Image 7B preset.</summary>
    public static QwenImageConfig V1_7B => new()
    {
        HiddenSize = 3072,
        NumHeads = 24,
        Depth = 24,
        SupportsEditing = false,
    };

    /// <summary>Qwen-Image 2.0 14B preset.</summary>
    public static QwenImageConfig V2_14B => new()
    {
        HiddenSize = 4096,
        NumHeads = 32,
        Depth = 32,
        SupportsEditing = true,
    };

    /// <summary>Qwen-Image 2.0 20B preset.</summary>
    public static QwenImageConfig V2_20B => new()
    {
        HiddenSize = 5120,
        NumHeads = 40,
        Depth = 40,
        SupportsEditing = true,
    };
}

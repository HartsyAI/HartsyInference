namespace SharpInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for AuraFlow MMDiT transformer. Uses Pile T5-XXL text encoder and SDXL-compatible VAE (4-channel latent). Architecture features joint single-stream MMDiT blocks with separate text/image token streams joined via attention.</summary>
public record AuraFlowConfig
{
    /// <summary>Hidden dimension of the transformer.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Number of attention heads.</summary>
    public required int NumHeads { get; init; }

    /// <summary>Per-head dimension.</summary>
    public int HeadDim { get; init; } = 64;

    /// <summary>Number of joint double-stream MMDiT blocks.</summary>
    public required int NumDoubleBlocks { get; init; }

    /// <summary>Number of single-stream blocks (image-only after joint processing).</summary>
    public required int NumSingleBlocks { get; init; }

    /// <summary>Patch size for latent→token embedding. AuraFlow uses 2x2 patches.</summary>
    public int PatchSize { get; init; } = 2;

    /// <summary>Number of latent channels (4 for SDXL-compatible VAE).</summary>
    public int InChannels { get; init; } = 4;

    /// <summary>T5 text context dimension (4096 for T5-XXL).</summary>
    public int ContextDim { get; init; } = 4096;

    /// <summary>Maximum positional embedding grid size.</summary>
    public int PosEmbedMaxSize { get; init; } = 192;

    /// <summary>Number of register tokens (learnable, prepended to image tokens).</summary>
    public int NumRegisterTokens { get; init; } = 0;

    /// <summary>QK-norm epsilon.</summary>
    public float QkNormEps { get; init; } = 1e-6f;

    /// <summary>Whether to use QK-norm.</summary>
    public bool UseQkNorm { get; init; } = true;

    /// <summary>AuraFlow v0.3 preset (6.8B params, 4+32 blocks, 3072 hidden).</summary>
    public static AuraFlowConfig V03 => new()
    {
        HiddenSize = 3072,
        NumHeads = 48,
        HeadDim = 64,
        NumDoubleBlocks = 4,
        NumSingleBlocks = 32,
        PatchSize = 2,
        InChannels = 4,
    };
}

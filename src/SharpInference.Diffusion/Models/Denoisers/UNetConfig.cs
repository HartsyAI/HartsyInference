namespace SharpInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for UNet2DConditionModel. Covers SD1.5 and SDXL variants.</summary>
public record UNetConfig
{
    /// <summary>Number of input channels (4 for SD1.5/SDXL latent space).</summary>
    public int InChannels { get; init; } = 4;

    /// <summary>Number of output channels (4 for SD1.5/SDXL).</summary>
    public int OutChannels { get; init; } = 4;

    /// <summary>Base channel count for the first down block.</summary>
    public int ModelChannels { get; init; } = 320;

    /// <summary>Output channel counts per down block stage.</summary>
    public int[] BlockOutChannels { get; init; } = [320, 640, 1280, 1280];

    /// <summary>Number of ResNet layers per block.</summary>
    public int LayersPerBlock { get; init; } = 2;

    /// <summary>Cross-attention context dimension (768 for CLIP ViT-L/14 in SD1.5).</summary>
    public int CrossAttentionDim { get; init; } = 768;

    /// <summary>Number of attention heads per block. If single value, used for all blocks.</summary>
    public int[] AttentionHeadDim { get; init; } = [8, 8, 8, 8];

    /// <summary>Which down blocks have cross-attention. Last block in SD1.5 has no attention.</summary>
    public bool[] DownBlockHasAttention { get; init; } = [true, true, true, false];

    /// <summary>Which up blocks have cross-attention. First up block in SD1.5 has no attention.</summary>
    public bool[] UpBlockHasAttention { get; init; } = [false, true, true, true];

    /// <summary>GroupNorm number of groups.</summary>
    public int NormNumGroups { get; init; } = 32;

    /// <summary>GroupNorm epsilon.</summary>
    public float NormEps { get; init; } = 1e-5f;

    /// <summary>Preset for Stable Diffusion 1.5.</summary>
    public static UNetConfig Sd15 => new()
    {
        InChannels = 4,
        OutChannels = 4,
        ModelChannels = 320,
        BlockOutChannels = [320, 640, 1280, 1280],
        LayersPerBlock = 2,
        CrossAttentionDim = 768,
        AttentionHeadDim = [8, 8, 8, 8],
        DownBlockHasAttention = [true, true, true, false],
        UpBlockHasAttention = [false, true, true, true],
    };
}

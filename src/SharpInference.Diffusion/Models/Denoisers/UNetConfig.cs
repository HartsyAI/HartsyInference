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

    /// <summary>Cross-attention context dimension (768 for CLIP ViT-L/14 in SD1.5, 2048 for SDXL dual CLIP).</summary>
    public int CrossAttentionDim { get; init; } = 768;

    /// <summary>Number of attention heads per block. Matches diffusers' attention_head_dim parameter
    /// which (confusingly) specifies head count, not head dimension. head_dim = channels / num_heads.</summary>
    public int[] NumAttentionHeads { get; init; } = [8, 8, 8, 8];

    /// <summary>Which down blocks have cross-attention. Last block in SD1.5 has no attention.</summary>
    public bool[] DownBlockHasAttention { get; init; } = [true, true, true, false];

    /// <summary>Which up blocks have cross-attention. First up block in SD1.5 has no attention.</summary>
    public bool[] UpBlockHasAttention { get; init; } = [false, true, true, true];

    /// <summary>Number of transformer (BasicTransformerBlock) layers per attention block at each level. SD1.5 uses [1,1,1,1], SDXL uses [1,2,10].</summary>
    public int[] TransformerLayersPerBlock { get; init; } = [1, 1, 1, 1];

    /// <summary>GroupNorm number of groups.</summary>
    public int NormNumGroups { get; init; } = 32;

    /// <summary>GroupNorm epsilon.</summary>
    public float NormEps { get; init; } = 1e-5f;

    /// <summary>Whether to use linear projection (nn.Linear) instead of 1x1 Conv2d for attention proj_in/proj_out. SDXL uses true.</summary>
    public bool UseLinearProjection { get; init; } = false;

    /// <summary>Input dimension for the ADM conditioning vector. 0 means no ADM conditioning (SD1.5). 2816 for SDXL base, 2560 for SDXL refiner.</summary>
    public int AdmInChannels { get; init; } = 0;

    /// <summary>Dimension for each sinusoidal embedding of size/crop/target scalars in ADM conditioning. SDXL uses 256.</summary>
    public int AdditionTimeEmbedDim { get; init; } = 256;

    /// <summary>Preset for Stable Diffusion 1.5.</summary>
    public static UNetConfig Sd15 => new()
    {
        InChannels = 4,
        OutChannels = 4,
        ModelChannels = 320,
        BlockOutChannels = [320, 640, 1280, 1280],
        LayersPerBlock = 2,
        CrossAttentionDim = 768,
        NumAttentionHeads = [8, 8, 8, 8],
        DownBlockHasAttention = [true, true, true, false],
        UpBlockHasAttention = [false, true, true, true],
        TransformerLayersPerBlock = [1, 1, 1, 1],
        UseLinearProjection = false,
        AdmInChannels = 0,
    };

    /// <summary>Preset for SDXL base model. 3 levels, heterogeneous transformer depth [1,2,10], dual CLIP 2048-dim context, ADM micro-conditioning.</summary>
    public static UNetConfig SdxlBase => new()
    {
        InChannels = 4,
        OutChannels = 4,
        ModelChannels = 320,
        BlockOutChannels = [320, 640, 1280],
        LayersPerBlock = 2,
        CrossAttentionDim = 2048,
        NumAttentionHeads = [5, 10, 20],
        DownBlockHasAttention = [false, true, true],
        UpBlockHasAttention = [true, true, false],
        TransformerLayersPerBlock = [1, 2, 10],
        UseLinearProjection = true,
        AdmInChannels = 2816,
        AdditionTimeEmbedDim = 256,
    };

    /// <summary>Preset for SDXL refiner model. 4 levels, uniform transformer depth 4, CLIP-G only 1280-dim context, aesthetic score conditioning.</summary>
    public static UNetConfig SdxlRefiner => new()
    {
        InChannels = 4,
        OutChannels = 4,
        ModelChannels = 384,
        BlockOutChannels = [384, 768, 1536, 1536],
        LayersPerBlock = 2,
        CrossAttentionDim = 1280,
        NumAttentionHeads = [6, 12, 24, 24],
        DownBlockHasAttention = [false, true, true, false],
        UpBlockHasAttention = [false, true, true, false],
        TransformerLayersPerBlock = [4, 4, 4, 4],
        UseLinearProjection = true,
        AdmInChannels = 2560,
        AdditionTimeEmbedDim = 256,
    };
}

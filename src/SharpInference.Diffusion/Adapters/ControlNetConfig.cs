namespace SharpInference.Diffusion.Adapters;

/// <summary>Configuration for ControlNet conditioning adapters. Supports SD1.5, SDXL, and Flux-based ControlNets with various conditioning modes (depth, canny, openpose, scribble, tile, etc.).</summary>
public sealed record ControlNetConfig
{
    /// <summary>Base model architecture this ControlNet was trained for.</summary>
    public required ControlNetBaseModel BaseModel { get; init; }

    /// <summary>Conditioning mode (depth, canny, openpose, etc.).</summary>
    public required ControlNetMode Mode { get; init; }

    /// <summary>Number of input hint channels (typically 3 for RGB condition images).</summary>
    public int ConditioningChannels { get; init; } = 3;

    /// <summary>Channel multiplier schedule matching the base UNet/DiT block structure.</summary>
    public int[] BlockOutChannels { get; init; } = [320, 640, 1280, 1280];

    /// <summary>Conditioning scale applied to ControlNet outputs before injection into the base model.</summary>
    public float ConditioningScale { get; init; } = 1.0f;

    /// <summary>Number of ResNet layers per down block (matching base model).</summary>
    public int LayersPerBlock { get; init; } = 2;

    /// <summary>Whether to use cross-attention in transformer blocks.</summary>
    public bool UseCrossAttention { get; init; } = true;

    /// <summary>Cross-attention dimension (must match base model).</summary>
    public int CrossAttentionDim { get; init; } = 768;

    /// <summary>SD 1.5 ControlNet preset.</summary>
    public static ControlNetConfig Sd15(ControlNetMode mode) => new()
    {
        BaseModel = ControlNetBaseModel.Sd15,
        Mode = mode,
        BlockOutChannels = [320, 640, 1280, 1280],
        CrossAttentionDim = 768,
    };

    /// <summary>SDXL ControlNet preset.</summary>
    public static ControlNetConfig Sdxl(ControlNetMode mode) => new()
    {
        BaseModel = ControlNetBaseModel.Sdxl,
        Mode = mode,
        BlockOutChannels = [320, 640, 1280],
        CrossAttentionDim = 2048,
    };
}

/// <summary>Base model architecture a ControlNet was trained for.</summary>
public enum ControlNetBaseModel
{
    /// <summary>Stable Diffusion 1.5 UNet.</summary>
    Sd15,

    /// <summary>SDXL UNet.</summary>
    Sdxl,

    /// <summary>Flux DiT transformer.</summary>
    Flux,
}

/// <summary>ControlNet conditioning mode.</summary>
public enum ControlNetMode
{
    /// <summary>Depth map conditioning.</summary>
    Depth,

    /// <summary>Canny edge conditioning.</summary>
    Canny,

    /// <summary>OpenPose skeleton conditioning.</summary>
    OpenPose,

    /// <summary>Scribble/sketch conditioning.</summary>
    Scribble,

    /// <summary>Tile-based conditioning for upscaling.</summary>
    Tile,

    /// <summary>Normal map conditioning.</summary>
    Normal,

    /// <summary>Semantic segmentation conditioning.</summary>
    Segmentation,

    /// <summary>Inpainting mask conditioning.</summary>
    Inpaint,

    /// <summary>Line art conditioning.</summary>
    LineArt,

    /// <summary>SoftEdge / HED conditioning.</summary>
    SoftEdge,
}

namespace HartsyInference.ThreeD.Models.Hunyuan3D.Paint;

/// <summary>Architecture of a Hunyuan3D Paint multiview UNet (a Stable Diffusion 2.x UNet with the 2.5D attention extension). Head values are
/// head counts, as in the diffusers config's <c>attention_head_dim</c>.</summary>
public sealed record HunyuanPaintConfig
{
    /// <summary>Channels per resolution level.</summary>
    public int[] BlockOutChannels { get; init; } = [320, 640, 1280, 1280];

    /// <summary>Attention heads per level (head dimension is the level width over this).</summary>
    public int[] Heads { get; init; } = [5, 10, 20, 20];

    /// <summary>Residual layers per level.</summary>
    public int LayersPerBlock { get; init; } = 2;

    /// <summary>Width of the text and DINO context tokens.</summary>
    public int CrossAttentionDim { get; init; } = 1024;

    /// <summary>GroupNorm groups.</summary>
    public int NormGroups { get; init; } = 32;

    /// <summary>Input channels of the generating UNet: noisy latent, normal latent and position latent.</summary>
    public int InChannels { get; init; } = 12;

    /// <summary>Input channels of the reference (dual-stream) UNet.</summary>
    public int ReferenceInChannels { get; init; } = 4;

    /// <summary>Output (latent) channels.</summary>
    public int OutChannels { get; init; } = 4;

    /// <summary>Learned text tokens per material.</summary>
    public int TextTokens { get; init; } = 77;

    /// <summary>Materials generated together, in order; the first is the albedo that drives reference attention.</summary>
    public string[] Materials { get; init; } = ["albedo"];

    /// <summary>Camera classes of the 2.0 class embedding, or 0 for none.</summary>
    public int CameraClasses { get; init; }

    /// <summary>True for 2.1: a DINO attention after the text attention.</summary>
    public bool UseDino { get; init; }

    /// <summary>True for 2.1: 3D rotary encoding on the multiview attention.</summary>
    public bool UsePoseRope { get; init; }

    /// <summary>Time embedding width, four times the first level.</summary>
    public int TimeEmbeddingDim => BlockOutChannels[0] * 4;

    /// <summary>Number of materials.</summary>
    public int MaterialCount => Materials.Length;

    /// <summary>Paint 2.0 (albedo only, camera class embedding, no DINO or rotary encoding).</summary>
    public static HunyuanPaintConfig V20 => new() { CameraClasses = 49 };

    /// <summary>Paint 2.1 (albedo and metallic-roughness, DINO attention, rotary multiview attention).</summary>
    public static HunyuanPaintConfig V21 => new() { Materials = ["albedo", "mr"], UseDino = true, UsePoseRope = true };
}

using HartsyInference.ThreeD.Paint;

namespace HartsyInference.ThreeD.Models.Hunyuan3D.Paint;

/// <summary>Settings of one Paint run. Null values take the defaults of the checkpoint generation: 2.1 samples 15 UniPC steps at guidance 3, 2.0 samples 30
/// Euler-ancestral steps at guidance 2.</summary>
public sealed record HunyuanPaintOptions
{
    /// <summary>Views to paint, 6 to 12 (2.0 always paints the six fixed views).</summary>
    public int MaxViews { get; init; } = 6;

    /// <summary>Side of the images the diffusion model sees: 512 or 768 for the released checkpoints, any multiple of 64 mechanically.</summary>
    public int ViewResolution { get; init; } = 512;

    /// <summary>Sampler steps, or null for the generation default.</summary>
    public int? Steps { get; init; }

    /// <summary>Guidance scale, or null for the generation default.</summary>
    public float? GuidanceScale { get; init; }

    /// <summary>Seed of the initial noise (it does not reproduce Python's noise).</summary>
    public int Seed { get; init; }

    /// <summary>Side at which geometry maps are rendered and generated views are baked.</summary>
    public int RenderResolution { get; init; } = 2048;

    /// <summary>Side of the baked textures.</summary>
    public int TextureSize { get; init; } = 2048;

    /// <summary>Inpainting radius of the unpainted texels.</summary>
    public int InpaintRadius { get; init; } = 3;

    /// <summary>Back-projection constants.</summary>
    public PaintBakeOptions Bake { get; init; } = new();
}

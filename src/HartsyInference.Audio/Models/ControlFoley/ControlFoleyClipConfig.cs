namespace HartsyInference.Audio.Models.ControlFoley;

/// <summary>Geometry of the open_clip CLIP used by ControlFoley (<c>apple/DFN5B-CLIP-ViT-H-14-384</c>, QuickGELU).</summary>
public sealed record ControlFoleyClipConfig
{
    /// <summary>Joint embedding width (output of <c>visual.proj</c>).</summary>
    public required int EmbedDim { get; init; }

    /// <summary>Text transformer width.</summary>
    public required int TextWidth { get; init; }

    /// <summary>Text transformer depth.</summary>
    public required int TextLayers { get; init; }

    /// <summary>Text attention heads.</summary>
    public required int TextHeads { get; init; }

    /// <summary>Token embedding rows.</summary>
    public required int VocabSize { get; init; }

    /// <summary>Text context length.</summary>
    public int ContextLength { get; init; } = ControlFoleyClipTokenizer.ContextLength;

    /// <summary>Vision transformer width.</summary>
    public required int VisionWidth { get; init; }

    /// <summary>Vision transformer depth.</summary>
    public required int VisionLayers { get; init; }

    /// <summary>Vision attention heads.</summary>
    public required int VisionHeads { get; init; }

    /// <summary>Patch edge in pixels.</summary>
    public required int PatchSize { get; init; }

    /// <summary>Edge of the frames fed to <see cref="ControlFoleyClip.EncodeImages"/>; the patch grid is <c>InputSize / PatchSize</c> rounded down, as the strided patch convolution does.</summary>
    public required int InputSize { get; init; }

    /// <summary>MLP expansion ratio of both towers.</summary>
    public int MlpRatio { get; init; } = 4;

    /// <summary>LayerNorm epsilon.</summary>
    public float LayerNormEps { get; init; } = 1e-5f;

    /// <summary>Patch grid edge.</summary>
    public int GridSize => InputSize / PatchSize;

    /// <summary>The released DFN5B ViT-H/14 checkpoint fed with 384-pixel frames (27x27 patches over the 378-pixel position table).</summary>
    public static ControlFoleyClipConfig Dfn5bViTH14 { get; } = new()
    {
        EmbedDim = 1024, TextWidth = 1024, TextLayers = 24, TextHeads = 16, VocabSize = 49408,
        VisionWidth = 1280, VisionLayers = 32, VisionHeads = 16, PatchSize = 14, InputSize = 384,
    };
}

namespace SharpInference.Diffusion.Models.TextEncoders;

/// <summary>Configuration for CLIP text encoder. Covers ViT-L/14 (SD1.5) and ViT-bigG (SDXL second encoder).</summary>
public record ClipTextEncoderConfig
{
    /// <summary>Hidden size of the transformer layers.</summary>
    public int HiddenSize { get; init; } = 768;

    /// <summary>Size of the MLP intermediate layer (4x hidden for standard CLIP).</summary>
    public int IntermediateSize { get; init; } = 3072;

    /// <summary>Number of transformer layers.</summary>
    public int NumLayers { get; init; } = 12;

    /// <summary>Number of attention heads.</summary>
    public int NumHeads { get; init; } = 12;

    /// <summary>Maximum position embedding length (77 for CLIP).</summary>
    public int MaxPositionEmbeddings { get; init; } = 77;

    /// <summary>Vocabulary size (49408 for OpenAI CLIP).</summary>
    public int VocabSize { get; init; } = 49408;

    /// <summary>Layer norm epsilon.</summary>
    public float LayerNormEps { get; init; } = 1e-5f;

    /// <summary>Whether to use quick GELU (x * sigmoid(1.702 * x)) instead of exact GELU. OpenAI CLIP uses quick GELU.</summary>
    public bool UseQuickGelu { get; init; } = true;

    /// <summary>Preset for SD1.5 (OpenAI CLIP ViT-L/14).</summary>
    public static ClipTextEncoderConfig Sd15 => new()
    {
        HiddenSize = 768,
        IntermediateSize = 3072,
        NumLayers = 12,
        NumHeads = 12,
        MaxPositionEmbeddings = 77,
        VocabSize = 49408,
        UseQuickGelu = true,
    };
}

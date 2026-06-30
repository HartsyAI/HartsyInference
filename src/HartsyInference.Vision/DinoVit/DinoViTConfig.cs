namespace HartsyInference.Vision.DinoVit;

/// <summary>Configuration for a HuggingFace <c>ViTModel</c> (DINOv1 <c>facebook/dino-vitb16</c>) used as an
/// image tokenizer. Distinct from the DINOv2 backbone: standard ViT key layout
/// (<c>layernorm_before/after</c>, <c>intermediate.dense</c>, <c>output.dense</c>), no LayerScale, exact-erf
/// GELU, and bicubic position-embedding interpolation when the conditioning image differs from the native
/// pretraining size.</summary>
public sealed record DinoViTConfig
{
    /// <summary>ViT hidden size.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Transformer layers.</summary>
    public required int NumLayers { get; init; }

    /// <summary>Attention heads.</summary>
    public required int NumHeads { get; init; }

    /// <summary>FFN intermediate size.</summary>
    public required int IntermediateSize { get; init; }

    /// <summary>Conditioning image side in pixels (TripoSR uses 512).</summary>
    public required int ImageSize { get; init; }

    /// <summary>Native pretraining image side (sets the position-embedding grid that gets interpolated).</summary>
    public required int NativeImageSize { get; init; }

    /// <summary>Patch size.</summary>
    public required int PatchSize { get; init; }

    /// <summary>LayerNorm epsilon.</summary>
    public float LayerNormEps { get; init; } = 1e-12f;

    /// <summary>Patches along one side at the conditioning size.</summary>
    public int PatchGrid => ImageSize / PatchSize;

    /// <summary>Patches along one side at the native size (position-embedding grid).</summary>
    public int NativePatchGrid => NativeImageSize / PatchSize;

    /// <summary>Patch tokens at the conditioning size.</summary>
    public int NumPatches => PatchGrid * PatchGrid;

    /// <summary>Token sequence length: CLS + patches.</summary>
    public int SequenceLength => 1 + NumPatches;

    /// <summary><c>facebook/dino-vitb16</c> at TripoSR's 512px conditioning image.</summary>
    public static DinoViTConfig DinoVitB16_512 => new()
    {
        HiddenSize = 768, NumLayers = 12, NumHeads = 12, IntermediateSize = 3072,
        ImageSize = 512, NativeImageSize = 224, PatchSize = 16, LayerNormEps = 1e-12f,
    };
}

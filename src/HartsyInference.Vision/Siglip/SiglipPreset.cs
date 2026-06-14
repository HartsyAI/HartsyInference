namespace HartsyInference.Vision.Siglip;

/// <summary>Configuration record for SigLIP variants. Differs from CLIP in key ways:
/// <list type="bullet">
///   <item><b>No CLS token</b>: position embeddings cover only patch tokens (e.g. <c>196 × 768</c> for 224/16, not <c>197 × 768</c>).</item>
///   <item><b>No pre-layernorm</b>: standard ViT structure (norm inside each block, not before the first block).</item>
///   <item><b>Attention pooling head</b>: a learned <c>probe</c> attends over the patch tokens → single pooled vector → LN → MLP. Replaces CLIP's "<c>[:, 0, :]</c> + post_layernorm + visual_projection" pattern.</item>
///   <item><b>Standard GELU</b> (not Quick-GELU). All SigLIP variants.</item>
///   <item><b>Image normalize</b>: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5] (matching Inception/EfficientNet rather than CLIP's per-channel values).</item>
///   <item><b>Image preprocessing</b>: simple square resize (no shortest-edge / center-crop).</item>
/// </list></summary>
public sealed record SiglipPreset
{
    /// <summary>Hub-style model identifier (e.g. <c>google/siglip-base-patch16-224</c>).</summary>
    public required string Name { get; init; }

    /// <summary>Hidden dim of the ViT (768 base, 1024 large, 1152 so400m).</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Number of ViT layers (12 base, 24 large, 27 so400m).</summary>
    public required int NumLayers { get; init; }

    /// <summary>Number of attention heads (12 base — head_dim = 64).</summary>
    public required int NumHeads { get; init; }

    /// <summary>Intermediate (FFN) size (3072 base).</summary>
    public required int IntermediateSize { get; init; }

    /// <summary>Input image side in pixels (224, 256, 384, 512 are common).</summary>
    public required int ImageSize { get; init; }

    /// <summary>Patch size (typically 16 for base/large, 14 for so400m).</summary>
    public required int PatchSize { get; init; }

    /// <summary>Output embedding dim (= hidden size in SigLIP base since the head MLP output is dim=hidden).</summary>
    public required int EmbeddingDim { get; init; }

    /// <summary>LayerNorm epsilon (1e-6 in SigLIP, smaller than CLIP's 1e-5).</summary>
    public float LayerNormEps { get; init; } = 1e-6f;

    /// <summary>Number of patches (derived).</summary>
    public int NumPatches => (ImageSize / PatchSize) * (ImageSize / PatchSize);

    /// <summary><c>google/siglip-base-patch16-224</c> — entry-level SigLIP. 86M params per tower.</summary>
    public static SiglipPreset Base16_224 => new()
    {
        Name = "google/siglip-base-patch16-224",
        HiddenSize = 768,
        NumLayers = 12,
        NumHeads = 12,
        IntermediateSize = 3072,
        ImageSize = 224,
        PatchSize = 16,
        EmbeddingDim = 768,
    };

    /// <summary><c>google/siglip-large-patch16-384</c> — 305M-params per tower, 384 input.</summary>
    public static SiglipPreset Large16_384 => new()
    {
        Name = "google/siglip-large-patch16-384",
        HiddenSize = 1024,
        NumLayers = 24,
        NumHeads = 16,
        IntermediateSize = 4096,
        ImageSize = 384,
        PatchSize = 16,
        EmbeddingDim = 1024,
    };

    /// <summary><c>google/siglip-so400m-patch14-384</c> — shape-optimized 400M variant, 384/14 grid.</summary>
    public static SiglipPreset So400m14_384 => new()
    {
        Name = "google/siglip-so400m-patch14-384",
        HiddenSize = 1152,
        NumLayers = 27,
        NumHeads = 16,
        IntermediateSize = 4304,
        ImageSize = 384,
        PatchSize = 14,
        EmbeddingDim = 1152,
    };
}

namespace SharpInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for SD3 MMDiT and SD3.5 MMDiT-X transformers. Architecture parameters are derived from depth: hidden_size = 64 * depth, num_heads = depth.</summary>
public record Sd3Config
{
    /// <summary>Number of JointTransformerBlocks (24 for SD3 Medium, 38 for SD3.5 Large).</summary>
    public required int Depth { get; init; }

    /// <summary>Hidden dimension (= 64 * Depth). 1536 for SD3 Medium.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Number of attention heads (= Depth). 24 for SD3 Medium.</summary>
    public required int NumHeads { get; init; }

    /// <summary>Per-head dimension. Always 64 for SD3.</summary>
    public int HeadDim { get; init; } = 64;

    /// <summary>Patch size for the latent→token embedding. Always 2 for SD3.</summary>
    public int PatchSize { get; init; } = 2;

    /// <summary>Number of latent channels (16 for SD3 VAE).</summary>
    public int InChannels { get; init; } = 16;

    /// <summary>Text context dimension before projection (4096 = padded CLIP + T5 feature dim).</summary>
    public int JointAttentionDim { get; init; } = 4096;

    /// <summary>Pooled projection dimension (2048 = CLIP-L 768 + CLIP-G 1280).</summary>
    public int PooledProjectionDim { get; init; } = 2048;

    /// <summary>Maximum positional embedding grid size. 192 for standard SD3.</summary>
    public int PosEmbedMaxSize { get; init; } = 192;

    /// <summary>QK-norm epsilon. Used when UseQkNorm is true.</summary>
    public float QkNormEps { get; init; } = 1e-6f;

    /// <summary>Whether to apply RMSNorm to Q/K before attention dot product. True for SD3.5, optional for SD3 Medium.</summary>
    public bool UseQkNorm { get; init; } = true;

    /// <summary>Indices of layers that use dual self-attention (SD3.5 MMDiT-X). Null for standard SD3 MMDiT.</summary>
    public int[]? DualAttentionLayers { get; init; }

    /// <summary>SD3 Medium preset (2B params, 24 layers, 1536 hidden).</summary>
    public static Sd3Config Medium => new()
    {
        Depth = 24,
        HiddenSize = 1536,
        NumHeads = 24,
        UseQkNorm = false, // Original SD3 Medium release may not have QK-norm
    };

    /// <summary>SD3.5 Medium preset (2.5B params, 24 layers, 1536 hidden, dual attention + QK-norm).</summary>
    public static Sd3Config Medium35 => new()
    {
        Depth = 24,
        HiddenSize = 1536,
        NumHeads = 24,
        UseQkNorm = true,
        DualAttentionLayers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    };

    /// <summary>SD3.5 Large preset (8B params, 38 layers, 2432 hidden, dual attention + QK-norm).</summary>
    public static Sd3Config Large35 => new()
    {
        Depth = 38,
        HiddenSize = 2432,
        NumHeads = 38,
        UseQkNorm = true,
        DualAttentionLayers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    };

    /// <summary>Auto-detects config from a weight tensor shape. Key formula: depth = patch_embed_weight.shape[0] / 64.</summary>
    public static Sd3Config FromWeightShape(int patchEmbedOutChannels)
    {
        int depth = patchEmbedOutChannels / 64;
        return new Sd3Config
        {
            Depth = depth,
            HiddenSize = 64 * depth,
            NumHeads = depth,
        };
    }
}

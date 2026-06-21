namespace HartsyInference.Diffusion.Models.TextEncoders;

/// <summary>Configuration for the Qwen3-VL vision tower (<c>Qwen3VLVisionModel</c>). Verified against
/// <c>Boogu-Image-0.1-Base/mllm/config.json</c> <c>vision_config</c>. A ViT that turns pixel patches into image tokens
/// merged into the Qwen3-VL language stream, with the Qwen3-VL <b>deepstack</b> feature taps and a learned position
/// embedding interpolated to the grid.</summary>
public sealed record Qwen3VlVisionConfig
{
    /// <summary>Number of vision transformer blocks (27).</summary>
    public int Depth { get; init; } = 27;

    /// <summary>Vision hidden size (1152).</summary>
    public int HiddenSize { get; init; } = 1152;

    /// <summary>Attention heads (16). head_dim = HiddenSize / NumHeads = 72.</summary>
    public int NumHeads { get; init; } = 16;

    /// <summary>Per-head dim.</summary>
    public int HeadDim => HiddenSize / NumHeads;

    /// <summary>MLP intermediate size (4304).</summary>
    public int IntermediateSize { get; init; } = 4304;

    /// <summary>Input channels (3, RGB).</summary>
    public int InChannels { get; init; } = 3;

    /// <summary>Spatial patch size (16).</summary>
    public int PatchSize { get; init; } = 16;

    /// <summary>Spatial merge size (2): a 2×2 block of patches merges into one image token.</summary>
    public int SpatialMergeSize { get; init; } = 2;

    /// <summary>Temporal patch size (2): a still image is duplicated to fill one temporal patch.</summary>
    public int TemporalPatchSize { get; init; } = 2;

    /// <summary>Output hidden size after the merger (4096 = LM hidden).</summary>
    public int OutHiddenSize { get; init; } = 4096;

    /// <summary>Number of learned position embeddings (2304); the per-side grid is sqrt of this.</summary>
    public int NumPositionEmbeddings { get; init; } = 2304;

    /// <summary>Vision blocks whose outputs feed the LM deepstack mergers ([8, 16, 24]).</summary>
    public int[] DeepstackVisualIndexes { get; init; } = [8, 16, 24];

    /// <summary>LayerNorm epsilon for the vision norms (1e-6).</summary>
    public float NormEps { get; init; } = 1e-6f;

    /// <summary>Per-side grid count for the learned position embedding = round(sqrt(NumPositionEmbeddings)) = 48.</summary>
    public int NumGridPerSide => (int)Math.Round(Math.Sqrt(NumPositionEmbeddings));

    /// <summary>Flattened patch-embed input dim = InChannels · TemporalPatchSize · PatchSize² = 1536.</summary>
    public int PatchEmbedInDim => InChannels * TemporalPatchSize * PatchSize * PatchSize;

    /// <summary>Boogu-Image / Qwen3-VL-8B vision preset.</summary>
    public static Qwen3VlVisionConfig Qwen3Vl8B => new();
}

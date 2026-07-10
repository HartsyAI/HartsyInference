namespace HartsyInference.Diffusion.Models.TextEncoders;

/// <summary>Configuration for the Qwen2.5-VL vision tower (<c>Qwen2_5_VLVisionModel</c>). Unlike the Qwen3-VL tower
/// (<see cref="Qwen3VlVisionConfig"/>), Qwen2.5-VL uses RMSNorm blocks, a SwiGLU MLP, no learned position table
/// (2D RoPE only), and window attention: full attention only on <see cref="FullAttentionLayers"/>, block-diagonal
/// windows of <see cref="WindowSize"/> pixels elsewhere.</summary>
public sealed record Qwen25VlVisionConfig
{
    /// <summary>Vision hidden size (1280).</summary>
    public int HiddenSize { get; init; } = 1280;

    /// <summary>Number of vision transformer blocks (32).</summary>
    public int NumLayers { get; init; } = 32;

    /// <summary>Attention heads (16). head_dim = HiddenSize / NumHeads = 80.</summary>
    public int NumHeads { get; init; } = 16;

    /// <summary>Per-head dim.</summary>
    public int HeadDim => HiddenSize / NumHeads;

    /// <summary>SwiGLU MLP intermediate size (3420).</summary>
    public int IntermediateSize { get; init; } = 3420;

    /// <summary>Input channels (3, RGB).</summary>
    public int InChannels { get; init; } = 3;

    /// <summary>Spatial patch size (14).</summary>
    public int PatchSize { get; init; } = 14;

    /// <summary>Temporal patch size (2): a still image is duplicated to fill one temporal patch.</summary>
    public int TemporalPatchSize { get; init; } = 2;

    /// <summary>Spatial merge size (2): a 2×2 block of patches merges into one image token.</summary>
    public int SpatialMergeSize { get; init; } = 2;

    /// <summary>Window attention size in pixels (112 = 8×8 patches = 4×4 merge units).</summary>
    public int WindowSize { get; init; } = 112;

    /// <summary>Blocks that attend over all patches; every other block uses windowed block-diagonal attention.</summary>
    public int[] FullAttentionLayers { get; init; } = [7, 15, 23, 31];

    /// <summary>Output hidden size after the merger (3584 = Qwen2.5-VL-7B LM hidden).</summary>
    public int OutHiddenSize { get; init; } = 3584;

    /// <summary>Epsilon for the RMSNorm layers (1e-6).</summary>
    public float RmsNormEps { get; init; } = 1e-6f;

    /// <summary>OpenAI CLIP per-channel normalization mean.</summary>
    public float[] ImageMean { get; init; } = [0.48145466f, 0.4578275f, 0.40821073f];

    /// <summary>OpenAI CLIP per-channel normalization std.</summary>
    public float[] ImageStd { get; init; } = [0.26862954f, 0.26130258f, 0.27577711f];

    /// <summary>Merged units per window side = WindowSize / PatchSize / SpatialMergeSize (= 4).</summary>
    public int WindowPatches => WindowSize / PatchSize / SpatialMergeSize;

    /// <summary>Patch-embed input dim after temporal-summing the conv kernel = InChannels · PatchSize² = 588.</summary>
    public int PatchEmbedInDim => InChannels * PatchSize * PatchSize;

    /// <summary>Qwen2.5-VL-7B-Instruct vision preset (the Qwen-Image / Qwen-Image-Edit text-encoder tower).</summary>
    public static Qwen25VlVisionConfig Qwen2_5_VL_7B => new();
}

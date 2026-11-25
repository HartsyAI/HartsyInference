namespace HartsyInference.LLM.Hy4Flash;

/// <summary>The vision tower and projector shape (<c>vision_config</c>; the checkpoint prefix is <c>vit.</c>).</summary>
public sealed record Hy4FlashVisionConfig
{
    /// <summary>Tower width (<c>hidden_size</c>).</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Tower MLP width (<c>intermediate_size</c>).</summary>
    public required int IntermediateSize { get; init; }

    /// <summary>Tower blocks (<c>num_hidden_layers</c>).</summary>
    public required int NumHiddenLayers { get; init; }

    /// <summary>Tower attention heads (<c>num_attention_heads</c>).</summary>
    public required int NumAttentionHeads { get; init; }

    /// <summary>Projector output width, the text width (<c>out_hidden_size</c>).</summary>
    public required int OutHiddenSize { get; init; }

    /// <summary>Patch edge in pixels (<c>patch_size</c>).</summary>
    public required int PatchSize { get; init; }

    /// <summary>Merge window edge of the projector (<c>spatial_merge_size</c>).</summary>
    public required int SpatialMergeSize { get; init; }

    /// <summary>Largest image edge in pixels; sets the learned position grid (<c>max_image_size</c>).</summary>
    public required int MaxImageSize { get; init; }

    /// <summary>Norm epsilon (<c>rms_norm_eps</c>).</summary>
    public double RmsNormEps { get; init; } = 1e-5;

    /// <summary>Learned positions per image edge.</summary>
    public int PositionEdge => MaxImageSize / PatchSize;

    /// <summary>Rows of the position table: the class row plus one per grid cell.</summary>
    public int PositionRows => PositionEdge * PositionEdge + 1;
}

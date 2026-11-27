namespace HartsyInference.LLM.Glm5Next;

/// <summary>Shape of the <c>glm5_next_vision</c> tower and the image/video placeholder ids around it.</summary>
public sealed record Glm5NextVisionConfig
{
    /// <summary>Transformer blocks.</summary>
    public required int Depth { get; init; }

    /// <summary>Tower width.</summary>
    public required int Hidden { get; init; }

    /// <summary>Attention heads.</summary>
    public required int Heads { get; init; }

    /// <summary>Gated MLP inner width.</summary>
    public required int Intermediate { get; init; }

    /// <summary>Square patch edge in pixels.</summary>
    public required int PatchSize { get; init; }

    /// <summary>Frames per temporal patch.</summary>
    public required int TemporalPatchSize { get; init; }

    /// <summary>Spatial merge factor per side; the merger folds <c>merge²</c> patches into one token.</summary>
    public required int SpatialMergeSize { get; init; }

    /// <summary>Width of the rows the merger hands the language model (<c>out_hidden_size</c>).</summary>
    public required int OutHidden { get; init; }

    /// <summary>Inner width of the merger's gated MLP (<c>projection_intermediate_size</c>).</summary>
    public required int MergerIntermediate { get; init; }

    /// <summary>RMSNorm epsilon.</summary>
    public required float RmsNormEps { get; init; }

    /// <summary>Clamp of the gated MLP (<c>swiglu_limit</c>).</summary>
    public required float SwigluLimit { get; init; }

    /// <summary>Per-head width.</summary>
    public int HeadDim => Hidden / Heads;

    /// <summary>Refuses a config the tower cannot run: the width must split evenly into heads, and each head's width must be a multiple of 4 for the
    /// two-axis rope (each axis takes half the head, split into pairs).</summary>
    /// <exception cref="InvalidDataException">The widths do not divide as the tower needs.</exception>
    public void Validate()
    {
        if (Hidden <= 0 || Heads <= 0 || Hidden % Heads != 0)
            throw new InvalidDataException($"Vision hidden_size {Hidden} must be a positive multiple of num_heads {Heads}.");
        if (HeadDim % 4 != 0)
            throw new InvalidDataException($"Vision head width {HeadDim} (hidden_size / num_heads) must be a multiple of 4 for the 2-D rope.");
    }
}

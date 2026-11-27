namespace HartsyInference.LLM.Kimi.Vision;

/// <summary>MoonViT-V2 tower and patch-merger geometry, from the checkpoint's <c>vision_config</c>.</summary>
public sealed record class KimiVisionConfig
{
    /// <summary>The published K3.1 values (<c>vision_config</c> in <c>config.json</c>, read 2026-11-07).</summary>
    public static KimiVisionConfig Published { get; } = new()
    {
        Hidden = 1024,
        Heads = 12,
        QkvHidden = 1536,
        MlpDim = 4096,
        Layers = 27,
        PatchSize = 14,
        MergeSize = 2,
        InitPosGrid = 64,
        TextHidden = 7168,
        ProjectorEpsilon = 1e-5f,
        BlockNormEpsilon = 1.1920929e-7f,
        RopeTheta = 10000f,
        ImageSize = 448,
    };

    /// <summary>Channels of the tower's hidden state (<c>vt_hidden_size</c>).</summary>
    public required int Hidden { get; init; }

    /// <summary>Attention heads; <see cref="QkvHidden"/> must divide evenly by this.</summary>
    public required int Heads { get; init; }

    /// <summary>Width of the fused q, k and v projections (<c>qkv_hidden_size</c>).</summary>
    public required int QkvHidden { get; init; }

    /// <summary>Feed-forward width (<c>vt_intermediate_size</c>).</summary>
    public required int MlpDim { get; init; }

    /// <summary>Number of encoder blocks (<c>vt_num_hidden_layers</c>).</summary>
    public required int Layers { get; init; }

    /// <summary>Square patch edge in pixels.</summary>
    public required int PatchSize { get; init; }

    /// <summary>Square spatial merge: each output token covers <c>MergeSize²</c> patches.</summary>
    public required int MergeSize { get; init; }

    /// <summary>Side of the learned position table (<c>init_pos_emb_height</c>, <c>init_pos_emb_width</c>).</summary>
    public required int InitPosGrid { get; init; }

    /// <summary>Width of the text decoder's hidden state that the projector emits.</summary>
    public required int TextHidden { get; init; }

    /// <summary>Epsilon of the projector's post-norm (<c>projector_ln_eps</c>).</summary>
    public required float ProjectorEpsilon { get; init; }

    /// <summary>Epsilon of the tower's RMS norms. The checkpoint does not store it: it is the PyTorch default for the dtype the tower runs in,
    /// and the value here is the F32 one.</summary>
    public required float BlockNormEpsilon { get; init; }

    /// <summary>Base of the rotary frequencies.</summary>
    public required float RopeTheta { get; init; }

    /// <summary>Side in pixels of the square image the preferred-size seam reports.</summary>
    public required int ImageSize { get; init; }

    /// <summary>Per-head width of the attention.</summary>
    public int HeadDim => QkvHidden / Heads;

    /// <summary>Width of one merged token before the projector's first linear.</summary>
    public int MergedDim => Hidden * MergeSize * MergeSize;

    /// <summary>Throws when the geometry cannot describe a tower: rotary pairs need a head width divisible by four, and the merge must tile the patch grid.</summary>
    public void Validate()
    {
        if (Hidden <= 0 || Heads <= 0 || QkvHidden <= 0 || MlpDim <= 0 || TextHidden <= 0 || InitPosGrid <= 0)
            throw new ArgumentOutOfRangeException(nameof(Hidden), "Kimi vision dimensions must be positive.");
        if (Layers < 0)
            throw new ArgumentOutOfRangeException(nameof(Layers), "A vision tower has zero or more blocks.");
        if (QkvHidden % Heads != 0 || HeadDim % 4 != 0)
            throw new ArgumentOutOfRangeException(nameof(QkvHidden), $"The head width {HeadDim} must be a multiple of 4.");
        if (PatchSize <= 0 || MergeSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(PatchSize), "Patch and merge sizes must be positive.");
        if (ImageSize <= 0 || ImageSize % (PatchSize * MergeSize) != 0)
            throw new ArgumentOutOfRangeException(nameof(ImageSize), "The preferred image size must be a whole number of merge blocks.");
        if (!(RopeTheta > 0f) || !(ProjectorEpsilon > 0f) || !(BlockNormEpsilon > 0f))
            throw new ArgumentOutOfRangeException(nameof(RopeTheta), "Rope base and epsilons must be positive.");
    }
}

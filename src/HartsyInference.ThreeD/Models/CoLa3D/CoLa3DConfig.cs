namespace HartsyInference.ThreeD.Models.CoLa3D;

/// <summary>CoLa3D decomposer configuration (<c>cola3d_*_q20.pt</c>): a 6-block, 768-wide query transformer over a 64³ sparse latent with precomputed Utonia point features.</summary>
public sealed record CoLa3DConfig
{
    /// <summary>Where the initial query tokens come from.</summary>
    public enum QuerySourceKind
    {
        /// <summary>Learned or positional tokens indexed by part slot (<c>K_max</c> of them).</summary>
        Fixed,

        /// <summary>One token per prompt 2D mask (CNN over the mask plus the mask-centre encoding).</summary>
        PromptMask,

        /// <summary>One token per prompt 3D point (Fourier position plus the latent feature at that voxel).</summary>
        PromptPoint,
    }

    /// <summary>Initial token scheme for <see cref="QuerySourceKind.Fixed"/>.</summary>
    public enum QueryInitKind
    {
        /// <summary>Sinusoidal slot index through a two-layer MLP.</summary>
        Positional,

        /// <summary>A free parameter table <c>[1, MaxParts, HiddenDim]</c>.</summary>
        Learnable,
    }

    /// <summary>Which latent the point prompt reads: the raw latent, or the latent fused with Utonia features.</summary>
    public enum PointFeatureSourceKind
    {
        Latent,
        LatentUtonia,
    }

    /// <summary>Mask prompt encoder variant: CNN only, or CNN plus the mask-centre positional term.</summary>
    public enum MaskEncoderKind
    {
        Cnn,
        CnnCenterPe,
    }

    /// <summary>Latent channels per voxel (32 for the shipped checkpoints).</summary>
    public required int LatentChannels { get; init; }

    /// <summary>Latent grid resolution per axis (64 for the shipped checkpoints).</summary>
    public required int LatentResolution { get; init; }

    /// <summary>Precomputed Utonia feature channels per voxel (576).</summary>
    public required int UtoniaFeatureDim { get; init; }

    /// <summary>Transformer width (768).</summary>
    public required int HiddenDim { get; init; }

    /// <summary>Number of query/latent blocks (6).</summary>
    public required int NumLayers { get; init; }

    /// <summary>Attention heads; <c>HiddenDim / NumHeads</c> is the head dimension (64).</summary>
    public required int NumHeads { get; init; }

    /// <summary>Slot count for fixed queries (<c>K_max</c>, 20 for the q20 checkpoints).</summary>
    public required int MaxParts { get; init; }

    /// <summary>Query source of this checkpoint.</summary>
    public required QuerySourceKind QuerySource { get; init; }

    /// <summary>Initial token scheme; only read for <see cref="QuerySourceKind.Fixed"/>.</summary>
    public QueryInitKind QueryInit { get; init; } = QueryInitKind.Positional;

    /// <summary>Latent that the point prompt reads; only read for <see cref="QuerySourceKind.PromptPoint"/>.</summary>
    public PointFeatureSourceKind PointFeatureSource { get; init; } = PointFeatureSourceKind.Latent;

    /// <summary>Mask encoder variant; only read for <see cref="QuerySourceKind.PromptMask"/>.</summary>
    public MaskEncoderKind MaskEncoder { get; init; } = MaskEncoderKind.Cnn;

    /// <summary>Widest hidden channel of the mask CNN (128); its stages are a quarter, a half and the full width.</summary>
    public int MaskEncoderHidden { get; init; } = 128;

    /// <summary>Sinusoidal channels per axis for the mask-centre encoding (128).</summary>
    public int MaskPePerAxis { get; init; } = 128;

    /// <summary>Random Fourier features per coordinate for the point prompt (64).</summary>
    public int PointFourierFeatures { get; init; } = 64;

    /// <summary>Feed-forward expansion ratio (4).</summary>
    public double MlpRatio { get; init; } = 4.0;

    /// <summary>Head dimension, valid after <see cref="Validate"/>.</summary>
    public int HeadDim => HiddenDim / NumHeads;

    /// <summary>Feed-forward width, <c>(int)(HiddenDim * MlpRatio)</c>.</summary>
    public int MlpHidden => (int)(HiddenDim * MlpRatio);

    /// <summary>Throws <see cref="ArgumentOutOfRangeException"/> or <see cref="ArgumentException"/> for any combination the decomposer cannot run.</summary>
    public void Validate()
    {
        Positive(LatentChannels, nameof(LatentChannels));
        Positive(LatentResolution, nameof(LatentResolution));
        Positive(UtoniaFeatureDim, nameof(UtoniaFeatureDim));
        Positive(HiddenDim, nameof(HiddenDim));
        Positive(NumLayers, nameof(NumLayers));
        Positive(NumHeads, nameof(NumHeads));
        Positive(MaxParts, nameof(MaxParts));
        if (HiddenDim % NumHeads != 0)
            throw new ArgumentException($"HiddenDim {HiddenDim} must be divisible by NumHeads {NumHeads}.", nameof(NumHeads));
        if (HeadDim % 2 != 0 || HeadDim < 6)
            throw new ArgumentException($"Head dimension {HeadDim} must be even and at least 6 (three RoPE axes, one pair each).", nameof(NumHeads));
        if (!double.IsFinite(MlpRatio) || MlpRatio <= 0.0 || MlpHidden < 1)
            throw new ArgumentOutOfRangeException(nameof(MlpRatio), MlpRatio, "MlpRatio must be finite and give a positive feed-forward width.");
        if (MaskEncoderHidden < 4 || MaskEncoderHidden % 4 != 0)
            throw new ArgumentOutOfRangeException(nameof(MaskEncoderHidden), MaskEncoderHidden, "Must be a multiple of 4 and at least 4.");
        if (MaskPePerAxis < 2 || MaskPePerAxis % 2 != 0)
            throw new ArgumentOutOfRangeException(nameof(MaskPePerAxis), MaskPePerAxis, "MaskPePerAxis must be even and at least 2.");
        if (PointFourierFeatures < 1)
            throw new ArgumentOutOfRangeException(nameof(PointFourierFeatures), PointFourierFeatures, "PointFourierFeatures must be at least 1.");
    }

    private static void Positive(int value, string name)
    {
        if (value < 1)
            throw new ArgumentOutOfRangeException(name, value, $"{name} must be at least 1.");
    }
}

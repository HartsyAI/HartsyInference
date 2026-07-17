namespace HartsyInference.Vision.Detection;

/// <summary>RT-DETR (Real-Time DEtection TRansformer) architecture configuration, matching the
/// <c>PekingU/rtdetr_r*vd</c> HuggingFace checkpoints: a ResNet-vd CNN backbone, a hybrid encoder
/// (AIFI transformer + CCFM convolutional fusion), and a deformable-attention two-stage decoder.
/// The default preset is <see cref="R18vd"/> (ResNet-18vd, 3 decoder layers).</summary>
public sealed record RtDetrConfig
{
    /// <summary>Variant identifier (e.g. <c>"rtdetr_r18vd"</c>) — logging / asset lookup only.</summary>
    public required string Name { get; init; }

    /// <summary>Transformer hidden dimension (<c>d_model</c> / <c>encoder_hidden_dim</c>, 256).</summary>
    public int HiddenDim { get; init; } = 256;

    /// <summary>Attention heads in the AIFI encoder and the decoder (8).</summary>
    public int NumHeads { get; init; } = 8;

    /// <summary>Deformable-attention decoder layer count (3 for r18, 6 for r50).</summary>
    public int NumDecoderLayers { get; init; } = 3;

    /// <summary>Object queries selected from the encoder memory (300).</summary>
    public int NumQueries { get; init; } = 300;

    /// <summary>Detection classes (80 for COCO).</summary>
    public int NumClasses { get; init; } = 80;

    /// <summary>Multi-scale feature levels the decoder samples over (3).</summary>
    public int NumLevels { get; init; } = 3;

    /// <summary>Sampling points per (head, level) in deformable attention (<c>decoder_n_points</c>, 4).</summary>
    public int NumPoints { get; init; } = 4;

    /// <summary>Feed-forward inner dimension of the AIFI + decoder FFNs (1024).</summary>
    public int FeedForwardDim { get; init; } = 1024;

    /// <summary>Backbone residual-stage output channels, length 4 (<c>[64,128,256,512]</c> for r18).</summary>
    public required IReadOnlyList<int> BackboneHiddenSizes { get; init; }

    /// <summary>Basic blocks per backbone stage (<c>[2,2,2,2]</c> for r18).</summary>
    public required IReadOnlyList<int> BackboneDepths { get; init; }

    /// <summary>Stem output channels after the deep-stem embedder (64).</summary>
    public int EmbeddingSize { get; init; } = 64;

    /// <summary>The three backbone feature-map channel counts fed to the encoder input projection
    /// (<c>[128,256,512]</c> — strides 8/16/32).</summary>
    public required IReadOnlyList<int> EncoderInChannels { get; init; }

    /// <summary>CSPRepLayer hidden-channel fraction of the hidden dim (0.5 → 128).</summary>
    public float HiddenExpansion { get; init; } = 0.5f;

    /// <summary>2D sincos position-embedding temperature (10000).</summary>
    public float PositionalEncodingTemperature { get; init; } = 10000f;

    /// <summary>Anchor base grid size for two-stage query selection (0.05).</summary>
    public float AnchorGridSize { get; init; } = 0.05f;

    /// <summary>LayerNorm epsilon across the transformer blocks (1e-5).</summary>
    public float LayerNormEps { get; init; } = 1e-5f;

    /// <summary>BatchNorm epsilon used when folding BN into conv weights (1e-5).</summary>
    public float BatchNormEps { get; init; } = 1e-5f;

    /// <summary>Per-head channel width — <c>HiddenDim / NumHeads</c>.</summary>
    public int HeadDim => HiddenDim / NumHeads;

    /// <summary>CSPRepLayer / RepVGG hidden channel count — <c>round(HiddenDim · HiddenExpansion)</c>.</summary>
    public int RepHiddenChannels => (int)(HiddenDim * HiddenExpansion);

    /// <summary>PekingU/rtdetr_r18vd preset — ResNet-18vd, 3 decoder layers.</summary>
    public static RtDetrConfig R18vd => new()
    {
        Name = "rtdetr_r18vd",
        HiddenDim = 256,
        NumHeads = 8,
        NumDecoderLayers = 3,
        NumQueries = 300,
        NumClasses = 80,
        NumLevels = 3,
        NumPoints = 4,
        FeedForwardDim = 1024,
        BackboneHiddenSizes = [64, 128, 256, 512],
        BackboneDepths = [2, 2, 2, 2],
        EmbeddingSize = 64,
        EncoderInChannels = [128, 256, 512],
        HiddenExpansion = 0.5f,
    };

    /// <summary>Validates the interdependent shape constraints; throws <see cref="ArgumentException"/> on a bad combo.</summary>
    public void Validate()
    {
        if (HiddenDim % NumHeads != 0)
            throw new ArgumentException($"HiddenDim ({HiddenDim}) must be divisible by NumHeads ({NumHeads}).");
        if (HiddenDim % 4 != 0)
            throw new ArgumentException($"HiddenDim ({HiddenDim}) must be divisible by 4 for the 2D sincos position embedding.");
        if (NumLevels != 3)
            throw new ArgumentException($"This port fixes NumLevels at 3; got {NumLevels}.");
        if (BackboneHiddenSizes.Count != 4 || BackboneDepths.Count != 4)
            throw new ArgumentException("BackboneHiddenSizes and BackboneDepths must both have length 4.");
        if (EncoderInChannels.Count != 3)
            throw new ArgumentException("EncoderInChannels must have length 3.");
        if (NumQueries <= 0 || NumClasses <= 0 || NumPoints <= 0 || NumDecoderLayers <= 0)
            throw new ArgumentException("NumQueries, NumClasses, NumPoints, and NumDecoderLayers must all be positive.");
    }
}

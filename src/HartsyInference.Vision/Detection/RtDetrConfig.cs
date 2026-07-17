namespace HartsyInference.Vision.Detection;

/// <summary>RT-DETR (Real-Time DEtection TRansformer) architecture configuration. Drives the CNN
/// backbone widths, the hybrid-encoder hidden dimension, and the deformable-attention decoder
/// shape (queries, heads, feature levels, sampling points).
/// <para><b>Backbone note.</b> The reference RT-DETR-R50 uses a ResNet-50 (or HGNetv2) backbone;
/// this port builds a structurally-equivalent CSP-style CNN from the shared
/// <see cref="Blocks.ConvBnSilu"/> / <see cref="Blocks.C2f"/> blocks so the three multi-scale
/// feature maps (strides 8/16/32) have the right channel progression. Real ResNet weights would
/// need a distinct backbone module; this choice is documented and parity-unverified.</para></summary>
public sealed record RtDetrConfig
{
    /// <summary>Variant identifier (e.g. <c>"rtdetr-r50"</c>) — logging / asset lookup only.</summary>
    public required string Name { get; init; }

    /// <summary>Backbone stage output channels, length 5: <c>[stem0, stem1, C3, C4, C5]</c>. C3/C4/C5
    /// (indices 2/3/4) are the three feature maps fed to the neck at strides 8/16/32.</summary>
    public required IReadOnlyList<int> BackboneChannels { get; init; }

    /// <summary>C2f bottleneck repeat count per backbone stage (the depth knob).</summary>
    public int BackboneRepeat { get; init; } = 2;

    /// <summary>Transformer hidden dimension the neck projects every level to (256 in the reference). Must be divisible by 4 (2D sincos pos-embed) and by <see cref="NumHeads"/>.</summary>
    public int HiddenDim { get; init; } = 256;

    /// <summary>Attention heads in both the AIFI encoder block and the decoder (8 in the reference).</summary>
    public int NumHeads { get; init; } = 8;

    /// <summary>Deformable-attention decoder layer count (6 in the reference).</summary>
    public int NumDecoderLayers { get; init; } = 6;

    /// <summary>Object queries selected from the encoder memory (300 in the reference).</summary>
    public int NumQueries { get; init; } = 300;

    /// <summary>Detection classes (80 for COCO).</summary>
    public int NumClasses { get; init; } = 80;

    /// <summary>Multi-scale feature levels the decoder samples over (3 — the neck's P3/P4/P5 outputs).</summary>
    public int NumLevels { get; init; } = 3;

    /// <summary>Sampling points per (head, level) in deformable attention (4 in the reference).</summary>
    public int NumPoints { get; init; } = 4;

    /// <summary>Feed-forward inner dimension of the AIFI + decoder FFNs (1024 in the reference).</summary>
    public int FeedForwardDim { get; init; } = 1024;

    /// <summary>LayerNorm epsilon shared across the encoder + decoder transformer blocks.</summary>
    public float LayerNormEps { get; init; } = 1e-5f;

    /// <summary>Per-head channel width — <c>HiddenDim / NumHeads</c>.</summary>
    public int HeadDim => HiddenDim / NumHeads;

    /// <summary>The three neck-input backbone channels (C3, C4, C5) at strides 8/16/32.</summary>
    public (int C3, int C4, int C5) NeckInputChannels =>
        (BackboneChannels[2], BackboneChannels[3], BackboneChannels[4]);

    /// <summary>RT-DETR-R50 preset — the flagship ResNet-50-scale variant (hidden 256, 6 decoder layers).</summary>
    public static RtDetrConfig R50 => new()
    {
        Name = "rtdetr-r50",
        BackboneChannels = [32, 64, 512, 1024, 2048],
        BackboneRepeat = 2,
        HiddenDim = 256,
        FeedForwardDim = 1024,
    };

    /// <summary>RT-DETR-R18 preset — the lightweight ResNet-18-scale variant (narrower backbone, 3 decoder layers).</summary>
    public static RtDetrConfig R18 => new()
    {
        Name = "rtdetr-r18",
        BackboneChannels = [16, 32, 128, 256, 512],
        BackboneRepeat = 1,
        HiddenDim = 256,
        NumDecoderLayers = 3,
        FeedForwardDim = 1024,
    };

    /// <summary>Validates the interdependent shape constraints; throws <see cref="ArgumentException"/> on a bad combo.</summary>
    public void Validate()
    {
        if (BackboneChannels.Count != 5)
            throw new ArgumentException($"BackboneChannels must have length 5 [stem0,stem1,C3,C4,C5]; got {BackboneChannels.Count}.");
        if (HiddenDim % NumHeads != 0)
            throw new ArgumentException($"HiddenDim ({HiddenDim}) must be divisible by NumHeads ({NumHeads}).");
        if (HiddenDim % 4 != 0)
            throw new ArgumentException($"HiddenDim ({HiddenDim}) must be divisible by 4 for the 2D sincos position embedding.");
        if (NumLevels != 3)
            throw new ArgumentException($"This port fixes NumLevels at 3 (P3/P4/P5); got {NumLevels}.");
        if (NumQueries <= 0 || NumClasses <= 0 || NumPoints <= 0 || NumDecoderLayers <= 0)
            throw new ArgumentException("NumQueries, NumClasses, NumPoints, and NumDecoderLayers must all be positive.");
    }
}

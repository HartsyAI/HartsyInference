namespace HartsyInference.Vision.Detection;

/// <summary>Configuration for the open-vocabulary Grounding DINO detector: the shared transformer width, the
/// text-tower width, the cross-modality feature enhancer / decoder depth, and the CNN backbone channel schedule.
/// The <see cref="Tiny"/> and <see cref="Base"/> presets mirror the two published checkpoints' hidden sizes; the
/// backbone here is a lightweight CNN stand-in (see <c>GroundingDinoBackbone</c>), so its channel schedule is
/// ours, not Swin's.</summary>
public sealed record GroundingDinoConfig
{
    /// <summary>Shared transformer / query embedding width (Grounding DINO's <c>d_model</c>, 256 in both checkpoints).</summary>
    public int ImageHiddenDim { get; init; } = 256;

    /// <summary>Text-tower embedding width returned by <see cref="IGroundingTextEncoder"/> (BERT-base = 768).</summary>
    public int TextDim { get; init; } = 768;

    /// <summary>Attention head count shared by the enhancer, decoder, and cross-modality attention.</summary>
    public int NumHeads { get; init; } = 8;

    /// <summary>Multi-scale feature levels the backbone emits (Grounding DINO uses 4; the CNN stand-in emits 3).</summary>
    public int NumFeatureLevels { get; init; } = 3;

    /// <summary>Cross-modality feature-enhancer layer count (6 in both checkpoints).</summary>
    public int NumEnhancerLayers { get; init; } = 6;

    /// <summary>Cross-modality decoder layer count (6 in both checkpoints).</summary>
    public int NumDecoderLayers { get; init; } = 6;

    /// <summary>Object-query count fed to the decoder after language-guided query selection (900 in production).
    /// Clamped down to the available image-token count when the feature map is smaller than this.</summary>
    public int NumQueries { get; init; } = 900;

    /// <summary>Feed-forward inner width for the image stream of the enhancer and for the decoder.</summary>
    public int FeedForwardDim { get; init; } = 2048;

    /// <summary>Feed-forward inner width for the text stream of the enhancer (kept smaller — the text stream is narrow).</summary>
    public int TextFeedForwardDim { get; init; } = 1024;

    /// <summary>Upper bound on the tokenized caption length (Grounding DINO caps at 256).</summary>
    public int MaxTextTokens { get; init; } = 256;

    /// <summary>Per-downsample output channels of the CNN backbone stand-in. Five stride-2 stages; the last three
    /// (strides 8 / 16 / 32) are the feature levels that get projected to <see cref="ImageHiddenDim"/>.</summary>
    public int[] BackboneChannels { get; init; } = [64, 128, 256, 512, 512];

    /// <summary>Grounding-DINO-Tiny preset (Swin-T / BERT-base widths).</summary>
    public static GroundingDinoConfig Tiny => new();

    /// <summary>Grounding-DINO-Base preset — wider decoder feed-forward, same transformer/text widths as Tiny in
    /// the public release (the base checkpoint scales the Swin backbone, which our CNN stand-in does not model).</summary>
    public static GroundingDinoConfig Base => new()
    {
        FeedForwardDim = 2048,
        BackboneChannels = [96, 192, 384, 768, 768],
    };
}

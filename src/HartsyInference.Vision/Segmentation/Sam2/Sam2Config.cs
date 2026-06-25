namespace HartsyInference.Vision.Segmentation.Sam2;

/// <summary>Configuration for a SAM 2 (Segment Anything Model 2) network: the Hiera image encoder, the
/// prompt encoder, the two-way mask decoder, and (for video) the memory-attention bank. Defaults match
/// the <c>sam2.1_hiera_base_plus</c> release; presets cover the tiny/small/large variants.</summary>
public sealed record Sam2Config
{
    /// <summary>Square model input resolution (SAM 2 uses 1024).</summary>
    public int ImageSize { get; init; } = 1024;

    /// <summary>Image-encoder embedding dim at the highest-resolution stage.</summary>
    public int EncoderEmbedDim { get; init; } = 112;

    /// <summary>Number of Hiera stages (blocks per stage given by <see cref="EncoderStageDepths"/>).</summary>
    public int[] EncoderStageDepths { get; init; } = [2, 3, 16, 3];

    /// <summary>Attention heads at the first stage (doubles each stage in Hiera).</summary>
    public int EncoderNumHeads { get; init; } = 2;

    /// <summary>Window sizes per stage for windowed attention (0 = global).</summary>
    public int[] EncoderWindowSizes { get; init; } = [8, 4, 14, 7];

    /// <summary>Channel dim shared by the prompt encoder, mask decoder, and image-feature projection.</summary>
    public int PromptEmbedDim { get; init; } = 256;

    /// <summary>Number of mask outputs the decoder produces per prompt (multi-mask ambiguity resolution).</summary>
    public int NumMultimaskOutputs { get; init; } = 3;

    /// <summary>Transformer depth of the two-way mask decoder.</summary>
    public int DecoderDepth { get; init; } = 2;

    /// <summary>Attention heads in the mask decoder / memory attention.</summary>
    public int DecoderNumHeads { get; init; } = 8;

    /// <summary>Number of past frames retained in the video memory bank.</summary>
    public int MemoryBankSize { get; init; } = 7;

    /// <summary>sam2.1 hiera tiny.</summary>
    public static Sam2Config HieraTiny => new()
    {
        EncoderEmbedDim = 96,
        EncoderStageDepths = [1, 2, 7, 2],
        EncoderNumHeads = 1,
        EncoderWindowSizes = [8, 4, 14, 7],
    };

    /// <summary>sam2.1 hiera small.</summary>
    public static Sam2Config HieraSmall => new()
    {
        EncoderEmbedDim = 96,
        EncoderStageDepths = [1, 2, 11, 2],
        EncoderNumHeads = 1,
        EncoderWindowSizes = [8, 4, 14, 7],
    };

    /// <summary>sam2.1 hiera base+ (default).</summary>
    public static Sam2Config HieraBasePlus => new();

    /// <summary>sam2.1 hiera large.</summary>
    public static Sam2Config HieraLarge => new()
    {
        EncoderEmbedDim = 144,
        EncoderStageDepths = [2, 6, 36, 4],
        EncoderNumHeads = 2,
        EncoderWindowSizes = [8, 4, 16, 8],
    };
}

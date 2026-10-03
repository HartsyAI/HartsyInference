namespace HartsyInference.Audio.Models.IndexTts;

/// <summary>Config for <see cref="IndexTtsConformerEncoder"/> — IndexTTS-1.5's <c>gpt.condition_module</c> (ESPnet Conformer, non-macaron, relative-position attention, conv module, <c>conv2d2</c> subsampling).</summary>
public sealed record IndexTtsConformerConfig
{
    public required int InputSize { get; init; }     // mel bands (100)
    public required int OutputSize { get; init; }     // 512
    public required int AttentionHeads { get; init; } // 8
    public required int LinearUnits { get; init; }    // 2048
    public required int NumBlocks { get; init; }      // 6
    public int ConvKernel { get; init; } = 15;
    public float LayerNormEps { get; init; } = 1e-5f;

    /// <summary>IndexTTS-1.5's <c>condition_module</c> preset from <c>config.yaml</c>.</summary>
    public static IndexTtsConformerConfig V1_5 => new()
    {
        InputSize = 100,
        OutputSize = 512,
        AttentionHeads = 8,
        LinearUnits = 2048,
        NumBlocks = 6,
    };
}

using HartsyInference.Core.Exceptions;

namespace HartsyInference.ThreeD.Models.UniRig;

/// <summary>Geometry of the OPT decoder in the skeleton checkpoint (<c>unirig_ar_350m_1024_81920_float32</c>).</summary>
public sealed record UniRigDecoderConfig
{
    public required int HiddenSize { get; init; }
    public required int NumLayers { get; init; }
    public required int NumHeads { get; init; }
    public required int FfnSize { get; init; }
    public required int VocabSize { get; init; }

    /// <summary>Usable positions; the checkpoint's position table holds two more rows (OPT's offset).</summary>
    public required int MaxPositions { get; init; }

    /// <summary>The shipped decoder: 24 layers of width 1024, 16 heads, FFN 4096, 267 tokens, 3,076 positions.</summary>
    public static UniRigDecoderConfig Default => new()
    {
        HiddenSize = 1024,
        NumLayers = 24,
        NumHeads = 16,
        FfnSize = 4096,
        VocabSize = UniRigTokenizer.VocabSize,
        MaxPositions = 3076,
    };

    /// <summary>Throws for a config the decoder cannot build.</summary>
    public void Validate()
    {
        if (HiddenSize <= 0 || NumLayers <= 0 || NumHeads <= 0 || HiddenSize % NumHeads != 0 || FfnSize <= 0 || VocabSize <= 0 || MaxPositions <= 2)
        {
            throw new HartsyInferenceException(
                $"UniRig decoder config is invalid: hidden {HiddenSize}, layers {NumLayers}, heads {NumHeads}, ffn {FfnSize}, "
                + $"vocab {VocabSize}, positions {MaxPositions}.");
        }
    }
}

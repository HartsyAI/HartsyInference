using HartsyInference.Audio.Models.Codecs.Mimi;

namespace HartsyInference.Audio.Models.BreezeTts;

/// <summary>Configuration contract for Breeze TTS 2. The model reuses the shared Mimi codec but has
/// its own text encoder, backbone, depth decoder, and conditioning modes.</summary>
public sealed record BreezeTts2Config
{
    public int HiddenSize { get; init; } = 2_048;
    public int NumBackboneLayers { get; init; } = 28;
    public int NumBackboneHeads { get; init; } = 16;
    public int NumBackboneKeyValueHeads { get; init; } = 8;
    public int BackboneIntermediateSize { get; init; } = 6_144;
    public int NumDepthDecoderLayers { get; init; } = 12;
    public int DepthDecoderHiddenSize { get; init; } = 1_024;
    public int DepthDecoderIntermediateSize { get; init; } = 8_192;
    public int DepthDecoderHeads { get; init; } = 8;
    public int DepthDecoderKeyValueHeads { get; init; } = 2;
    public int DepthDecoderMaxPositions { get; init; } = 33;
    public int TextEncoderHiddenSize { get; init; } = 1_152;
    public int TextEncoderLayers { get; init; } = 26;
    public int TextEncoderVocabSize { get; init; } = 262_158;
    public int NumCodebooks { get; init; } = 16;
    public MimiConfig Codec { get; init; } = MimiConfig.Mimi24kHzDsm;
    public float CodecFrameRate { get; init; } = 12.5f;
    public int AudioTokenId { get; init; } = 262_144;
    public int AudioEosTokenId { get; init; } = 262_145;
    public int AudioVocabSize { get; init; } = 2_051;
    public bool SupportsVoiceDesign { get; init; } = true;
    public bool SupportsVoiceDirection { get; init; } = true;
    public bool SupportsVoiceClone { get; init; } = true;
    public bool SupportsEnglishAndChinese { get; init; } = true;

    public static BreezeTts2Config Default => new();
}

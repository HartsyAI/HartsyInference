namespace HartsyInference.Audio.Models.BreezeTts;

/// <summary>Configuration contract for Breeze TTS 2. The model reuses the Qwen3 12 Hz codec family but has
/// its own text encoder, backbone, depth decoder, and conditioning modes.</summary>
public sealed record BreezeTts2Config
{
    public int HiddenSize { get; init; } = 2_048;
    public int NumBackboneLayers { get; init; } = 28;
    public int NumBackboneHeads { get; init; } = 16;
    public int NumBackboneKeyValueHeads { get; init; } = 8;
    public int NumDepthDecoderLayers { get; init; } = 12;
    public int NumCodebooks { get; init; } = 16;
    public int CodecFrameRate { get; init; } = 12;
    public int SampleRate { get; init; } = 24_000;
    public bool SupportsVoiceDesign { get; init; } = true;
    public bool SupportsVoiceDirection { get; init; } = true;
    public bool SupportsVoiceClone { get; init; } = true;
    public bool SupportsEnglishAndChinese { get; init; } = true;

    public static BreezeTts2Config Default => new();
}

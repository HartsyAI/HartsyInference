namespace HartsyInference.Audio.Models.OpenVoice;

/// <summary>Configuration for <see cref="OpenVoiceSpeakerEncoder"/>. Defaults follow the OpenVoice tone-color
/// extractor: an 80-mel input, a stride-2 Conv1d stack, and a 256-d (gin) output. The channel schedule is a
/// structurally-faithful stand-in (reconcile against the downloaded checkpoint).</summary>
public sealed record OpenVoiceSpeakerConfig
{
    public int SampleRate { get; init; } = 22_050;
    public int NumMels { get; init; } = 80;

    /// <summary>Per-layer output channels for the Conv1d stack.</summary>
    public IReadOnlyList<int> Channels { get; init; } = [128, 256, 256, 256];

    public int KernelSize { get; init; } = 3;
    public int Stride { get; init; } = 2;
    public int NormGroups { get; init; } = 8;

    /// <summary>Speaker-conditioning channel count (the tone converter's <c>g</c> width).</summary>
    public int Gin { get; init; } = 256;

    public static OpenVoiceSpeakerConfig Default => new();
}

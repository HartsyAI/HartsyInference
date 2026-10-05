namespace HartsyInference.Audio.Models.ControlFoley;

/// <summary>Pipeline contract for ControlFoley video-to-audio generation.</summary>
public sealed record ControlFoleyConfig
{
    public int SampleRate { get; init; } = 44_100;
    public int Channels { get; init; } = 1;
    public int DefaultFrameRate { get; init; } = 24;
    public bool SupportsTextConditioning { get; init; } = true;
    public bool SupportsVideoConditioning { get; init; } = true;
    public bool SupportsReferenceAudio { get; init; } = true;
    public bool CommercialUseAllowed { get; init; } = false;
    public static ControlFoleyConfig Default => new();
}

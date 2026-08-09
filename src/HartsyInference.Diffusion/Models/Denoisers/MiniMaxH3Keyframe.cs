namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>A first/last-frame keyframe anchor for fl2va.</summary>
public sealed record MiniMaxH3Keyframe
{
    public required int ResolvedFrameIndex { get; init; }
}

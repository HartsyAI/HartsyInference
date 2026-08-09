namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>A reference block for ref2va conditioning.</summary>
public sealed record MiniMaxH3RefBlock
{
    public required string Kind { get; init; }   // "image" | "audio" | "video" | "video_audio"
    public int LatentT { get; init; }
    public int LatentH { get; init; }
    public int LatentW { get; init; }
    public int RefAudioT { get; init; }
}

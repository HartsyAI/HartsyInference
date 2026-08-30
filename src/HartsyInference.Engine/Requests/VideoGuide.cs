namespace HartsyInference.Engine.Requests;

/// <summary>An arbitrary target-relative MiniMax-H3 visual and/or audio conditioning anchor.</summary>
public sealed record VideoGuide
{
    /// <summary>Zero-based target frame. Negative values resolve from the aligned target end, so -1 is the last frame.</summary>
    public required int FrameIndex { get; init; }

    /// <summary>Single visual guide. Exactly one of this and <see cref="Video"/> may be set.</summary>
    public ImageData? Image { get; init; }

    /// <summary>Visual guide clip. Exactly one of this and <see cref="Image"/> may be set.</summary>
    public VideoClip? Video { get; init; }

    /// <summary>Optional audio guide beginning at the same target position.</summary>
    public AudioClip? Audio { get; init; }

    /// <summary>Visual fitting behavior.</summary>
    public VideoGuideFitMode FitMode { get; init; } = VideoGuideFitMode.Cover;
}

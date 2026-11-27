namespace HartsyInference.Engine.Requests;

/// <summary>Two people in one picture, each with their own speech, for a family that declares multi-speaker support (LongCat-Video-Avatar 1.5
/// and 1.6). The request needs an init image that shows them, and no <see cref="VideoRequest.VideoAudioReference"/>. The result's soundtrack is
/// the two recordings combined by <see cref="Timing"/>, as 16 kHz mono.</summary>
/// <remarks>Likeness warning: the output is generated talking people. Use it only with the consent of everyone depicted.</remarks>
public sealed record VideoSpeakers
{
    /// <summary>The first speaker.</summary>
    public required VideoSpeaker First { get; init; }

    /// <summary>The second speaker.</summary>
    public required VideoSpeaker Second { get; init; }

    /// <summary>Whether they talk at once or one after the other.</summary>
    public VideoSpeakerTiming Timing { get; init; } = VideoSpeakerTiming.Parallel;

    /// <summary>Boxes of further people in the picture, who stay silent; null or empty for none. They need both speakers' boxes.</summary>
    public IReadOnlyList<VideoSpeakerBox>? SilentBoxes { get; init; }
}

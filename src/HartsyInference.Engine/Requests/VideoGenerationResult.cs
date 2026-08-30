using HartsyInference.Engine.Planning;

namespace HartsyInference.Engine.Requests;

/// <summary>One video generation's full output: the decoded frames plus the soundtrack that belongs with them, so a model that generates audio jointly with video (LTX-2.3, MiniMax-H3) can hand both to the caller that muxes them.</summary>
public sealed record VideoGenerationResult
{
    /// <summary>Decoded frames in sequence order.</summary>
    public required IReadOnlyList<VideoFrame> Frames { get; init; }

    /// <summary>The audio to mux alongside <see cref="Frames"/>; null for a silent generation.</summary>
    public AudioBuffer? Audio { get; init; }

    /// <summary>Playback rate the pipeline pinned (e.g. matched to a decoded driving clip); null defers to the request's fps / family default. <see cref="Services.VideoService"/> resolves the final value onto this property.</summary>
    public int? Fps { get; init; }

    /// <summary>Resolved checkpoint profile, aligned geometry, sampling values, and component formats used by this run.</summary>
    /// <summary>The exact profile-resolved execution record for MiniMax-H3. Null for legacy families whose final
    /// geometry or schedule is still resolved inside the pipeline rather than by <c>VideoPlan</c>.</summary>
    public VideoExecutionSummary? Execution { get; init; }

    /// <summary>Wraps a frame-only generation.</summary>
    public static VideoGenerationResult FromFrames(IReadOnlyList<VideoFrame> frames) =>
        new VideoGenerationResult { Frames = frames };
}

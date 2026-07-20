using HartsyInference.Conditioning;
using HartsyInference.Video;

namespace HartsyInference.World.Sessions;

/// <summary>The model-specific compute body a <see cref="BackgroundComputeSession"/> drives. Called only from the
/// session's dedicated compute thread, one action at a time — implementations need no internal locking. Segment-based
/// models (Matrix-Game) buffer actions internally and return an empty list until a segment boundary, then return the
/// whole decoded segment.</summary>
public interface IFrameStepper
{
    /// <summary>Output frame width in pixels.</summary>
    int FrameWidth { get; }

    /// <summary>Output frame height in pixels.</summary>
    int FrameHeight { get; }

    /// <summary>Advances the model by one action; returns zero or more finished frames.</summary>
    IReadOnlyList<VideoFrame> Step(in ActionInput action);

    /// <summary>Applies a speed/quality trade-off; takes effect at the next natural boundary (frame or segment).</summary>
    void ApplyQualityProfile(QualityProfile profile);
}

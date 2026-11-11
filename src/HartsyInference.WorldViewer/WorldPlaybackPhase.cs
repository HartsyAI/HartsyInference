namespace HartsyInference.WorldViewer;

/// <summary>Where playback stands, for the viewer's status line.</summary>
public enum WorldPlaybackPhase
{
    /// <summary>No frame has arrived yet; the first chunk is still generating.</summary>
    Generating,

    /// <summary>Frames are playing at the frame rate.</summary>
    Playing,

    /// <summary>Playback ran out of frames and waits for generation to catch up.</summary>
    Rebuffering,

    /// <summary>The stream ended and every frame was shown.</summary>
    Finished,

    /// <summary>The stream failed; frames already buffered still play.</summary>
    Failed,
}

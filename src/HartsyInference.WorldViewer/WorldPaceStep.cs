namespace HartsyInference.WorldViewer;

/// <summary>What the presenter does on one clock tick.</summary>
public enum WorldPaceStep
{
    /// <summary>Show the next frame now.</summary>
    Show,

    /// <summary>Keep the frame on screen; the next one is not due yet.</summary>
    Hold,

    /// <summary>Waiting for frames: the start-up buffer or a rebuffer after an underrun has not filled.</summary>
    Buffering,

    /// <summary>The stream has ended and every buffered frame has been shown.</summary>
    Finished,
}

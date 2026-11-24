namespace HartsyInference.Engine.Requests;

/// <summary>How a video model groups the frames of a <see cref="VisionBatchRequest"/> (<see cref="VisionBatchRequest.FrameBuffering"/>).</summary>
public enum VisionFrameBuffering
{
    /// <summary>Overlapping clips run jointly and are cross-faded: a frame also sees later frames and is returned once its clip is done.</summary>
    Buffered,

    /// <summary>Each frame runs as it arrives, seeing itself and the frames before it through a key/value cache.</summary>
    Unbuffered,
}

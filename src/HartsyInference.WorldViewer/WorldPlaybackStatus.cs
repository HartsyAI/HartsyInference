namespace HartsyInference.WorldViewer;

/// <summary>A snapshot of playback for the viewer: the phase, how far generation runs ahead of the screen, and how fast frames arrive.</summary>
/// <remarks><see cref="DeliveredPerSecond"/> is frames received over the wall time since the first one. Once the buffer is full the
/// stream waits on the viewer, so this reads at the play rate: it falls below that rate only when generation is the bottleneck.</remarks>
public readonly record struct WorldPlaybackStatus(
    WorldPlaybackPhase Phase,
    int Fps,
    int BufferedFrames,
    int BufferCapacity,
    int FramesShown,
    long FramesReceived,
    double DeliveredPerSecond,
    string? Error)
{
    /// <summary>Seconds of generated video waiting to play.</summary>
    public double BufferedSeconds => (double)BufferedFrames / Fps;
}

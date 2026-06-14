namespace HartsyInference.Interactive.Sessions;

/// <summary>Point-in-time health snapshot of an <see cref="IInteractiveSession"/>.</summary>
/// <param name="P50StepMs">Median model-step latency over the recent window (ms).</param>
/// <param name="P99StepMs">99th-percentile model-step latency over the recent window (ms).</param>
/// <param name="FramesEmitted">Total frames produced since session start.</param>
/// <param name="DroppedFrames">Frames discarded because the consumer fell behind the bounded output queue.</param>
/// <param name="ActionQueueDepth">Actions currently waiting for the compute thread.</param>
/// <param name="FrameQueueDepth">Frames currently waiting for the consumer.</param>
public readonly record struct InteractiveSessionStats(
    double P50StepMs,
    double P99StepMs,
    long FramesEmitted,
    long DroppedFrames,
    int ActionQueueDepth,
    int FrameQueueDepth);

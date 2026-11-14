namespace HartsyInference.Core.Exceptions;

/// <summary>A stage made no progress for far longer than its steps take, which on Windows usually means the device memory the process may use is exhausted and the driver is paging it. Thrown at the next step boundary after the stall, because a step already running cannot be interrupted.</summary>
public sealed class StageStalledException : HartsyInferenceException
{
    /// <summary>The stage that stalled.</summary>
    public string Stage { get; }

    /// <summary>Seconds the stage went without finishing a step when the stall was recorded.</summary>
    public double StalledSeconds { get; }

    /// <summary>Seconds one step of the stage normally takes (the median so far), or 0 when none had finished.</summary>
    public double ExpectedStepSeconds { get; }

    /// <summary>Free device memory in bytes when the stall was recorded, or 0 when it was not known.</summary>
    public long FreeBytes { get; }

    /// <summary>Total device memory in bytes, or 0 when it was not known.</summary>
    public long TotalBytes { get; }

    public StageStalledException(string stage, double stalledSeconds, double expectedStepSeconds, long freeBytes, long totalBytes)
        : base(Describe(stage, stalledSeconds, expectedStepSeconds, freeBytes, totalBytes))
    {
        Stage = stage;
        StalledSeconds = stalledSeconds;
        ExpectedStepSeconds = expectedStepSeconds;
        FreeBytes = freeBytes;
        TotalBytes = totalBytes;
    }

    /// <summary>The message: what stalled, for how long against what, the memory state, and what to try.</summary>
    internal static string Describe(string stage, double stalledSeconds, double expectedStepSeconds, long freeBytes, long totalBytes)
    {
        string normal = expectedStepSeconds > 0 ? $"a step normally takes {expectedStepSeconds:F0} s" : "no step had finished yet";
        string memory = totalBytes > 0 ? $" The device had {freeBytes / (1024 * 1024)} MiB free of {totalBytes / (1024 * 1024)} MiB." : "";
        return $"Stage '{stage}' made no progress for {stalledSeconds:F0} s ({normal}).{memory} On Windows this usually means the process went past the video memory it is given and the driver is paging: "
            + "close other programs that use the GPU, lower the resolution, or set numerics.flowF16Weights to halve the flow weights.";
    }
}

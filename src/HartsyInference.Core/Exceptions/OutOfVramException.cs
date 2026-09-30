namespace HartsyInference.Core.Exceptions;

/// <summary>Thrown when an operation cannot be completed because there is not enough GPU VRAM available. Callers can catch this to fall back to tiled processing, CPU offloading, or rejecting the request.</summary>
public sealed class OutOfVramException : HartsyInferenceException
{
    /// <summary>Amount of VRAM requested, in bytes.</summary>
    public long RequestedBytes { get; }

    /// <summary>Amount of VRAM available at the time of the error, in bytes.</summary>
    public long AvailableBytes { get; }

    public OutOfVramException(long requestedBytes, long availableBytes)
        : this(requestedBytes, availableBytes, 0) { }

    /// <summary>A driver refusal. Free can exceed the request: near-full, the driver reports memory it will not hand out.</summary>
    public OutOfVramException(long requestedBytes, long availableBytes, long totalBytes)
        : base(totalBytes > 0
            ? $"Out of VRAM: the driver refused a {requestedBytes / (1024 * 1024)} MB allocation; {availableBytes / (1024 * 1024)} MB of {totalBytes / (1024 * 1024)} MB was free."
            : $"Out of VRAM: the driver refused a {requestedBytes / (1024 * 1024)} MB allocation; {availableBytes / (1024 * 1024)} MB was free.")
    {
        RequestedBytes = requestedBytes;
        AvailableBytes = availableBytes;
    }

    public OutOfVramException(string message) : base(message) { }

    public OutOfVramException(string message, Exception innerException) : base(message, innerException) { }
}

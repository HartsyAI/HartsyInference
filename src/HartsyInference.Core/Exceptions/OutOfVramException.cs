namespace HartsyInference.Core.Exceptions;

/// <summary>Thrown when an operation cannot be completed because there is not enough GPU VRAM available. Callers can catch this to fall back to tiled processing, CPU offloading, or rejecting the request.</summary>
public sealed class OutOfVramException : HartsyInferenceException
{
    /// <summary>Amount of VRAM requested, in bytes.</summary>
    public long RequestedBytes { get; }

    /// <summary>Amount of VRAM available at the time of the error, in bytes.</summary>
    public long AvailableBytes { get; }

    public OutOfVramException(long requestedBytes, long availableBytes)
        : base($"Out of VRAM: requested {requestedBytes / (1024 * 1024)} MB but only {availableBytes / (1024 * 1024)} MB available.")
    {
        RequestedBytes = requestedBytes;
        AvailableBytes = availableBytes;
    }

    public OutOfVramException(string message) : base(message) { }

    public OutOfVramException(string message, Exception innerException) : base(message, innerException) { }
}

namespace HartsyInference.Core.IO;

/// <summary>Random-access reads of weight bytes without mapping the file, for tensors far larger than the host can keep resident.</summary>
public interface IWeightByteSource : IDisposable
{
    /// <summary>Length of the underlying file in bytes.</summary>
    long Length { get; }

    /// <summary>Fills <paramref name="destination"/> from <paramref name="offset"/>, or throws if the file ends first.</summary>
    void ReadAt(long offset, Span<byte> destination);

    /// <summary>Reads every range into <paramref name="destination"/> back to back, in range order.</summary>
    Task ReadBatchAsync(IReadOnlyList<ByteRange> ranges, Memory<byte> destination, CancellationToken cancellationToken = default);
}

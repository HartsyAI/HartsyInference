using HartsyInference.Core.Exceptions;
using Microsoft.Win32.SafeHandles;

namespace HartsyInference.Core.IO;

/// <summary>Positional file reads (pread) through <see cref="RandomAccess"/>; nothing is ever memory-mapped.</summary>
public sealed class PreadByteSource : IWeightByteSource
{
    private const int MaxReadsInFlight = 32;

    private readonly SafeFileHandle _handle;
    private readonly string _path;

    /// <summary>Opens <paramref name="path"/> for positional reads.</summary>
    public PreadByteSource(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        _path = path;
        _handle = File.OpenHandle(path, FileMode.Open, FileAccess.Read, FileShare.Read,
            FileOptions.Asynchronous | FileOptions.RandomAccess);
        Length = RandomAccess.GetLength(_handle);
    }

    /// <inheritdoc/>
    public long Length { get; }

    /// <inheritdoc/>
    public void ReadAt(long offset, Span<byte> destination)
    {
        CheckRange(offset, destination.Length);
        int done = 0;
        while (done < destination.Length)
        {
            int read = RandomAccess.Read(_handle, destination[done..], offset + done);
            if (read == 0)
                throw ShortRead(offset, destination.Length, done);
            done += read;
        }
    }

    /// <inheritdoc/>
    public async Task ReadBatchAsync(IReadOnlyList<ByteRange> ranges, Memory<byte> destination,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(ranges);
        long total = 0;
        foreach (ByteRange range in ranges)
        {
            CheckRange(range.Offset, range.Length);
            total += range.Length;
        }
        if (total > destination.Length)
        {
            throw new ArgumentException(
                $"The {ranges.Count} ranges total {total} bytes but the destination holds {destination.Length}.",
                nameof(destination));
        }

        List<Task> inFlight = new List<Task>(Math.Min(ranges.Count, MaxReadsInFlight));
        long written = 0;
        foreach (ByteRange range in ranges)
        {
            if (inFlight.Count == MaxReadsInFlight)
            {
                Task finished = await Task.WhenAny(inFlight).ConfigureAwait(false);
                inFlight.Remove(finished);
                await finished.ConfigureAwait(false);
            }
            inFlight.Add(ReadRangeAsync(range, destination.Slice((int)written, range.Length), cancellationToken));
            written += range.Length;
        }
        await Task.WhenAll(inFlight).ConfigureAwait(false);
    }

    private async Task ReadRangeAsync(ByteRange range, Memory<byte> destination, CancellationToken cancellationToken)
    {
        int done = 0;
        while (done < range.Length)
        {
            int read = await RandomAccess.ReadAsync(_handle, destination[done..], range.Offset + done, cancellationToken)
                .ConfigureAwait(false);
            if (read == 0)
                throw ShortRead(range.Offset, range.Length, done);
            done += read;
        }
    }

    private void CheckRange(long offset, int length)
    {
        if (offset < 0 || length < 0 || offset > Length - length)
        {
            throw new HartsyInferenceException(
                $"Read of {length} bytes at offset {offset} is outside '{_path}' ({Length} bytes). "
                + "The file is truncated or the tensor offsets do not belong to it.");
        }
    }

    private HartsyInferenceException ShortRead(long offset, int length, int done) => new(
        $"'{_path}' ended after {done} of {length} bytes at offset {offset}; the file changed while it was open.");

    /// <inheritdoc/>
    public void Dispose() => _handle.Dispose();
}

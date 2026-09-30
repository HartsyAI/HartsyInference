namespace HartsyInference.PhoneLink.Tests;

/// <summary>Read-only stream that hands out at most <c>firstChunk</c> bytes on the first read and <c>laterChunk</c> on every read after,
/// so a frame can be split at any byte boundary. The Memory overloads are overridden explicitly: the base array overloads would route
/// through the thread pool and defeat both the chunking and any per-thread allocation measurement.</summary>
internal sealed class ChunkedReadStream(byte[] data, int firstChunk, int laterChunk) : Stream
{
    private int _position;
    private bool _first = true;

    public override bool CanRead => true;
    public override bool CanSeek => false;
    public override bool CanWrite => false;
    public override long Length => data.Length;
    public override long Position { get => _position; set => throw new NotSupportedException(); }

    public override int Read(Span<byte> buffer)
    {
        int chunk = _first ? firstChunk : laterChunk;
        _first = false;
        int count = Math.Min(Math.Min(chunk, buffer.Length), data.Length - _position);
        data.AsSpan(_position, count).CopyTo(buffer);
        _position += count;
        return count;
    }

    public override int Read(byte[] buffer, int offset, int count) => Read(buffer.AsSpan(offset, count));

    public override ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default) => new(Read(buffer.Span));

    public override Task<int> ReadAsync(byte[] buffer, int offset, int count, CancellationToken cancellationToken) =>
        Task.FromResult(Read(buffer.AsSpan(offset, count)));

    public override void Flush() { }
    public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
    public override void SetLength(long value) => throw new NotSupportedException();
    public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
}

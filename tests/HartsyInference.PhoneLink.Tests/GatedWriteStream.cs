namespace HartsyInference.PhoneLink.Tests;

/// <summary>Write-only stream that remembers the memory of its last write and, when gated, parks every write until
/// <see cref="Release"/>, to hold a writer mid-frame.</summary>
internal sealed class GatedWriteStream(bool gated = true) : Stream
{
    private readonly TaskCompletionSource _gate = new(TaskCreationOptions.RunContinuationsAsynchronously);

    public int Writes { get; private set; }

    /// <summary>The exact memory the writer handed over on the last write, still pointing at the writer's array.</summary>
    public ReadOnlyMemory<byte> LastWrite { get; private set; }

    public void Release() => _gate.TrySetResult();

    public override bool CanRead => false;
    public override bool CanSeek => false;
    public override bool CanWrite => true;
    public override long Length => throw new NotSupportedException();
    public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }

    public override ValueTask WriteAsync(ReadOnlyMemory<byte> buffer, CancellationToken cancellationToken = default)
    {
        Writes++;
        LastWrite = buffer;
        return gated ? new ValueTask(_gate.Task.WaitAsync(cancellationToken)) : default;
    }

    public override Task WriteAsync(byte[] buffer, int offset, int count, CancellationToken cancellationToken) =>
        WriteAsync(buffer.AsMemory(offset, count), cancellationToken).AsTask();

    public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    public override void Flush() { }
    public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
    public override void SetLength(long value) => throw new NotSupportedException();
}

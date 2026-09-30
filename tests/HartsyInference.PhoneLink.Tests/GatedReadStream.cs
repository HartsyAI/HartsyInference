namespace HartsyInference.PhoneLink.Tests;

/// <summary>Read-only stream that remembers the memory the reader handed it and, when gated, parks the read until
/// <see cref="Release"/>; the data is delivered in one read once released, then end of stream.</summary>
internal sealed class GatedReadStream(byte[] data, bool gated = true) : Stream
{
    private readonly TaskCompletionSource _gate = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private bool _delivered;

    /// <summary>The exact memory the reader handed over on the last read, still pointing at the reader's array.</summary>
    public Memory<byte> LastRead { get; private set; }

    public void Release() => _gate.TrySetResult();

    public override bool CanRead => true;
    public override bool CanSeek => false;
    public override bool CanWrite => false;
    public override long Length => data.Length;
    public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }

    public override async ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
    {
        LastRead = buffer;
        if (gated) await _gate.Task.WaitAsync(cancellationToken);
        if (_delivered) return 0;
        _delivered = true;
        data.CopyTo(buffer);
        return data.Length;
    }

    public override Task<int> ReadAsync(byte[] buffer, int offset, int count, CancellationToken cancellationToken) =>
        ReadAsync(buffer.AsMemory(offset, count), cancellationToken).AsTask();

    public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    public override void Flush() { }
    public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
    public override void SetLength(long value) => throw new NotSupportedException();
    public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
}

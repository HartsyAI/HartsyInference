using Microsoft.Win32.SafeHandles;

namespace HartsyInference.ModelAssets.Dcp;

/// <summary>A read-only seekable view of <c>[offset, offset + length)</c> of an open file, read by positioned reads so concurrent views share one handle.</summary>
internal sealed class SliceReadStream : Stream
{
    private readonly SafeFileHandle _handle;
    private readonly long _offset;
    private readonly long _length;
    private long _position;

    public SliceReadStream(SafeFileHandle handle, long offset, long length)
    {
        ArgumentNullException.ThrowIfNull(handle);
        ArgumentOutOfRangeException.ThrowIfNegative(offset);
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        _handle = handle;
        _offset = offset;
        _length = length;
    }

    public override bool CanRead => true;

    public override bool CanSeek => true;

    public override bool CanWrite => false;

    public override long Length => _length;

    public override long Position
    {
        get => _position;
        set => _position = value >= 0 ? value : throw new ArgumentOutOfRangeException(nameof(value));
    }

    public override int Read(byte[] buffer, int offset, int count) => Read(buffer.AsSpan(offset, count));

    public override int Read(Span<byte> buffer)
    {
        long remaining = _length - _position;
        if (remaining <= 0 || buffer.IsEmpty) return 0;
        int read = RandomAccess.Read(_handle, buffer[..(int)Math.Min(buffer.Length, remaining)], _offset + _position);
        _position += read;
        return read;
    }

    public override long Seek(long offset, SeekOrigin origin)
    {
        long target = origin switch
        {
            SeekOrigin.Begin => offset,
            SeekOrigin.Current => _position + offset,
            _ => _length + offset,
        };
        Position = target;
        return target;
    }

    public override void Flush()
    {
    }

    public override void SetLength(long value) => throw new NotSupportedException();

    public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
}

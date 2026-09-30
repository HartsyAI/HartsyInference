namespace HartsyInference.PhoneLink;

/// <summary>Incremental frame parser over one direction of the link. Reads whatever the stream yields, so a header or payload
/// split across any number of reads is reassembled, and rejects an oversize payload from the 16 header bytes alone, before
/// buffering any of it.
///
/// <para>One pooled buffer serves every frame: <see cref="LinkFrame.Payload"/> is a slice of it and is invalid after the next
/// <see cref="ReadAsync"/> or <see cref="Dispose"/>, which may compact, replace or release the buffer. The buffer grows only up
/// to the largest frame seen. Single reader per instance: a second concurrent <see cref="ReadAsync"/> throws. The stream is not
/// owned. <see cref="Dispose"/> should follow completion of the last read; disposing while a read is still awaiting the stream is
/// tolerated (the pending read then completes as end of stream), but the buffer that read was filling is left to the GC
/// instead of being returned to <see cref="ArrayPool{T}.Shared"/>, because the stream may still write into it.</para></summary>
public sealed class LinkFrameReader : IDisposable
{
    private const int InitialCapacity = 16 * 1024;

    private readonly Stream _stream;
    private byte[] _buffer;
    private int _start;
    private int _end;
    private int _reading;
    private int _disposed;

    public LinkFrameReader(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);
        if (!stream.CanRead)
            throw new ArgumentException("PhoneLink reader needs a readable stream.", nameof(stream));
        _stream = stream;
        _buffer = ArrayPool<byte>.Shared.Rent(InitialCapacity);
    }

    /// <summary>Bytes received but not yet consumed by a returned frame.</summary>
    public int BufferedBytes => _end - _start;

    /// <summary>Returns the next frame, or null when the stream ends cleanly between frames. A stream that ends inside a frame,
    /// or a header declaring more than <see cref="LinkProtocol.MaxPayloadBytes"/>, throws <see cref="LinkProtocolException"/>.
    /// A frame already in the buffer, or a read the stream completes synchronously, never enters an async state machine.</summary>
    public ValueTask<LinkFrame?> ReadAsync(CancellationToken cancel)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (Interlocked.CompareExchange(ref _reading, 1, 0) != 0)
            throw new InvalidOperationException("LinkFrameReader is single-reader: another ReadAsync is in progress on this instance.");
        try
        {
            while (true)
            {
                if (TryTakeFrame(out LinkFrame frame))
                {
                    Release();
                    return new ValueTask<LinkFrame?>(frame);
                }
                ValueTask<int> pending = _stream.ReadAsync(_buffer.AsMemory(_end), cancel);
                if (!pending.IsCompletedSuccessfully) return ReadSlowAsync(pending, cancel);
                if (!Consume(pending.Result))
                {
                    Release();
                    return new ValueTask<LinkFrame?>((LinkFrame?)null);
                }
            }
        }
        catch
        {
            Release();
            throw;
        }
    }

    /// <summary>Continues a read the stream did not complete synchronously; owns the <see cref="_reading"/> claim until it returns.</summary>
    [AsyncMethodBuilder(typeof(PoolingAsyncValueTaskMethodBuilder<>))]
    private async ValueTask<LinkFrame?> ReadSlowAsync(ValueTask<int> pending, CancellationToken cancel)
    {
        try
        {
            while (true)
            {
                int read = await pending.ConfigureAwait(false);
                // Disposed while the stream was filling the buffer: the bytes landed in an array nobody will slice again.
                if (Volatile.Read(ref _disposed) != 0) return null;
                if (!Consume(read)) return null;
                if (TryTakeFrame(out LinkFrame frame)) return frame;
                pending = _stream.ReadAsync(_buffer.AsMemory(_end), cancel);
            }
        }
        finally
        {
            Release();
        }
    }

    private void Release() => Volatile.Write(ref _reading, 0);

    /// <summary>Slices the next complete frame out of the buffer; otherwise makes room for it and returns false.</summary>
    private bool TryTakeFrame(out LinkFrame frame)
    {
        if (LinkFrameHeader.TryRead(_buffer.AsSpan(_start, _end - _start), out LinkFrameHeader header))
        {
            if (header.PayloadLength > LinkProtocol.MaxPayloadBytes)
            {
                throw new LinkProtocolException(
                    $"PhoneLink frame declares a {header.PayloadLength}-byte payload; the limit is {LinkProtocol.MaxPayloadBytes} bytes.");
            }
            int frameLength = LinkFrameHeader.Size + (int)header.PayloadLength;
            if (_end - _start >= frameLength)
            {
                frame = new LinkFrame(header, _buffer.AsMemory(_start + LinkFrameHeader.Size, (int)header.PayloadLength));
                _start += frameLength;
                return true;
            }
            EnsureRoom(frameLength);
        }
        else
        {
            EnsureRoom(LinkFrameHeader.Size);
        }
        frame = default;
        return false;
    }

    /// <summary>Accounts for one read: false at a clean end of stream, throws when the stream ends mid-frame.</summary>
    private bool Consume(int read)
    {
        if (read > 0)
        {
            _end += read;
            return true;
        }
        if (_end == _start) return false;
        throw new LinkProtocolException($"PhoneLink stream ended inside a frame with {_end - _start} bytes pending.");
    }

    /// <summary>Guarantees the pending bytes sit in a buffer with free space after them and room for
    /// <paramref name="frameLength"/> in total.</summary>
    private void EnsureRoom(int frameLength)
    {
        int pending = _end - _start;
        if (_buffer.Length < frameLength)
        {
            // The whole buffer is too small for this frame: rent a larger one and move the pending bytes to its front.
            byte[] grown = ArrayPool<byte>.Shared.Rent(frameLength);
            _buffer.AsSpan(_start, pending).CopyTo(grown);
            ArrayPool<byte>.Shared.Return(_buffer);
            _buffer = grown;
        }
        else if (_start > 0 && (_end == _buffer.Length || _buffer.Length - _start < frameLength))
        {
            // Big enough, but the pending bytes sit too far in: either no free space follows them or the tail cannot
            // hold the frame. Slide them to the front (overlapping copy is safe with CopyTo).
            _buffer.AsSpan(_start, pending).CopyTo(_buffer);
        }
        else
        {
            // Pending bytes already have room after them for the whole frame; nothing to move.
            return;
        }
        _start = 0;
        _end = pending;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        // Claiming the reader stops any later read. Failing to claim means a read is still awaiting the stream, which may
        // yet write into the buffer, so that array is left to the GC instead of being handed to the pool's next renter.
        if (Interlocked.CompareExchange(ref _reading, 1, 0) != 0) return;
        byte[] buffer = _buffer;
        _buffer = [];
        _start = _end = 0;
        ArrayPool<byte>.Shared.Return(buffer);
    }
}

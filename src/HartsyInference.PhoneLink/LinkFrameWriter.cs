using System.Buffers.Binary;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Serialization.Metadata;

namespace HartsyInference.PhoneLink;

/// <summary>Writes frames for one direction of the link. Each frame is assembled, header and payload, in one pooled staging
/// buffer and handed to the stream in a single write, so an audio frame costs no allocation and one syscall.
///
/// <para><b>Single writer.</b> One instance serves one direction and one thread at a time; the owner serializes callers (the
/// design is one sender thread per direction). A second concurrent write throws <see cref="InvalidOperationException"/> rather
/// than interleaving bytes. <see cref="Dispose"/> returns the staging buffer to <see cref="ArrayPool{T}.Shared"/>; the stream is
/// not owned.</para></summary>
public sealed class LinkFrameWriter : IDisposable
{
    private const int InitialCapacity = 4096;
    private const int MaxOutboundSamples = (LinkProtocol.MaxPayloadBytes - sizeof(uint)) / 2;

    private readonly Stream _stream;
    private readonly StagingBuffer _staging;
    private readonly Utf8JsonWriter _json;
    private uint _sequence;
    private int _busy;
    private int _disposed;

    public LinkFrameWriter(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);
        if (!stream.CanWrite)
            throw new ArgumentException("PhoneLink writer needs a writable stream.", nameof(stream));
        _stream = stream;
        _staging = new StagingBuffer(InitialCapacity);
        // The peer is our own UTF-8 reader, never a browser: "+" in E.164 numbers and non-ASCII text go out verbatim.
        _json = new Utf8JsonWriter(_staging, new JsonWriterOptions { Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping });
    }

    /// <summary>Sequence number the next frame will carry: every frame this writer has sent, counted from zero.</summary>
    public uint NextSequence => _sequence;

    /// <summary>Writes an arbitrary frame. The typed methods below are preferred; this is for tests and forward-compatible
    /// extensions.</summary>
    public ValueTask WriteAsync(LinkMessageType type, LinkFrameFlags flags, uint callId, ReadOnlyMemory<byte> payload, CancellationToken cancel)
    {
        if (payload.Length > LinkProtocol.MaxPayloadBytes)
        {
            throw new ArgumentOutOfRangeException(
                nameof(payload), payload.Length, $"PhoneLink payloads are capped at {LinkProtocol.MaxPayloadBytes} bytes.");
        }
        Begin();
        payload.Span.CopyTo(_staging.GetSpan(payload.Length));
        _staging.Advance(payload.Length);
        return FlushFrameAsync(type, flags, callId, cancel);
    }

    public ValueTask WriteHelloAsync(LinkHello hello, CancellationToken cancel)
    {
        if (hello.Token is null)
            throw new ArgumentException("Hello needs a token; use an empty string for none.", nameof(hello));
        int tokenBytes = Encoding.UTF8.GetByteCount(hello.Token);
        if (tokenBytes > ushort.MaxValue)
            throw new ArgumentException($"Hello token is {tokenBytes} UTF-8 bytes; the wire allows {ushort.MaxValue}.", nameof(hello));
        Begin();
        Span<byte> body = _staging.GetSpan(8 + tokenBytes);
        BinaryPrimitives.WriteUInt16LittleEndian(body, hello.Version);
        BinaryPrimitives.WriteUInt32LittleEndian(body.Slice(2), hello.InboundRate);
        BinaryPrimitives.WriteUInt16LittleEndian(body.Slice(6), (ushort)tokenBytes);
        Encoding.UTF8.GetBytes(hello.Token, body.Slice(8));
        _staging.Advance(8 + tokenBytes);
        return FlushFrameAsync(LinkMessageType.Hello, LinkFrameFlags.None, LinkProtocol.ConnectionCallId, cancel);
    }

    public ValueTask WriteHelloAckAsync(LinkHelloAck ack, CancellationToken cancel)
    {
        if (!LinkProtocol.IsOutboundSampleRate(ack.OutboundRate))
            throw new ArgumentException($"Outbound rate {ack.OutboundRate} is not one of the protocol's rates.", nameof(ack));
        if (ack.MaxFrameMs == 0)
            throw new ArgumentException("MaxFrameMs must be positive.", nameof(ack));
        Begin();
        Span<byte> body = _staging.GetSpan(6);
        BinaryPrimitives.WriteUInt32LittleEndian(body, ack.OutboundRate);
        BinaryPrimitives.WriteUInt16LittleEndian(body.Slice(4), ack.MaxFrameMs);
        _staging.Advance(6);
        return FlushFrameAsync(LinkMessageType.HelloAck, LinkFrameFlags.None, LinkProtocol.ConnectionCallId, cancel);
    }

    public ValueTask WriteCallStartAsync(uint callId, CallStartMessage message, CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(message);
        Begin();
        StageJson(message, LinkJsonContext.Default.CallStartMessage);
        return FlushFrameAsync(LinkMessageType.CallStart, LinkFrameFlags.None, callId, cancel);
    }

    public ValueTask WriteCallEndAsync(uint callId, LinkCallEndReason reason, CancellationToken cancel)
    {
        Begin();
        _staging.GetSpan(1)[0] = (byte)reason;
        _staging.Advance(1);
        return FlushFrameAsync(LinkMessageType.CallEnd, LinkFrameFlags.None, callId, cancel);
    }

    /// <summary>One 20 ms frame of caller audio: exactly <see cref="LinkProtocol.InboundFrameSamples"/> samples at
    /// <see cref="LinkProtocol.InboundSampleRate"/>. Allocation-free.</summary>
    public ValueTask WriteInboundAudioAsync(uint callId, ReadOnlyMemory<short> pcm, bool concealed, CancellationToken cancel)
    {
        if (pcm.Length != LinkProtocol.InboundFrameSamples)
            throw new ArgumentException($"InboundAudio carries exactly {LinkProtocol.InboundFrameSamples} samples, got {pcm.Length}.", nameof(pcm));
        Begin();
        StagePcm(pcm.Span);
        return FlushFrameAsync(LinkMessageType.InboundAudio, concealed ? LinkFrameFlags.Concealed : LinkFrameFlags.None, callId, cancel);
    }

    /// <summary>Synthesized audio for one turn at the rate announced in <see cref="LinkHelloAck"/>. Allocation-free.</summary>
    public ValueTask WriteOutboundAudioAsync(uint callId, uint turnId, ReadOnlyMemory<short> pcm, CancellationToken cancel)
    {
        if (pcm.Length == 0 || pcm.Length > MaxOutboundSamples)
            throw new ArgumentException($"OutboundAudio carries 1..{MaxOutboundSamples} samples, got {pcm.Length}.", nameof(pcm));
        Begin();
        StageUInt32(turnId);
        StagePcm(pcm.Span);
        return FlushFrameAsync(LinkMessageType.OutboundAudio, LinkFrameFlags.None, callId, cancel);
    }

    public ValueTask WriteOutboundEndAsync(uint callId, uint turnId, CancellationToken cancel)
    {
        Begin();
        StageUInt32(turnId);
        return FlushFrameAsync(LinkMessageType.OutboundEnd, LinkFrameFlags.None, callId, cancel);
    }

    public ValueTask WriteFlushAsync(uint callId, uint turnId, CancellationToken cancel)
    {
        Begin();
        StageUInt32(turnId);
        return FlushFrameAsync(LinkMessageType.Flush, LinkFrameFlags.None, callId, cancel);
    }

    public ValueTask WriteFlushAckAsync(uint callId, LinkFlushAck ack, CancellationToken cancel)
    {
        Begin();
        StageUInt32(ack.TurnId);
        StageUInt32(ack.MsDiscarded);
        return FlushFrameAsync(LinkMessageType.FlushAck, LinkFrameFlags.None, callId, cancel);
    }

    public ValueTask WriteEventAsync(uint callId, LinkEventMessage message, CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(message);
        Begin();
        StageJson(message, LinkJsonContext.Default.LinkEventMessage);
        return FlushFrameAsync(LinkMessageType.Event, LinkFrameFlags.None, callId, cancel);
    }

    public ValueTask WriteDtmfAsync(uint callId, LinkDtmf dtmf, CancellationToken cancel)
    {
        if (!LinkDtmf.IsDigit(dtmf.Digit))
            throw new ArgumentException($"'{dtmf.Digit}' is not a DTMF key.", nameof(dtmf));
        Begin();
        Span<byte> body = _staging.GetSpan(3);
        body[0] = (byte)dtmf.Digit;
        BinaryPrimitives.WriteUInt16LittleEndian(body.Slice(1), dtmf.DurationMs);
        _staging.Advance(3);
        return FlushFrameAsync(LinkMessageType.DtmfEvent, LinkFrameFlags.None, callId, cancel);
    }

    public ValueTask WriteToolRequestAsync(uint callId, uint requestId, ToolRequestMessage message, CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(message);
        Begin();
        StageUInt32(requestId);
        StageJson(message, LinkJsonContext.Default.ToolRequestMessage);
        return FlushFrameAsync(LinkMessageType.ToolRequest, LinkFrameFlags.None, callId, cancel);
    }

    public ValueTask WriteToolResultAsync(uint callId, uint requestId, ToolResultMessage message, CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(message);
        Begin();
        StageUInt32(requestId);
        StageJson(message, LinkJsonContext.Default.ToolResultMessage);
        return FlushFrameAsync(LinkMessageType.ToolResult, LinkFrameFlags.None, callId, cancel);
    }

    public ValueTask WritePingAsync(ulong monotonicNs, CancellationToken cancel) => WriteTimestampAsync(LinkMessageType.Ping, monotonicNs, cancel);

    public ValueTask WritePongAsync(ulong monotonicNs, CancellationToken cancel) => WriteTimestampAsync(LinkMessageType.Pong, monotonicNs, cancel);

    public ValueTask WriteErrorAsync(uint callId, LinkErrorMessage message, CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(message);
        Begin();
        StageJson(message, LinkJsonContext.Default.LinkErrorMessage);
        return FlushFrameAsync(LinkMessageType.Error, LinkFrameFlags.None, callId, cancel);
    }

    private ValueTask WriteTimestampAsync(LinkMessageType type, ulong monotonicNs, CancellationToken cancel)
    {
        Begin();
        BinaryPrimitives.WriteUInt64LittleEndian(_staging.GetSpan(8), monotonicNs);
        _staging.Advance(8);
        return FlushFrameAsync(type, LinkFrameFlags.None, LinkProtocol.ConnectionCallId, cancel);
    }

    /// <summary>Claims the writer for one frame and reserves the header bytes; every caller must reach
    /// <see cref="FlushFrameAsync"/> or <see cref="Abort"/>.</summary>
    private void Begin()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (Interlocked.CompareExchange(ref _busy, 1, 0) != 0)
            throw new InvalidOperationException("LinkFrameWriter is single-writer: another frame is being written on this instance.");
        _staging.Reset(LinkFrameHeader.Size);
    }

    private void Abort() => Volatile.Write(ref _busy, 0);

    private void StageUInt32(uint value)
    {
        BinaryPrimitives.WriteUInt32LittleEndian(_staging.GetSpan(sizeof(uint)), value);
        _staging.Advance(sizeof(uint));
    }

    private void StagePcm(ReadOnlySpan<short> pcm)
    {
        int bytes = pcm.Length * 2;
        LinkPcm.Encode(pcm, _staging.GetSpan(bytes));
        _staging.Advance(bytes);
    }

    private void StageJson<T>(T value, JsonTypeInfo<T> typeInfo)
    {
        try
        {
            _json.Reset(_staging);
            JsonSerializer.Serialize(_json, value, typeInfo);
            _json.Flush();
        }
        catch
        {
            Abort();
            throw;
        }
    }

    /// <summary>Stamps the header and hands the frame to the stream. A write the stream completes synchronously never enters
    /// an async state machine.</summary>
    private ValueTask FlushFrameAsync(LinkMessageType type, LinkFrameFlags flags, uint callId, CancellationToken cancel)
    {
        ValueTask write;
        try
        {
            int payloadLength = _staging.Length - LinkFrameHeader.Size;
            if (payloadLength > LinkProtocol.MaxPayloadBytes)
            {
                throw new LinkProtocolException(
                    $"{type} payload is {payloadLength} bytes; the limit is {LinkProtocol.MaxPayloadBytes} bytes.");
            }
            new LinkFrameHeader((uint)payloadLength, type, flags, callId, _sequence).Write(_staging.Span);
            _sequence++;
            write = _stream.WriteAsync(_staging.WrittenMemory, cancel);
            if (!write.IsCompletedSuccessfully) return CompleteWriteAsync(write, cancel);
            Task flush = _stream.FlushAsync(cancel);
            if (!flush.IsCompletedSuccessfully) return CompleteFlushAsync(flush);
        }
        catch
        {
            Abort();
            throw;
        }
        Abort();
        return default;
    }

    [AsyncMethodBuilder(typeof(PoolingAsyncValueTaskMethodBuilder))]
    private async ValueTask CompleteWriteAsync(ValueTask write, CancellationToken cancel)
    {
        try
        {
            await write.ConfigureAwait(false);
            await _stream.FlushAsync(cancel).ConfigureAwait(false);
        }
        finally
        {
            Abort();
        }
    }

    [AsyncMethodBuilder(typeof(PoolingAsyncValueTaskMethodBuilder))]
    private async ValueTask CompleteFlushAsync(Task flush)
    {
        try
        {
            await flush.ConfigureAwait(false);
        }
        finally
        {
            Abort();
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _json.Dispose();
        _staging.Dispose();
    }

    /// <summary>Growable pooled frame buffer; the first <see cref="LinkFrameHeader.Size"/> bytes are reserved for the header
    /// by <see cref="Reset"/>.</summary>
    private sealed class StagingBuffer(int capacity) : IBufferWriter<byte>, IDisposable
    {
        private byte[] _array = ArrayPool<byte>.Shared.Rent(capacity);
        private int _length;

        public int Length => _length;

        public Span<byte> Span => _array.AsSpan(0, _length);

        public ReadOnlyMemory<byte> WrittenMemory => _array.AsMemory(0, _length);

        public void Reset(int reserved) => _length = reserved;

        public void Advance(int count)
        {
            if (count < 0 || _length + count > _array.Length)
                throw new ArgumentOutOfRangeException(nameof(count), count, "Advance past the span handed out.");
            _length += count;
        }

        public Memory<byte> GetMemory(int sizeHint = 0)
        {
            Ensure(sizeHint);
            return _array.AsMemory(_length);
        }

        public Span<byte> GetSpan(int sizeHint = 0)
        {
            Ensure(sizeHint);
            return _array.AsSpan(_length);
        }

        private void Ensure(int sizeHint)
        {
            if (sizeHint < 1) sizeHint = 1;
            if (_array.Length - _length >= sizeHint) return;
            byte[] grown = ArrayPool<byte>.Shared.Rent(Math.Max(_array.Length * 2, _length + sizeHint));
            _array.AsSpan(0, _length).CopyTo(grown);
            ArrayPool<byte>.Shared.Return(_array);
            _array = grown;
        }

        public void Dispose()
        {
            byte[] array = Interlocked.Exchange(ref _array, []);
            if (array.Length > 0) ArrayPool<byte>.Shared.Return(array);
        }
    }
}

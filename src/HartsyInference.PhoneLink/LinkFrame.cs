using System.Buffers.Binary;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization.Metadata;

namespace HartsyInference.PhoneLink;

/// <summary>One frame as <see cref="LinkFrameReader"/> delivers it. <see cref="Payload"/> is borrowed from the reader's buffer and
/// is dead after the next <see cref="LinkFrameReader.ReadAsync"/>, so decode it before reading on. The <c>Read*</c> helpers check
/// the frame type and payload size and throw <see cref="LinkProtocolException"/> when the peer sent something malformed.</summary>
public readonly record struct LinkFrame(LinkFrameHeader Header, ReadOnlyMemory<byte> Payload)
{
    private const int TurnIdBytes = sizeof(uint);
    private const int RequestIdBytes = sizeof(uint);

    public LinkMessageType Type => Header.Type;

    /// <summary>True on an <see cref="LinkMessageType.InboundAudio"/> frame the gateway filled by packet-loss concealment.</summary>
    public bool Concealed => (Header.Flags & LinkFrameFlags.Concealed) != 0;

    /// <summary>Samples an audio frame carries, or zero for every other type.</summary>
    public int PcmSampleCount => Header.Type switch
    {
        LinkMessageType.InboundAudio => Payload.Length / 2,
        LinkMessageType.OutboundAudio => Math.Max(0, Payload.Length - TurnIdBytes) / 2,
        _ => 0,
    };

    public LinkHello ReadHello()
    {
        ReadOnlySpan<byte> body = Body(LinkMessageType.Hello, 8, LinkProtocol.MaxPayloadBytes);
        int tokenLength = BinaryPrimitives.ReadUInt16LittleEndian(body.Slice(6));
        if (body.Length != 8 + tokenLength)
            throw new LinkProtocolException($"Hello declares a {tokenLength}-byte token but carries {body.Length - 8} bytes.");
        uint inboundRate = BinaryPrimitives.ReadUInt32LittleEndian(body.Slice(2));
        if (inboundRate != LinkProtocol.InboundSampleRate)
            throw new LinkProtocolException($"Hello offers inbound audio at {inboundRate} Hz; the link is {LinkProtocol.InboundSampleRate} Hz.");
        return new LinkHello(
            BinaryPrimitives.ReadUInt16LittleEndian(body), inboundRate, Encoding.UTF8.GetString(body.Slice(8, tokenLength)));
    }

    public LinkHelloAck ReadHelloAck()
    {
        ReadOnlySpan<byte> body = Body(LinkMessageType.HelloAck, 6, 6);
        LinkHelloAck ack = new(BinaryPrimitives.ReadUInt32LittleEndian(body), BinaryPrimitives.ReadUInt16LittleEndian(body.Slice(4)));
        if (!LinkProtocol.IsOutboundSampleRate(ack.OutboundRate))
            throw new LinkProtocolException($"HelloAck announces an unsupported outbound rate {ack.OutboundRate}.");
        if (ack.MaxFrameMs == 0)
            throw new LinkProtocolException("HelloAck announces a zero maximum frame length.");
        return ack;
    }

    public CallStartMessage ReadCallStart() => ReadJson(LinkMessageType.CallStart, 0, LinkJsonContext.Default.CallStartMessage);

    public LinkCallEndReason ReadCallEnd() => (LinkCallEndReason)Body(LinkMessageType.CallEnd, 1, 1)[0];

    /// <summary>Copies the samples of an <see cref="LinkMessageType.InboundAudio"/> or <see cref="LinkMessageType.OutboundAudio"/>
    /// frame into <paramref name="destination"/> and returns how many were written. Allocation-free.</summary>
    public int ReadPcm(Span<short> destination)
    {
        int offset = Header.Type switch
        {
            LinkMessageType.InboundAudio => 0,
            LinkMessageType.OutboundAudio => TurnIdBytes,
            _ => throw new LinkProtocolException($"{Header.Type} frames carry no PCM."),
        };
        ReadOnlySpan<byte> pcm = Body(Header.Type, offset + 2, LinkProtocol.MaxPayloadBytes).Slice(offset);
        if ((pcm.Length & 1) != 0)
            throw new LinkProtocolException($"{Header.Type} payload of {pcm.Length} PCM bytes is not a whole number of samples.");
        int samples = pcm.Length / 2;
        if (destination.Length < samples)
        {
            throw new ArgumentException(
                $"PCM destination holds {destination.Length} samples, frame carries {samples}.", nameof(destination));
        }
        LinkPcm.Decode(pcm, destination.Slice(0, samples));
        return samples;
    }

    /// <summary>The turn id prefix of <see cref="LinkMessageType.OutboundAudio"/>, <see cref="LinkMessageType.OutboundEnd"/>,
    /// <see cref="LinkMessageType.Flush"/> or <see cref="LinkMessageType.FlushAck"/>.</summary>
    public uint ReadTurnId()
    {
        ReadOnlySpan<byte> body = Header.Type switch
        {
            LinkMessageType.OutboundAudio => Body(Header.Type, TurnIdBytes, LinkProtocol.MaxPayloadBytes),
            LinkMessageType.OutboundEnd or LinkMessageType.Flush => Body(Header.Type, TurnIdBytes, TurnIdBytes),
            LinkMessageType.FlushAck => Body(Header.Type, 8, 8),
            _ => throw new LinkProtocolException($"{Header.Type} frames carry no turn id."),
        };
        return BinaryPrimitives.ReadUInt32LittleEndian(body);
    }

    public LinkFlushAck ReadFlushAck()
    {
        ReadOnlySpan<byte> body = Body(LinkMessageType.FlushAck, 8, 8);
        return new LinkFlushAck(BinaryPrimitives.ReadUInt32LittleEndian(body), BinaryPrimitives.ReadUInt32LittleEndian(body.Slice(4)));
    }

    public LinkEventMessage ReadEvent() => ReadJson(LinkMessageType.Event, 0, LinkJsonContext.Default.LinkEventMessage);

    public LinkDtmf ReadDtmf()
    {
        ReadOnlySpan<byte> body = Body(LinkMessageType.DtmfEvent, 3, 3);
        char digit = (char)body[0];
        if (!LinkDtmf.IsDigit(digit))
            throw new LinkProtocolException($"DtmfEvent carries byte 0x{body[0]:X2}, which is not a DTMF key.");
        return new LinkDtmf(digit, BinaryPrimitives.ReadUInt16LittleEndian(body.Slice(1)));
    }

    public ToolRequestMessage ReadToolRequest(out uint requestId)
    {
        ToolRequestMessage message = ReadJson(LinkMessageType.ToolRequest, RequestIdBytes, LinkJsonContext.Default.ToolRequestMessage);
        requestId = BinaryPrimitives.ReadUInt32LittleEndian(Payload.Span);
        return message;
    }

    public ToolResultMessage ReadToolResult(out uint requestId)
    {
        ToolResultMessage message = ReadJson(LinkMessageType.ToolResult, RequestIdBytes, LinkJsonContext.Default.ToolResultMessage);
        requestId = BinaryPrimitives.ReadUInt32LittleEndian(Payload.Span);
        return message;
    }

    /// <summary>The sender's monotonic clock in nanoseconds from a <see cref="LinkMessageType.Ping"/> or
    /// <see cref="LinkMessageType.Pong"/>.</summary>
    public ulong ReadTimestampNs()
    {
        if (Header.Type is not (LinkMessageType.Ping or LinkMessageType.Pong))
            throw new LinkProtocolException($"{Header.Type} frames carry no timestamp.");
        return BinaryPrimitives.ReadUInt64LittleEndian(Body(Header.Type, 8, 8));
    }

    public LinkErrorMessage ReadError() => ReadJson(LinkMessageType.Error, 0, LinkJsonContext.Default.LinkErrorMessage);

    private ReadOnlySpan<byte> Body(LinkMessageType expected, int minLength, int maxLength)
    {
        if (Header.Type != expected)
            throw new LinkProtocolException($"Expected a {expected} frame, got {Header.Type}.");
        if (Payload.Length < minLength || Payload.Length > maxLength)
        {
            string range = minLength == maxLength ? minLength.ToString() : $"{minLength}..{maxLength}";
            throw new LinkProtocolException($"{expected} payload is {Payload.Length} bytes; expected {range}.");
        }
        return Payload.Span;
    }

    private T ReadJson<T>(LinkMessageType expected, int offset, JsonTypeInfo<T> typeInfo) where T : class
    {
        ReadOnlySpan<byte> json = Body(expected, offset + 2, LinkProtocol.MaxPayloadBytes).Slice(offset);
        try
        {
            return JsonSerializer.Deserialize(json, typeInfo)
                ?? throw new LinkProtocolException($"{expected} payload is JSON null.");
        }
        catch (JsonException ex)
        {
            throw new LinkProtocolException($"{expected} payload is not valid JSON: {ex.Message}", ex);
        }
    }
}

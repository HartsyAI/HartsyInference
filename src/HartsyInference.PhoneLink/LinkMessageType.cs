namespace HartsyInference.PhoneLink;

/// <summary>Frame type byte at header offset 4. Values are the wire encoding and never change meaning; new types take unused
/// values.</summary>
public enum LinkMessageType : byte
{
    /// <summary>Gateway → host, first frame of every connection: version, inbound rate and the shared token.</summary>
    Hello = 0x01,
    /// <summary>Host → gateway, reply to <see cref="Hello"/>: outbound rate and the largest frame the host will send.</summary>
    HelloAck = 0x02,
    /// <summary>Gateway → host: a call has been answered or placed; JSON payload.</summary>
    CallStart = 0x10,
    /// <summary>Either direction: the call is over; one <see cref="LinkCallEndReason"/> byte.</summary>
    CallEnd = 0x11,
    /// <summary>Gateway → host: one 20 ms PCM16 frame at 16 kHz from the caller.</summary>
    InboundAudio = 0x20,
    /// <summary>Host → gateway: turnId, then PCM16 at the negotiated outbound rate.</summary>
    OutboundAudio = 0x21,
    /// <summary>Host → gateway: no more audio will follow for this turnId.</summary>
    OutboundEnd = 0x22,
    /// <summary>Host → gateway: discard every queued outbound frame whose turnId is at or below this one.</summary>
    Flush = 0x23,
    /// <summary>Gateway → host: the flush was applied; carries the milliseconds of audio thrown away.</summary>
    FlushAck = 0x24,
    /// <summary>Host → gateway: state, transcript or latency event; JSON payload.</summary>
    Event = 0x30,
    /// <summary>Gateway → host: a DTMF digit was received from the caller.</summary>
    DtmfEvent = 0x31,
    /// <summary>Host → gateway: run a telephony tool (hang up, DTMF, transfer, hold, play prompt); requestId then JSON.</summary>
    ToolRequest = 0x40,
    /// <summary>Gateway → host: outcome of a <see cref="ToolRequest"/>; requestId then JSON.</summary>
    ToolResult = 0x41,
    /// <summary>Either direction: liveness probe carrying the sender's monotonic clock in nanoseconds.</summary>
    Ping = 0x50,
    /// <summary>Either direction: echo of a <see cref="Ping"/> timestamp.</summary>
    Pong = 0x51,
    /// <summary>Either direction: a protocol or application error; JSON payload.</summary>
    Error = 0x60,
}

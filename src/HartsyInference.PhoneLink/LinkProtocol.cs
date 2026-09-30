namespace HartsyInference.PhoneLink;

/// <summary>Wire constants shared by both ends of the link. The spec is <c>docs/Research/PHONE_LINK_PROTOCOL.md</c>.</summary>
public static class LinkProtocol
{
    /// <summary>Protocol version carried in <see cref="LinkMessageType.Hello"/>; the host refuses any other value.</summary>
    public const ushort Version = 1;

    /// <summary>Largest payload either side accepts; a header declaring more is a protocol error before any payload is read.</summary>
    public const int MaxPayloadBytes = 1 << 20;

    /// <summary>Sample rate of every <see cref="LinkMessageType.InboundAudio"/> frame.</summary>
    public const int InboundSampleRate = 16000;

    /// <summary>Samples per <see cref="LinkMessageType.InboundAudio"/> frame: 20 ms at <see cref="InboundSampleRate"/>.</summary>
    public const int InboundFrameSamples = 320;

    /// <summary>Call id of connection-level frames (<see cref="LinkMessageType.Hello"/>, <see cref="LinkMessageType.HelloAck"/>, <see cref="LinkMessageType.Ping"/>, <see cref="LinkMessageType.Pong"/>); real calls start at 1.</summary>
    public const uint ConnectionCallId = 0;

    /// <summary>Outbound rates a host may announce in <see cref="LinkMessageType.HelloAck"/>.</summary>
    public static ReadOnlySpan<uint> OutboundSampleRates => [8000, 16000, 22050, 24000, 48000];

    /// <summary>True when <paramref name="rate"/> is one of <see cref="OutboundSampleRates"/>.</summary>
    public static bool IsOutboundSampleRate(uint rate) => OutboundSampleRates.IndexOf(rate) >= 0;
}

namespace HartsyInference.PhoneLink;

/// <summary>Binary payload of <see cref="LinkMessageType.HelloAck"/>: <c>u32 outboundRate | u16 maxFrameMs</c>. The rate must be
/// one of <see cref="LinkProtocol.OutboundSampleRates"/>.</summary>
public readonly record struct LinkHelloAck(uint OutboundRate, ushort MaxFrameMs);

namespace HartsyInference.PhoneLink;

/// <summary>Binary payload of <see cref="LinkMessageType.Hello"/>: <c>u16 version | u32 inboundRate | u16 tokenLength | UTF-8 token</c>.</summary>
public readonly record struct LinkHello(ushort Version, uint InboundRate, string Token);

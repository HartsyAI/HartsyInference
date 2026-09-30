using HartsyInference.PhoneLink;

namespace HartsyInference.PhoneGateway.Transport;

/// <summary>One queued control frame for the link writer thread. Which fields mean what depends on <see cref="Type"/>:
/// <c>CallEnd</c> carries the reason in <see cref="Arg"/>; <c>FlushAck</c> the turn in <see cref="Arg"/> and the
/// milliseconds in <see cref="Arg2"/>; <c>DtmfEvent</c> the digit in <see cref="Arg"/> and the duration in
/// <see cref="Arg2"/>; <c>ToolResult</c> the request id in <see cref="Arg"/> and the message in <see cref="Payload"/>;
/// <c>Pong</c> the echoed clock in <see cref="Timestamp"/>; <c>CallStart</c> and <c>Error</c> their message in
/// <see cref="Payload"/>.</summary>
internal readonly record struct LinkControlItem(LinkMessageType Type, uint CallId, uint Arg, uint Arg2, ulong Timestamp, object? Payload);

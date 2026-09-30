namespace HartsyInference.PhoneLink;

/// <summary>Binary payload of <see cref="LinkMessageType.FlushAck"/>: <c>u32 turnId | u32 msDiscarded</c>, the flushed epoch
/// and how much queued audio it threw away.</summary>
public readonly record struct LinkFlushAck(uint TurnId, uint MsDiscarded);

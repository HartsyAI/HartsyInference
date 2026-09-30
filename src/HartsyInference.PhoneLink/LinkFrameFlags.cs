namespace HartsyInference.PhoneLink;

/// <summary>Frame flag bits at header offset 5. Undefined bits are written as zero and ignored on read.</summary>
[Flags]
public enum LinkFrameFlags : byte
{
    None = 0,
    /// <summary>On <see cref="LinkMessageType.InboundAudio"/>: the samples were synthesized by packet-loss concealment, not received.</summary>
    Concealed = 1,
}

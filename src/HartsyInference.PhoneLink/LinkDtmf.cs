namespace HartsyInference.PhoneLink;

/// <summary>Binary payload of <see cref="LinkMessageType.DtmfEvent"/>: <c>u8 digit | u16 durationMs</c>. The digit is the ASCII
/// key: 0-9, *, # or A-D.</summary>
public readonly record struct LinkDtmf(char Digit, ushort DurationMs)
{
    /// <summary>True for the sixteen DTMF keys the wire accepts.</summary>
    public static bool IsDigit(char digit) => digit is (>= '0' and <= '9') or '*' or '#' or (>= 'A' and <= 'D');
}

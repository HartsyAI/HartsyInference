namespace HartsyInference.PhoneLink;

/// <summary>Single-byte payload of <see cref="LinkMessageType.CallEnd"/>. Unknown values are preserved, not rejected, so a
/// newer peer can add reasons.</summary>
public enum LinkCallEndReason : byte
{
    /// <summary>The call ran to completion and was hung up by the agent.</summary>
    Completed = 0,
    /// <summary>The far end hung up.</summary>
    RemoteHangup = 1,
    /// <summary>The gateway or an operator ended the call.</summary>
    LocalHangup = 2,
    /// <summary>An outbound call was rejected as busy.</summary>
    Busy = 3,
    /// <summary>An outbound call was never answered.</summary>
    NoAnswer = 4,
    /// <summary>SIP or media failure.</summary>
    Failed = 5,
}

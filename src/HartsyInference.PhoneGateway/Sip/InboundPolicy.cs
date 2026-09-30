namespace HartsyInference.PhoneGateway.Sip;

/// <summary>What happens to an INVITE from the network.</summary>
public enum InboundPolicy
{
    /// <summary>Answer every call.</summary>
    AllowAll,
    /// <summary>Answer callers whose user part is on the allowlist; decline the rest with 603.</summary>
    Allowlist,
    /// <summary>Decline every inbound call with 603 (outbound-only deployments).</summary>
    Reject,
}

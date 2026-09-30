namespace HartsyInference.PhoneGateway.Sip;

/// <summary>Where the address written into SIP Contact and SDP comes from.</summary>
public enum PublicAddressMode
{
    /// <summary>The local interface address (LAN softphones, or a provider that latches on the first packet).</summary>
    None,
    /// <summary>A fixed address from configuration.</summary>
    Literal,
    /// <summary>Asked from a STUN server, re-asked on every registration refresh.</summary>
    Stun,
}

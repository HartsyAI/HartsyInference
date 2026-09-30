namespace HartsyInference.PhoneLink;

/// <summary>Who placed the call; serialized by name in <see cref="CallStartMessage"/>.</summary>
public enum LinkCallDirection
{
    /// <summary>The far end called us and the gateway answered.</summary>
    Inbound,
    /// <summary>The gateway placed the call on the agent's behalf.</summary>
    Outbound,
}

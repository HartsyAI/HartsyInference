namespace HartsyInference.PhoneGateway.Sip;

/// <summary>The one-call state machine of <see cref="CallController"/>.</summary>
public enum CallState
{
    Idle,
    /// <summary>An INVITE is being answered or an outbound call is ringing; a second INVITE gets 486.</summary>
    Ringing,
    /// <summary>Media is flowing and the host has been told.</summary>
    Active,
    /// <summary>Teardown in progress.</summary>
    Ending,
}

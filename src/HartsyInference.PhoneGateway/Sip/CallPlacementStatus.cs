namespace HartsyInference.PhoneGateway.Sip;

/// <summary>How <see cref="CallController.PlaceCallAsync"/> ended.</summary>
public enum CallPlacementStatus
{
    /// <summary>The far end answered and the call is announced to the host.</summary>
    Placed,
    /// <summary>A call is already up or being set up.</summary>
    Busy,
    /// <summary>The voice host link is down.</summary>
    HostUnavailable,
    /// <summary>The destination does not match <c>sip.destinationPrefixes</c>.</summary>
    NotAllowed,
    /// <summary>The destination cannot be dialled as given (a bare number with no registrar).</summary>
    Invalid,
    /// <summary>The far end refused, did not answer, or the SIP attempt failed.</summary>
    NotAnswered,
    /// <summary>Media could not be set up, or a media thread faulted before the call was announced.</summary>
    Failed,
}

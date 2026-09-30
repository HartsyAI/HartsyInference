namespace HartsyInference.PhoneGateway.Media;

/// <summary>Which G.711 law the gateway offers, and in what order. Only PCMU and PCMA at 8 kHz are ever negotiated.</summary>
public enum AudioCodecPreference
{
    /// <summary>Offer PCMU then PCMA; the far end picks.</summary>
    Any,
    /// <summary>Offer μ-law only.</summary>
    Pcmu,
    /// <summary>Offer A-law only.</summary>
    Pcma,
}

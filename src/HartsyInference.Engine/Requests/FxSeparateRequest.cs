namespace HartsyInference.Engine.Requests;

/// <summary>Stem-separation request (Demucs): split a mix into its component stems.</summary>
public sealed record FxSeparateRequest
{
    /// <summary>The mixed audio to separate.</summary>
    public required AudioClip Audio { get; init; }

    /// <summary>Model/variant selector (e.g. "htdemucs", "htdemucs_6s", "htdemucs_ft"); null uses the default.</summary>
    public string? Model { get; init; }
}

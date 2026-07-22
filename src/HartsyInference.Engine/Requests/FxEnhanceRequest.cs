namespace HartsyInference.Engine.Requests;

/// <summary>Speech-enhancement request (Resemble-Enhance): denoise and enhance a recording.</summary>
public sealed record FxEnhanceRequest
{
    /// <summary>The audio to enhance.</summary>
    public required AudioClip Audio { get; init; }

    /// <summary>Denoise/enhance blend (lambda); pipeline default when null.</summary>
    public double? Lambd { get; init; }

    /// <summary>CFM temperature (tau); pipeline default when null.</summary>
    public double? Tau { get; init; }

    /// <summary>Sampling seed.</summary>
    public int Seed { get; init; }
}

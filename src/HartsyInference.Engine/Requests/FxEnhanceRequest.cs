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

    /// <summary>CFM function-evaluation budget. Upstream's inference app exposes 1–128 (default 64);
    /// higher is slower and generally cleaner. Null uses the pipeline default.</summary>
    public int? Nfe { get; init; }

    /// <summary>CFM ODE solver: <c>euler</c>, <c>midpoint</c> or <c>rk4</c> (upstream default midpoint).
    /// Null uses the pipeline default.</summary>
    public string? Solver { get; init; }

    /// <summary>Sampling seed.</summary>
    public int Seed { get; init; }
}

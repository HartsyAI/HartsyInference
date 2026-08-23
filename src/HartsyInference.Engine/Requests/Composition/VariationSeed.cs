namespace HartsyInference.Engine.Requests;

/// <summary>Variation-seed blending: mixes a secondary seed's noise into the primary seed to produce a controlled variation of a base image.</summary>
public sealed record VariationSeed
{
    /// <summary>The variation seed.</summary>
    public required long Seed { get; init; }

    /// <summary>Blend strength (0 = base seed only, 1 = variation seed only).</summary>
    public double Strength { get; init; }
}

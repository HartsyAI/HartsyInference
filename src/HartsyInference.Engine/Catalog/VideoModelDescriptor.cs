using HartsyInference.Engine.Recipes;

namespace HartsyInference.Engine.Catalog;

/// <summary>A video family's descriptor: its recommended defaults, what it can generate, the conditioning it applies and
/// the sampler and schedule selections it accepts.</summary>
public sealed record VideoModelDescriptor : ModelDescriptor
{
    /// <summary>The values a request left unset resolves to.</summary>
    public required VideoDefaults Defaults { get; init; }

    /// <summary>The frame counts the family can generate.</summary>
    public required VideoConstraints Constraints { get; init; }

    /// <summary>The conditioning the family applies for this variant.</summary>
    public required VideoFeatures Features { get; init; }

    /// <summary>The sampler and sigma-schedule names the family accepts.</summary>
    public required SamplingCapabilities.SamplingSupport Sampling { get; init; }
}

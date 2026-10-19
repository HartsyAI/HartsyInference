namespace HartsyInference.Core.Rope;

/// <summary>Per-axis RoPE configuration. <see cref="RopeFrequencyPlan.Create"/> validates and freezes it, so later use
/// never reads this record again.</summary>
public sealed record RopeSpec
{
    /// <summary>Channel count per axis. Each entry is non-negative and even, and a zero-width axis contributes no pairs; their sum is the head dimension.</summary>
    public required IReadOnlyList<int> AxisDims { get; init; }

    /// <summary>Base frequency shared by every axis before NTK scaling.</summary>
    public required double Theta { get; init; }

    /// <summary>Optional NTK scale per axis, one entry per axis. Larger values extend the context.</summary>
    public IReadOnlyList<double>? NtkScales { get; init; }
}

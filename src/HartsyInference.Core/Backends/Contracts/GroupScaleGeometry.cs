namespace HartsyInference.Core.Backends.Contracts;

/// <summary>The flattened geometry of a validated <see cref="Gating.GroupScaleSpec"/> over an activation of <see cref="Rows"/> rows of
/// <see cref="Width"/> elements: <see cref="GroupWidth"/> elements per gate, <see cref="Groups"/> gates per row.</summary>
public readonly record struct GroupScaleGeometry(long Rows, int Width, int GroupWidth, int Groups);

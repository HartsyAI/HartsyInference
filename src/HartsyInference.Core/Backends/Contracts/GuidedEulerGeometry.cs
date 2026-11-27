namespace HartsyInference.Core.Backends.Contracts;

/// <summary>The extents a guided Euler step runs over: every element, and the rows of the last dimension a per-row norm
/// reduces (<c>Rows · Width == Count</c>).</summary>
public readonly record struct GuidedEulerGeometry(long Count, long Rows, long Width);

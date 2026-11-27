namespace HartsyInference.Core.Backends.Norm;

/// <summary>Which axis a <see cref="NormModulateCall"/> treats as the row it normalizes and modulates.</summary>
public enum NormAxis
{
    /// <summary>The last axis: a row is one contiguous vector.</summary>
    LastDim,

    /// <summary>Axis 1 of <c>[N, C, ...]</c>: a row is the channel vector at one spatial position, its elements <c>spatial</c> apart.</summary>
    Channel,
}

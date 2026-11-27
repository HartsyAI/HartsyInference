namespace HartsyInference.Core.Backends.Norm;

/// <summary>How a modulation scale multiplies: as stored, or as a residual around identity.</summary>
public enum ScaleMode
{
    /// <summary><c>v · scale</c>.</summary>
    Raw,

    /// <summary><c>v · (1 + scale)</c>, the adaLN convention where a zero modulation leaves the value untouched.</summary>
    OnePlus,
}

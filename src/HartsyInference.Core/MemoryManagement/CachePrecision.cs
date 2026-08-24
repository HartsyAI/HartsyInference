namespace HartsyInference.Core.MemoryManagement;

/// <summary>Storage precision for the cross-step caches that are a precision trade rather than a placement one — KV caches and the Wan-Animate-2 driving cache. Halving them diverges from the reference forward, so it is never implied by streaming.</summary>
public enum CachePrecision
{
    /// <summary>Measure, and halve only where the full-precision cache would not fit.</summary>
    Auto = 0,

    /// <summary>Keep exact numerics whatever it costs; parity runs need this.</summary>
    Full = 1,

    /// <summary>Half precision everywhere it is supported, for roughly half the cache bytes.</summary>
    Half = 2,
}
